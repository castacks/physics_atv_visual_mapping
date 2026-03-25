import datetime
import os
from pathlib import Path

import hydra
import numpy as np
import torch
import torch.nn.functional as F
import torch.utils.checkpoint as checkpoint
import torchvision.transforms as T
from hydra.core.hydra_config import HydraConfig
from hydra.utils import instantiate, log
from omegaconf import DictConfig, OmegaConf
from rich import print
from rich.console import Console
from rich.syntax import Syntax
from torch import autocast
from tqdm import tqdm

from utils.training import get_batch, get_dataloaders, logger

FREQ = 100


def add_blur_noise(x):
    if torch.rand(1) < 0.5:
        x = T.GaussianBlur(5, sigma=1.0)(x)
    if torch.rand(1) < 0.3:
        x += 0.02 * torch.randn_like(x)
    return x.clamp(0, 1)


def round_to_nearest_multiple(value, multiple=14):
    return multiple * round(value / multiple)


def backbone_feats(cfg, image_batch, backbone):
    _, _, H, W = image_batch.shape  # Get original height and width
    with torch.no_grad():
        hr_patch_tokens, _ = backbone(image_batch)
        # Downscale
        if cfg.ratio == "fixed":
            downscale_factor = 0.5
        else:
            downscale_factor = np.random.uniform(0.25, 0.5)
        new_H = round_to_nearest_multiple(H * downscale_factor, backbone.patch_size)
        new_W = round_to_nearest_multiple(W * downscale_factor, backbone.patch_size)
        low_res_batch = F.interpolate(image_batch, size=(new_H, new_W), mode="bilinear")
        lr_patch_tokens, _ = backbone(low_res_batch)
    return hr_patch_tokens, lr_patch_tokens


@hydra.main(config_path="config", config_name="base")
def trainer(cfg: DictConfig):
    yaml_syntax = Syntax(OmegaConf.to_yaml(cfg), "yaml", theme="monokai", line_numbers=True)
    print(yaml_syntax)
    seed = 0
    print(f"Seed: {seed}")
    torch.manual_seed(seed)

    # ============ Logger ============ #
    log_dir = HydraConfig.get().runtime.output_dir
    writer, _, new_log_dir = logger(cfg, log_dir)

    terminal_console = Console()  # Terminal output
    file_name = f"train.log"
    file_console = Console(
        file=open(file_name, "w"),
    )

    def log_print(*args, **kwargs):
        """Log to both terminal and file with immediate flushing"""
        # Print to terminal
        terminal_console.print(*args, **kwargs)
        # Print to file and flush immediately
        file_console.print(*args, **kwargs)
        file_console.file.flush()  # Force immediate write to disk

        # Start logging

    timestamp = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    log_print(f"\n[bold blue]{'='*50}[/bold blue]")
    log_print(f"[bold blue]Starting at {timestamp}[/bold blue]")
    log_print(f"[bold green]Configuration:[/bold green]")
    log_print(OmegaConf.to_yaml(cfg))

    # ============ Load Backbones ============ #
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    backbone = instantiate(cfg.backbone)
    backbone = backbone.to(device)

    log_print(f"[bold yellow]Using device: {device}[/bold yellow]")
    log_print(f"\n[bold cyan]Image size: {cfg.img_size}[/bold cyan]")

    # ============ Load Student Upsamplers ============ #
    jafar = instantiate(cfg.model)
    jafar.cuda()
    jafar.train()

    # ============ preparing Datasets and Dataloaders ============ #
    train_dataloader, _ = get_dataloaders(cfg, backbone, is_evaluation=False)
    log_print(f"[bold cyan]Train Dataset size: {len(train_dataloader.dataset)}[/bold cyan]")

    # ============ Get training criterion ================= #
    criterion = instantiate(cfg.loss, dim=backbone.embed_dim)
    epoch_start = 0

    # ============ preparing Optimizers ============ #
    # Jafar
    all_params = []
    all_params.extend(list(jafar.parameters()))
    print("Number of parameters: ", sum(p.numel() for p in all_params))
    optimizer_jafar = instantiate(cfg.optimizer, params=all_params)

    total_batches = cfg.max_steps
    checkpoint_interval = int(total_batches * 0.25)

    # Calculate total training steps
    total_epochs = cfg.epochs
    total_steps = total_epochs * total_batches

    # Loop
    for epoch in range(epoch_start, cfg.epochs):
        # Training loop
        for batch_idx, batch in enumerate(
            tqdm(
                train_dataloader,
                desc=f"Epoch {epoch}",
            )
        ):
            current_step = epoch * len(train_dataloader) + batch_idx
            overall_progress = (current_step / total_steps) * 100

            batch = get_batch(batch, device)
            image_batch = batch["image"]

            with autocast(device_type="cuda", enabled=cfg.bfloat16, dtype=torch.bfloat16):
                # Helper function for checkpointed/non-checkpointed forward
                def run_model(model, inputs, **kwargs):
                    return model(*inputs, **kwargs)

                # ============ Extract Backbone Features ============ #
                with torch.no_grad():
                    hr_feats, lr_feats = backbone_feats(cfg, image_batch, backbone)
                _, _, h, w = hr_feats.shape

                # ============ Feature Prediction ============ #
                image_batch = F.interpolate(image_batch, scale_factor=0.5, mode="bilinear")
                jafar_hr_feats = run_model(jafar, [image_batch, lr_feats, (h, w)])

                # ============ Loss JAFAR ============ #
                loss = {"jafar_hr": 0.0}
                loss["jafar_hr"] = criterion(jafar_hr_feats, hr_feats)["total"]

            loss_jafar = loss["jafar_hr"]

            optimizer_jafar.zero_grad()
            loss_jafar.backward()
            optimizer_jafar.step()

            # Optional: Update tqdm with loss information
            if batch_idx % FREQ == 0:
                # Log all loss components to tensorboard
                for loss_name, loss_value in loss.items():
                    if loss_value != 0:
                        writer.add_scalar(
                            f"Loss/{loss_name}",
                            loss_value.item(),
                            current_step,
                        )

                # Log learning rates
                writer.add_scalar(
                    "Learning Rate JAFAR",
                    optimizer_jafar.param_groups[0]["lr"],
                    current_step,
                )

                # Build concise log message
                loss_str = " | ".join([f"{k}: {v.item():.4f}" for k, v in loss.items() if v != 0])
                log_print(
                    f"Epoch={epoch}/{total_epochs} | "
                    f"Batch={batch_idx}/{len(train_dataloader)} | "
                    f"Progress: {overall_progress:.1f}% | "
                    f"{loss_str}"
                )

            # Save checkpoint at every 10% interval
            if (batch_idx % checkpoint_interval == 0 and batch_idx != 0) or (current_step >= cfg.max_steps):
                checkpoint_path = os.path.join(new_log_dir, f"model_{current_step}steps.pth")

                save_dict = {
                    "optimizer_jafar": optimizer_jafar.state_dict(),
                    "epoch": epoch,
                    "cfg": cfg,
                    "jafar": jafar.state_dict(),
                }

                torch.save(save_dict, checkpoint_path)
                log_print(f"Saved checkpoint: {checkpoint_path}")

                if current_step >= cfg.max_steps:
                    break

            if cfg.sanity:
                break

        writer.flush()
        file_console.file.close()


if __name__ == "__main__":
    trainer()
