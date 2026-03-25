import os
import random

import numpy as np
import torch
import torchvision.transforms as T
from hydra.utils import instantiate
from torch.utils.tensorboard import SummaryWriter
from torchvision.transforms.functional import InterpolationMode

from utils.img import PILToTensor


def seed_worker():
    worker_seed = torch.initial_seed() % 2**32
    np.random.seed(worker_seed)
    random.seed(worker_seed)


def logger(args, base_log_dir):
    os.makedirs(base_log_dir, exist_ok=True)
    existing_versions = [
        int(d.split("_")[-1])
        for d in os.listdir(base_log_dir)
        if os.path.isdir(os.path.join(base_log_dir, d)) and d.startswith("version_")
    ]
    new_version = max(existing_versions, default=-1) + 1
    new_log_dir = os.path.join(base_log_dir, f"version_{new_version}")

    # Create the SummaryWriter with the new log directory
    writer = SummaryWriter(log_dir=new_log_dir)
    return writer, new_version, new_log_dir


def get_dataloaders(cfg, backbone, is_evaluation=False, mean=None, std=None, shuffle=True):
    """Get dataloaders for either training or evaluation.

    Args:
        cfg: Configuration object
        backbone: Backbone model for normalization parameters
        is_evaluation: If True, use evaluation dataset config, else use training dataset config
    """
    # Default ImageNet normalization values
    default_mean = [0.485, 0.456, 0.406]
    default_std = [0.229, 0.224, 0.225]

    if mean is None:
        try:
            mean = backbone.config["mean"]
        except (AttributeError, KeyError):
            mean = default_mean
            print(f"Warning: Using default mean values: {mean}")

    if std is None:
        try:
            std = backbone.config["std"]
        except (AttributeError, KeyError):
            std = default_std
            print(f"Warning: Using default std values: {std}")

    transforms = {
        "image": T.Compose(
            [
                T.Resize(cfg.img_size, interpolation=InterpolationMode.BILINEAR),
                T.CenterCrop((cfg.img_size, cfg.img_size)),
                T.ToTensor(),
                T.Normalize(mean=mean, std=std),
            ]
        ),
        "label": (
            T.Compose(
                [
                    T.Resize(cfg.target_size, interpolation=InterpolationMode.NEAREST_EXACT),
                    T.CenterCrop((cfg.target_size, cfg.target_size)),
                    PILToTensor(),
                ]
            )
            if is_evaluation
            else None
        ),
    }

    if is_evaluation:
        # For evaluation datasets
        train_dataset = instantiate(
            cfg.dataset_evaluation,
            transform=transforms["image"],
            target_transform=transforms["label"],
        )
        cfg.dataset_evaluation.split = "val"
        val_dataset = instantiate(
            cfg.dataset_evaluation,
            transform=transforms["image"],
            target_transform=transforms["label"],
        )
    else:
        # For training datasets
        train_dataset = instantiate(cfg.train_dataset, transform=transforms["image"])
        val_dataset = instantiate(cfg.val_dataset, transform=transforms["image"])

    # Create generator for reproducibility
    g = torch.Generator()
    if shuffle:
        g.manual_seed(0)

    return (
        instantiate(cfg.train_dataloader, dataset=train_dataset, generator=g),
        instantiate(cfg.val_dataloader, dataset=val_dataset, generator=g),
    )


def get_batch(batch, device):
    """Process batch and return required tensors."""
    image_batch = batch["image"].to(device)
    batch["image"] = image_batch
    return batch
