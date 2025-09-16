from os.path import join
import gc

import hydra
import matplotlib.pyplot as plt
import numpy as np
import cv2
import pytorch_lightning as pl
import torch
import torch.nn.functional as F
import torchvision.transforms as T
from torchvision.transforms import InterpolationMode

from omegaconf import DictConfig
from omegaconf import OmegaConf
from pytorch_lightning import Trainer
from pytorch_lightning import seed_everything
from pytorch_lightning.loggers import TensorBoardLogger
from torch.utils.data import DataLoader
from torchmetrics.classification import Accuracy, JaccardIndex

from upsamplers import load_lift_checkpoints, norm, unnorm, get_upsampler, load_upsampler_weights
from datasets import get_dataset, EmbeddingFile
from featurizers import get_featurizer
from utils import ADE20K_150_CATEGORIES, pca, ToTensorWithoutScaling


class SemSegEvaluator(pl.LightningModule):
    def __init__(self, n_dim, upsampler_type=None, upsampler_path=None, cfg=None, model=None):
        super().__init__()
        self.n_dim = n_dim
        self.lr = cfg.lr
        self.dataset = cfg.dataset
        self.upsampler_type = upsampler_type
        self.model = model
        self.color_map = {category["id"]: np.array(category["color"]) / 255.0 for category in ADE20K_150_CATEGORIES}
        self.name = f"{self.dataset}_{cfg.load_size}_{cfg.upsampler_type}_{cfg.model_type}"
        self.guidance_res = cfg.guidance_res
        self.train_small_res = cfg.train_small_res
        self.upsampler_path = upsampler_path
        self.visualize = cfg.visualize
        
        if cfg.dataset == 'cocostuff':
            n_classes = 27
        elif cfg.dataset == "cityscapes":
            n_classes = 19
        self.n_classes = n_classes
        self.classifier_path = upsampler_path[:-5] + f'{cfg.dataset}_{cfg.load_size}_classifier.ckpt'
        self.use_pretrained_classifier = cfg.use_pretrained_classifier

        self.classifier = torch.nn.Conv2d(n_dim, n_classes, 1)
        if self.use_pretrained_classifier: # Load a pre-trained classifier and freeze it
            ckpt_weight = torch.load(cfg.pretrained_classifier_path)
            self.classifier.load_state_dict(ckpt_weight)
            for param in self.classifier.parameters():
                param.requires_grad = False
            print(f"Loaded classifier from {cfg.pretrained_classifier_path} and froze it")

        if cfg.model_type == "dinov2":
            final_size = 16
            kernel_size = 14
        else:
            final_size = 14
            kernel_size = 16
        if upsampler_type == "lift":
            upsampler = load_lift_checkpoints("/path/to/your/lift.ckpt")
            for param in upsampler.parameters():
                param.requires_grad = False
        elif upsampler_type == "featup":
            upsampler = torch.hub.load("mhamilton723/FeatUp", cfg.model_type, use_norm=False).upsampler.to("cuda")
            for param in upsampler.parameters():
                param.requires_grad = False
        elif upsampler_type != "no": ## i.e., loftup, bilinear, etc.
            upsampler = get_upsampler(upsampler_type, n_dim, lr_size=final_size, cfg=cfg)
            if upsampler_type != "bilinear": # contains trainable weights
                upsampler = load_upsampler_weights(upsampler, upsampler_path, n_dim)
        elif upsampler_type == "no": ## i.e., no upsampler
            upsampler = None
        else:
            raise ValueError(f"Upsampler {upsampler_type} not implemented")

        self.upsampler = upsampler

        self.linear_acc_metric = Accuracy(num_classes=n_classes, task="multiclass")
        self.linear_acc_buff = self.register_buffer("linear_acc", torch.tensor(0.0))
        self.linear_iou_metric = JaccardIndex(num_classes=n_classes, task="multiclass")
        self.linear_iou_buff = self.register_buffer("linear_iou", torch.tensor(0.0))

        self.ce = torch.nn.CrossEntropyLoss()

    def training_step(self, batch, batch_idx):
        if self.model is not None:
            imgs = batch["img"].cuda()
            label = batch["label"]
            feats = self.model(imgs)
        else:
            imgs, feats, label = batch
        label = label.to(torch.int32)
        label[label == 255] = -1 
        label = label.float()
        if self.upsampler is not None:
            if self.guidance_res != 224:
                guidance_imgs = F.interpolate(imgs, size=(self.guidance_res, self.guidance_res), mode='bilinear')
            else:
                guidance_imgs = imgs
            feats = self.upsampler(feats, guidance_imgs)
        b, c, h, w = feats.shape
        

        if self.train_small_res: ## Consistent with FeatUp's setup to make labels' size the same as feats; use nearest interpolation
            labels = F.interpolate(
                label.unsqueeze(1),
                size=(h, w), mode='nearest').to(torch.int64)
        else: ## The common practice is actually to make feats' size the same as labels, so labels are the same size as images
            labels = F.interpolate(
                label.unsqueeze(1),
                size=(imgs.shape[2], imgs.shape[3]), mode='nearest').to(torch.int64)
        h, w = labels.shape[2], labels.shape[3]

        linear_preds = self.classifier(feats)

        ## Interpolate the linear_preds
        if not self.train_small_res:
            linear_preds = F.interpolate(
                linear_preds.float(),
                size=(imgs.shape[2], imgs.shape[3]), mode='nearest')


        flat_labels = labels.permute(0, 2, 3, 1).reshape(b * h * w)
        flat_linear_preds = linear_preds.permute(0, 2, 3, 1).reshape(b * h * w, -1)

        selected = flat_labels > -1
        linear_loss = self.ce(
                flat_linear_preds[selected],
                flat_labels[selected])
        loss = linear_loss
        self.log("linear_loss", linear_loss)
        self.log("loss", loss)


        if self.global_step % 100 == 0:
            gc.collect()
            torch.cuda.empty_cache()

        return loss

    def validation_step(self, batch, batch_idx):
        with torch.no_grad():
            if self.model is not None:
                imgs = batch["img"].cuda()
                label = batch["label"]
                feats = self.model(imgs)
            else:
                imgs, feats, label = batch
            label = label.to(torch.int32)
            label[label == 255] = -1
            lr_feats = feats.clone()
            
            if self.upsampler is not None:
                if self.guidance_res != 224:
                    guidance_imgs = F.interpolate(imgs, size=(self.guidance_res, self.guidance_res), mode='bilinear')
                else:
                    guidance_imgs = imgs
                feats = self.upsampler(feats, guidance_imgs)

            linear_preds = self.classifier(feats).argmax(1, keepdim=True)

            b, h, w = label.shape
            flat_labels = label.flatten()
            selected = flat_labels > -1
            flat_labels = flat_labels[selected]
            flat_linear_preds = F.interpolate(linear_preds.float(), (h, w), mode='nearest').to(torch.int64).flatten()[selected]
            self.linear_acc_metric.update(flat_linear_preds, flat_labels)
            self.linear_iou_metric.update(flat_linear_preds, flat_labels)
            if self.visualize:
                # if batch_idx < 40:
                    unnorm_imgs = unnorm(imgs)
                    # Use PCA on lr_feats to project feats
                    [red_lr_feats], fit_pca = pca([lr_feats[0].unsqueeze(0)])
                    [red_hr_feats], _ = pca([feats[0].unsqueeze(0)], fit_pca=fit_pca)
                    self.visualize_results(unnorm_imgs[0].cpu().permute(1, 2, 0).numpy(), red_hr_feats[0].cpu().numpy(), linear_preds[0].cpu().numpy().squeeze(), label[0].cpu().numpy(), batch_idx, name=self.name)
        return None
    
    def visualize_results(self, original_image, feats, seg_pred, gt_sem_seg, img_idx, name=""):
        seg_pred_rgb = np.zeros((*seg_pred.shape, 3), dtype=np.float32)
        gt_sem_seg_rgb = np.zeros((*gt_sem_seg.shape, 3), dtype=np.float32)

        for class_id, color in self.color_map.items():
            seg_pred_rgb[seg_pred == class_id] = color
            gt_sem_seg_rgb[gt_sem_seg == class_id] = color

        pca_feats = feats.transpose(1, 2, 0)

        # Overlay predictions and ground truth on the original image
        alpha = 0.8  # Transparency for overlay
        # Ensure the shapes of the overlays are the same
        seg_pred_rgb = cv2.resize(seg_pred_rgb, (original_image.shape[1], original_image.shape[0]))
        gt_sem_seg_rgb = cv2.resize(gt_sem_seg_rgb, (original_image.shape[1], original_image.shape[0]))
        seg_pred_overlay = (1 - alpha) * original_image + alpha * seg_pred_rgb
        gt_overlay = (1 - alpha) * original_image + alpha * gt_sem_seg_rgb

        # Plot results
        plt.figure(figsize=(20, 5))

        plt.subplot(1, 4, 1)
        plt.imshow(original_image)
        plt.title("Original Image")
        plt.axis("off")

        plt.subplot(1, 4, 2)
        plt.imshow(pca_feats)
        plt.title("Feature Visualization (PCA)")
        plt.axis("off")

        plt.subplot(1, 4, 3)
        plt.imshow(seg_pred_overlay)
        plt.title("Segmentation Prediction")
        plt.axis("off")

        plt.subplot(1, 4, 4)
        plt.imshow(gt_overlay)
        plt.title("Ground Truth Overlay")
        plt.axis("off")

        # plt.tight_layout()
        plt.savefig(f"visualization/{name}_seg_results_{img_idx}.png", bbox_inches="tight")
        plt.close()

    def on_validation_epoch_end(self):
        self.linear_acc = self.linear_acc_metric.compute()
        self.linear_iou = self.linear_iou_metric.compute()
        print('Validation:')
        print('Linear Accuracy:', self.linear_acc)
        print('Linear mIoU:', self.linear_iou)
        # Adding to logger for tensorboard
        writer = self.logger.experiment
        writer.add_scalar('Linear Accuracy', self.linear_acc, self.global_step)
        writer.add_scalar('Linear mIoU', self.linear_iou, self.global_step)
        self.linear_acc_metric.reset()
        self.linear_iou_metric.reset()
        
        if self.classifier_path is not None and (not self.use_pretrained_classifier):
            torch.save(self.classifier.state_dict(), self.classifier_path)
            print(f"Saved classifier to {self.classifier_path}")

    def configure_optimizers(self):
        return torch.optim.NAdam(self.classifier.parameters(), lr=self.lr)


@hydra.main(config_path="configs", config_name="eval_seg.yaml")
def my_app(cfg: DictConfig) -> None:
    print(OmegaConf.to_yaml(cfg))
    print(cfg.output_root)
    seed_everything(seed=0, workers=True)

    log_dir = join(cfg.output_root, f"logs/eval/semseg")
    name = (f"{cfg.model_type}_{cfg.upsampler_type}_"
            f"{cfg.dataset}_{cfg.load_size}"
            )
    upsampler_path = cfg.upsampler_path

    emb_root = join(cfg.pytorch_data_dir, cfg.dataset, "embedding", cfg.model_type)

    
    if cfg.image_train:
        ## Standard data loading. Will load images and transform them on the fly.
        ## Can be slow if evaluating multiple times. But might be the only option if the dataset is too large to fit in memory.
        load_size = cfg.load_size
        transform = T.Compose([
            T.Resize(load_size, InterpolationMode.BILINEAR),
            T.CenterCrop(load_size),
            T.ToTensor(),
            norm])
        

        target_transforms = T.Compose([
            T.Resize(load_size, InterpolationMode.NEAREST),
            T.CenterCrop(load_size),
            ToTensorWithoutScaling()
        ])

        train_dataset = get_dataset(
            cfg.pytorch_data_dir,
            cfg.dataset,
            "train",
            transform=transform,
            target_transform=target_transforms,
            include_labels=True)

        train_loader = DataLoader(
            train_dataset, cfg.batch_size, shuffle=True, num_workers=cfg.num_workers)
        
        val_dataset = get_dataset(
                cfg.pytorch_data_dir,
                cfg.dataset,
                "val",
                transform=transform,
                target_transform=target_transforms,
                include_labels=True)

        val_loader = DataLoader(
                val_dataset, cfg.batch_size, shuffle=False, num_workers=cfg.num_workers)
    else: 
        ## Extracting embeddings beforehand. Will save a lot of time during evaluation.
        train_dataset = EmbeddingFile(join(emb_root, "train", f"embeddings_{cfg.model_type}.npz"), loading_imgs=cfg.loading_imgs)
        train_loader = DataLoader(train_dataset, cfg.batch_size, shuffle=True, num_workers=cfg.num_workers)

        val_dataset = EmbeddingFile(join(emb_root, "val", f"embeddings_{cfg.model_type}.npz"), loading_imgs=cfg.loading_imgs)
        val_loader = DataLoader(val_dataset, cfg.batch_size, shuffle=True, num_workers=cfg.num_workers)

    if cfg.image_train:
        model, patch_size, dim = get_featurizer(cfg.model_type, cfg.activation_type, num_classes=1000)
        model.eval()
        for param in model.parameters():
            param.requires_grad = False
    else:
        _, _, dim = get_featurizer(cfg.model_type, cfg.activation_type, num_classes=1000)
        

    evaluator = SemSegEvaluator(dim, upsampler_type=cfg.upsampler_type, upsampler_path=upsampler_path, cfg=cfg, model=model)
    tb_logger = TensorBoardLogger(log_dir, default_hp_metric=False)

    trainer = Trainer(
        accelerator='gpu',
        strategy='ddp',
        devices=cfg.num_gpus,
        max_epochs=cfg.epochs,
        logger=tb_logger,
        log_every_n_steps=100,
        reload_dataloaders_every_n_epochs=1,
        check_val_every_n_epoch=1,
    )

    if cfg.use_pretrained_classifier: # Directly run validation
        trainer.validate(evaluator, val_loader)
    else:
        trainer.fit(evaluator, train_loader, val_loader)
    
    result = {
            "Linear Accuracy": float(evaluator.linear_acc),
            "Linear mIoU": float(evaluator.linear_iou),
            "Model": cfg.model_type
        }
    print(result)


if __name__ == "__main__":
    my_app()
