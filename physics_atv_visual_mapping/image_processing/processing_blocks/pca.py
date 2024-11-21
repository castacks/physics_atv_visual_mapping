import os
import torch

from physics_atv_visual_mapping.image_processing.processing_blocks.base import (
    ImageProcessingBlock,
)


class PCABlock(ImageProcessingBlock):
    """
    Block that applies a precomputed PCA to the image
    """

    def __init__(self, fp, models_dir, device):
        full_fp = os.path.join(models_dir, fp)
        self.pca = {
            k: v.to(device) for k, v in torch.load(full_fp, weights_only=False).items()
        }

    def run(self, image, intrinsics, image_orig):
        _pmean = self.pca["mean"].view(1, 1, -1)
        _pv = self.pca["V"].unsqueeze(0)

        # move to channels-last
        image_feats = image.flatten(start_dim=-2).permute(0, 2, 1)
        img_norm = image_feats - _pmean
        img_pca = img_norm @ _pv
        img_out = img_pca.permute(0, 2, 1).view(
            image.shape[0], _pv.shape[-1], image.shape[2], image.shape[3]
        )

        return img_out, intrinsics
