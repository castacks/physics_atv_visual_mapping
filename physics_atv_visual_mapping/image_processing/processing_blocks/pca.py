import os
import torch

from physics_atv_visual_mapping.image_processing.processing_blocks.base import (
    ImageProcessingBlock,
)
from physics_atv_visual_mapping.feature_key_list import FeatureKeyList

class PCABlock(ImageProcessingBlock):
    """
    Block that applies a precomputed PCA to the image
    """

    def __init__(self, fp, models_dir, device):
        full_fp = os.path.join(models_dir, fp)
        pca = torch.load(full_fp, weights_only=False)

        if "base_label" in pca.keys():
            self.base_label = pca["base_label"]
            self.base_metainfo = pca["base_metainfo"]
        #keep backward compatability with old pcas
        else:
            self.base_label = "pca"
            self.base_metainfo = "vfm"

        self.pca = {
            "mean": pca["mean"].to(device),
            "V": pca["V"].to(device),
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

    @property
    def output_feature_keys(self):
        return FeatureKeyList(
            label=[f"{self.base_label}_{i}" for i in range(self.pca["V"].shape[-1])],
            metainfo=["vfm" for i in range(self.pca["V"].shape[-1])]
        )