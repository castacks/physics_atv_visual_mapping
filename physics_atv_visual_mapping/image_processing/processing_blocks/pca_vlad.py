import os
import torch

from physics_atv_visual_mapping.image_processing.processing_blocks.base import (
    ImageProcessingBlock,
)
from physics_atv_visual_mapping.image_processing.processing_blocks.pca import PCABlock
from physics_atv_visual_mapping.image_processing.processing_blocks.vlad import VLADBlock
from physics_atv_visual_mapping.feature_key_list import FeatureKeyList

class PCAVLADBlock(ImageProcessingBlock):
    """
    Block that does both PCA and VLAD processing
    """

    def __init__(self, pca_fp, vlad_n_clusters, vlad_cache_dir, models_dir, device):
        self.pca_block = PCABlock(fp=pca_fp, models_dir=models_dir, device=device)
        self.vlad_block = VLADBlock(
            n_clusters=vlad_n_clusters,
            cache_dir=vlad_cache_dir,
            models_dir=models_dir,
            device=device,
        )
        self.device = device

    def run(self, image, intrinsics, image_orig):
        pca_image, pca_intrinsics = self.pca_block.run(image, intrinsics, image_orig)
        vlad_image, _ = self.vlad_block.run(image, intrinsics, image_orig)

        res_image = torch.cat([pca_image, vlad_image], dim=1)

        return res_image, pca_intrinsics
    
    @property
    def feature_key_list(self):
        pca_dim = self.pca_block['V'].shape[-1]
        vlad_dim = self.vlad_block.num_clusters

        label = [f"pca_{i}" for i in range(pca_dim)] + [f"vlad_{i}" for i in range(vlad_dim)]
        metadata = ["vfm"] * (pca_dim + vlad_dim)

        return FeatureKeyList(label=label, metadata=metadata)