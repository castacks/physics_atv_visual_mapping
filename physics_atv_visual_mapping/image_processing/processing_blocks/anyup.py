import os
import torch
import torchvision
import numpy as np
import torch.nn.functional as F
from types import MethodType

from physics_atv_visual_mapping.feature_key_list import FeatureKeyList
from physics_atv_visual_mapping.image_processing.processing_blocks.base import ImageProcessingBlock

class AnyUpBlock(ImageProcessingBlock):
    """
    Image processing block that upsamples vision foundation model features using AnyUp.
    """
    def __init__(self, image_outsize, device, models_dir=None):
        self.device = device
        # self.input_size = image_insize
        
        # Define patch size required by anyup (adjust if your VFM uses a different patch size, e.g., 14)
        self.image_outsize = image_outsize 
        
        torch.hub.set_dir(os.path.join(models_dir, "torch_hub"))
        self.upsampler = torch.hub.load(
            'wimmerth/anyup', 
            'anyup_multi_backbone', 
            use_natten=False
        ).to(self.device).eval()

    # def preprocess(self, img):
    #     assert len(img.shape) == 4, 'need to batch images'
    #     assert img.shape[1] == 3, 'expects channels-first'
    #     img = img.to(self.device).float()
    #     img = torchvision.transforms.functional.resize(img, (self.input_size[1], self.input_size[0]))
    #     return img

    def run(self, image, intrinsics, image_orig):
        """
        Args:
            image: Input high-resolution image tensor (B, C, H, W)
            intrinsics: Camera intrinsics matrix tensor
            image_orig: Original un-preprocessed image reference
        """
        img_ix, img_iy = image.shape[3], image.shape[2]
        
        with torch.no_grad():
            # img_input = self.preprocess(image)
            feat_out = self.upsampler(image_orig, image, self.image_outsize)
            
        # feat_out_processed = feat_out.squeeze(0).permute(1, 2, 0).cpu().numpy()

        # Dynamic intrinsics scaling based on the new upsampled feature dimensions
        # Assuming feat_out matches img_input dimensions, or tracking actual output shape:
        dy, dx = feat_out.shape[2], feat_out.shape[3]

        intrinsics[:, 0, 0] *= (dx / img_ix)
        intrinsics[:, 0, 2] *= (dx / img_ix)
        intrinsics[:, 1, 1] *= (dy / img_iy)
        intrinsics[:, 1, 2] *= (dy / img_iy)

        return feat_out, intrinsics

    @property
    def output_feature_keys(self):
        # Update this count depending on your specific target backbone embed dim (e.g., 384 for DINOv2-S)
        n_layers = 384 

        return FeatureKeyList(
            label=[f"anyup_{i}" for i in range(n_layers)],
            metainfo=["vfm" for i in range(n_layers)]
        )