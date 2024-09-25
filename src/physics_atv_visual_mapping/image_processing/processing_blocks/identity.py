import os
import torch
import rospkg

from physics_atv_visual_mapping.image_processing.processing_blocks.base import ImageProcessingBlock

class IdentityBlock(ImageProcessingBlock):
    """
    Image processing block that just passes through the original image
    """
    def __init__(self, device):
        ...

    def run(self, image, intrinsics, image_orig):
        return image.clone(), intrinsics.clone()
