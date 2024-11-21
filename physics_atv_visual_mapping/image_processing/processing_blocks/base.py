"""
Base class for image processing blocks
Should 
"""

import abc


class ImageProcessingBlock(abc.ABC):
    def __init__(self):
        pass

    @abc.abstractmethod
    def run(self, image, intriniscs, image_orig):
        """
        Args:
            image: [B x C x W x H] Image Tensor
            intrinsics: [B x 3 x 3] Intrinsics Tensor
            image_orig: [B x 3 x W x H] Tensor of original img (mostly for featup)
        Returns:
            feats: [B x C x W' x H'] Tensor of image features
            intrinsics_new: [B x 3 x 3] Tensor intrinsics of features
        """
        pass
