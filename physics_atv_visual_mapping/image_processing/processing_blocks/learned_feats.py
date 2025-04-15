import os
import torch

from physics_atv_visual_mapping.image_processing.processing_blocks.base import (
    ImageProcessingBlock,
)

class LinearNN(torch.nn.Module):
    def __init__(self, insize, outsize):
        super(LinearNN, self).__init__()
        self.linear = torch.nn.Linear(insize, outsize)

    def forward(self, x):
        return self.linear.forward(x)

class LearnedFeaturizerBlock(ImageProcessingBlock):
    """
    Block that applies a precomputed PCA to the image
    """
    def __init__(self, models_dir, device, insize=-1, outsize=-1, fp=None):
        if fp is None:
            self.net = LinearNN(insize, outsize).to(device)
        else:
            full_fp = os.path.join(models_dir, fp)
            self.net = torch.load(full_fp)

    def run(self, image, intrinsics, image_orig):
        # move to channels-last
        image_feats = image.flatten(start_dim=-2).permute(0, 2, 1)
        img_net = self.net.forward(image_feats)
        img_out = img_net.permute(0, 2, 1).view(
            image.shape[0], -1, image.shape[2], image.shape[3]
        )

        return img_out, intrinsics
