import os
import torch

from physics_atv_visual_mapping.image_processing.anyloc_utils import (
    DinoV2ExtractFeatures,
)
from physics_atv_visual_mapping.image_processing.processing_blocks.base import (
    ImageProcessingBlock,
)


class Dinov2Block(ImageProcessingBlock):
    """
    Image processing block that runs dino on the image
    """

    def __init__(
        self, dino_type, dino_layers, image_insize, desc_facet, device, models_dir
    ):
        torch.hub.set_dir(os.path.join(models_dir, "torch_hub"))

        if "dino" in dino_type:
            dino_dir = os.path.join(models_dir, "torch_hub", "facebookresearch_dinov2_main")
        elif "radio" in dino_type:
            dino_dir = os.path.join(models_dir, "torch_hub", "NVlabs_RADIO_main")

        self.dino = DinoV2ExtractFeatures(
            dino_dir,
            dino_model=dino_type,
            layers=dino_layers,
            input_size=image_insize,
            facet=desc_facet,
            device=device,
        )

    def run(self, image, intrinsics, image_orig):
        img_out = self.dino(image)

        ix = image.shape[3]
        dx = img_out.shape[3]
        iy = image.shape[2]
        dy = img_out.shape[2]

        intrinsics[:, 0, 0] *= dx / ix
        intrinsics[:, 0, 2] *= dx / ix

        intrinsics[:, 1, 1] *= dy / iy
        intrinsics[:, 1, 2] *= dy / iy

        return img_out, intrinsics
