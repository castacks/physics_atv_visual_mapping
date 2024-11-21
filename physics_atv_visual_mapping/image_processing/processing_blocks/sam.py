import os
import cv2
import torch
import rospkg
import numpy as np
from segment_anything import SamPredictor, sam_model_registry

from physics_atv_visual_mapping.image_processing.processing_blocks.base import (
    ImageProcessingBlock,
)


class SAMBlock(ImageProcessingBlock):
    """
    Image processing block that runs Segment Anything on the image
    """

    def __init__(self, sam_type, sam_path, image_insize, models_dir, device):
        sam_path = os.path.join(models_dir, sam_path)
        self.sam = sam_model_registry[sam_type](checkpoint=sam_path).to(device)
        self.predictor = SamPredictor(self.sam)

        self.image_insize = image_insize
        self.yimax = int(64 * (min(self.image_insize) / max(self.image_insize)))
        self.outsize = torch.tensor([64, self.yimax])
        self.nfeats = 256
        self.device = device

    def run(self, image, intrinsics, image_orig):
        sam_res = []
        for img in image:
            img = (img.permute(1, 2, 0).cpu().numpy() * 255).astype(np.uint8)

            img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
            img = cv2.resize(img, self.image_insize, interpolation=cv2.INTER_AREA)

            self.predictor.set_image(img)
            sres = (
                self.predictor.get_image_embedding()[0, :, : self.yimax, :]
                .to(self.device)
                .float()
            )

            sam_res.append(sres)

        img_out = torch.stack(sam_res, dim=0)

        ix = image.shape[3]
        dx = img_out.shape[3]
        iy = image.shape[2]
        dy = img_out.shape[2]

        intrinsics[:, 0, 0] *= dx / ix
        intrinsics[:, 0, 2] *= dx / ix

        intrinsics[:, 1, 1] *= dy / iy
        intrinsics[:, 1, 2] *= dy / iy

        return img_out, intrinsics
