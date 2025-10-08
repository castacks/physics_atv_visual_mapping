import os
import torch
import torchvision

import math
import torch.nn.functional as F
from torch.nn.modules.utils import _pair
from types import MethodType

from physics_atv_visual_mapping.image_processing.loftup.upsamplers import load_loftup_checkpoint, norm, unnorm
from physics_atv_visual_mapping.image_processing.loftup.featurizers import get_featurizer
from physics_atv_visual_mapping.feature_key_list import FeatureKeyList

from physics_atv_visual_mapping.image_processing.processing_blocks.base import ImageProcessingBlock
import cv2

class LoftUpBlock(ImageProcessingBlock):
    """
    Image processing block that runs clipseg on the image
    """
    def __init__(self, image_insize, device, models_dir):
        self.device = device
        self.input_size = image_insize

        featurizer_class = "dinov2" # "dinov2", "dinov2b", "dinov2s_reg", "dinov2b_reg", "clip", "siglip", "siglip2"
        torch_hub_name = "loftup_dinov2s" # "loftup_dinov2s", "loftup_dinov2b", "loftup_dinov2s_reg", "loftup_dinov2b_reg", "loftup_clip", "loftup_siglip", "loftup_siglip2"

        dino_dir = os.path.join(models_dir, "torch_hub", "facebookresearch_dinov2_main")
        loftup_dir = os.path.join(models_dir, "torch_hub", "andrehuang_loftup_main")

        self.loftup_type = torch_hub_name
        torch.hub.set_dir(os.path.join(models_dir, "torch_hub"))
        model, patch_size, dim = get_featurizer(featurizer_class, dino_dir)
        self.model = model.to('cuda')

        upsampler = torch.hub.load(loftup_dir, torch_hub_name, pretrained=True, source='local')
        self.upsampler = upsampler.to('cuda')

    def preprocess(self, img):
        assert len(img.shape) == 4, 'need to batch images'
        assert img.shape[1] == 3, 'expects channels-first'
        img = img.cuda().float()
        img = torchvision.transforms.functional.resize(img,(self.input_size[1],self.input_size[0]))
        return img

    def run(self, image, intrinsics, image_orig):
        import time
        now = time.perf_counter()

        with torch.no_grad():
            img = self.preprocess(image)
            img_out = self.model(norm(img)) # 1, dim, lr_size, lr_size
            # print(img_out.shape)
            ## Upsampling step
            # img_112 = F.interpolate(img, size=(128,128), mode='bilinear', align_corners=False)
            img_out = self.upsampler(img_out, img) # 1, dim, 224, 224
            # img_out = self.upsampler(img_out, img_112) # 1, dim, 224, 224

        # print(img_out.shape)
        ix = image.shape[3]
        dx = img_out.shape[3]
        iy = image.shape[2]
        dy = img_out.shape[2]

        intrinsics[:, 0, 0] *= (dx/ix)
        intrinsics[:, 0, 2] *= (dx/ix)

        intrinsics[:, 1, 1] *= (dy/iy)
        intrinsics[:, 1, 2] *= (dy/iy)
        # torch.cuda.synchronize()
        # print(time.perf_counter() - now)
        return img_out, intrinsics

    @property
    def output_feature_keys(self):
        #n_layers = sum of layer.outchannels for layer in self.dino.layers
        # n_layers = sum(self.dino.dino_model.embed_dim for layer in self.dino.layers)
        n_layers = 384

        return FeatureKeyList(
            label=[f"{self.loftup_type}_{i}" for i in range(n_layers)],
            metainfo=["vfm" for i in range(n_layers)]
        )