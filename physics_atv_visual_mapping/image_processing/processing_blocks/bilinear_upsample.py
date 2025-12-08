import os
import torch
import torchvision
import numpy as np
import math
import torch.nn.functional as F
from torch.nn.modules.utils import _pair
from types import MethodType

from physics_atv_visual_mapping.image_processing.loftup.upsamplers import load_loftup_checkpoint, norm, unnorm
from physics_atv_visual_mapping.image_processing.loftup.featurizers import get_featurizer
from physics_atv_visual_mapping.feature_key_list import FeatureKeyList

from physics_atv_visual_mapping.image_processing.processing_blocks.base import ImageProcessingBlock
import cv2

class BilinearUpBlock(ImageProcessingBlock):
    """
    Image processing block that runs upsamples image using bilinear interpolation
    """
    def __init__(self, image_insize, featurizer, grayscale, device, models_dir):
        self.device = device
        self.input_size = image_insize

        self.featurizer_class = featurizer # "dinov2", "dinov2b", "dinov2s_reg", "dinov2b_reg", "clip", "siglip", "siglip2"

        dino_dir = os.path.join(models_dir, "torch_hub", "facebookresearch_dinov2_main")

        model, patch_size, dim = get_featurizer(self.featurizer_class, dino_dir)
        self.model = model.to('cuda')

        self.dim = dim

        # boolean for whether to convert to grayscale before featurizing (applies for RGB images only)
        self.grayscale = grayscale

        ####
        self.save_samples = []
        self.save_count = 1

    def preprocess(self, img):
        assert len(img.shape) == 4, 'need to batch images'
        assert img.shape[1] == 3, 'expects channels-first'
        if self.grayscale:
            grayscale = torchvision.transforms.Grayscale(num_output_channels=3)
            img = grayscale(img)
        img = img.cuda().float()
        img = torchvision.transforms.functional.resize(img,(self.input_size[1],self.input_size[0]))
        return img

    def run(self, image, intrinsics, image_orig):
        with torch.no_grad():
            img = self.preprocess(image)
            img_out = self.model(img) # 1, dim, lr_size, lr_size
            # Upsampling
            img_out = F.interpolate(img_out, size=(224,224), mode='bilinear', align_corners=False)

        ix = image.shape[3]
        dx = img_out.shape[3]
        iy = image.shape[2]
        dy = img_out.shape[2]

        intrinsics[:, 0, 0] *= (dx/ix)
        intrinsics[:, 0, 2] *= (dx/ix)

        intrinsics[:, 1, 1] *= (dy/iy)
        intrinsics[:, 1, 2] *= (dy/iy)

        # save descriptors for vlad
        # save_samples = img_out[0,:,40:-20,:].cpu().flatten(1).numpy() #assuming 224x224
        # save_samples = save_samples.T
        # sample_ids = np.random.choice(len(save_samples), size=100)
        # save_samples = save_samples[sample_ids]
        # self.save_samples.append(save_samples)
        # if self.save_count % 10 == 0:
        #     np.save("/home/tartandriver/tartandriver_ws/bilinear_anythermal_thermal_224x224_feats", np.array(self.save_samples))
        #     print("_________________________________________", self.save_count)
        # self.save_count += 1

        return img_out, intrinsics

    @property
    def output_feature_keys(self):
        #n_layers = sum of layer.outchannels for layer in self.dino.layers
        # n_layers = sum(self.dino.dino_model.embed_dim for layer in self.dino.layers)
        n_layers = self.dim

        return FeatureKeyList(
            label=[f"{self.featurizer_class}_{i}" for i in range(n_layers)],
            metainfo=["vfm" for i in range(n_layers)]
        )