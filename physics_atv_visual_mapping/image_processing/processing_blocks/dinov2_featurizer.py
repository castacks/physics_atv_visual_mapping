from physics_atv_visual_mapping.image_processing.loftup.featurizers import get_featurizer
from physics_atv_visual_mapping.image_processing.processing_blocks.base import ImageProcessingBlock
from physics_atv_visual_mapping.feature_key_list import FeatureKeyList

import os
import torch
import torchvision
import numpy as np

class DinoV2FeaturizerBlock(ImageProcessingBlock):
    """
    Image processing block that outputs DINOv2 patch features
    """

    def __init__(self, model_type, input_size, grayscale, device, models_dir):
        # load model
        dino_dir = os.path.join(models_dir, "torch_hub", "facebookresearch_dinov2_main")
        model, patch_size, dim = get_featurizer(model_type, dino_dir)
        self.model = model.to('cuda')
        self.input_size = input_size
        self.dim = dim
        self.model_type = model_type
        # boolean for whether to convert to grayscale before featurizing (applies for RGB images only)
        self.grayscale = grayscale

        # for saving descriptors for fitting VLAD
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
            img_out = self.model(img)

        ix = image.shape[3]
        dx = img_out.shape[3]
        iy = image.shape[2]
        dy = img_out.shape[2]

        intrinsics[:, 0, 0] *= (dx/ix)
        intrinsics[:, 0, 2] *= (dx/ix)

        intrinsics[:, 1, 1] *= (dy/iy)
        intrinsics[:, 1, 2] *= (dy/iy)

        # save descriptors for vlad
        # save_samples = img_out[0,:,2:-1,:].cpu().flatten(1).numpy() #assuming 16x16
        # save_samples = save_samples.T
        # np.random.seed(42)
        # sample_ids = np.random.choice(len(save_samples), size=100)
        # save_samples = save_samples[sample_ids]
        # self.save_samples.append(save_samples)
        # if self.save_count % 10 == 0:
        #     np.save("/home/tartandriver/tartandriver_ws/dinov2b_rgb_224x224_feats", np.array(self.save_samples))
        #     print("_________________________________________", self.save_count)
        # self.save_count += 1
        
        return img_out, intrinsics
    
    @property
    def output_feature_keys(self):
        return FeatureKeyList(
            label=[f"{self.model_type}_{i}" for i in range(self.dim)],
            metainfo=["vfm" for i in range(self.dim)]
        )