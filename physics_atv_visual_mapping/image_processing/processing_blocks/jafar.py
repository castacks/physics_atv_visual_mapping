import os
import torch
import torchvision
import numpy as np
import math
import torchvision.transforms as T
import torch.nn.functional as F
from torch.nn.modules.utils import _pair
from types import MethodType
from physics_atv_visual_mapping.image_processing.JAFAR.src.backbone.vit_wrapper import PretrainedViTWrapper
from physics_atv_visual_mapping.image_processing.JAFAR.src.backbone.radio import RadioWrapper

from physics_atv_visual_mapping.image_processing.JAFAR.src.upsampler import JAFAR
from physics_atv_visual_mapping.feature_key_list import FeatureKeyList

from physics_atv_visual_mapping.image_processing.processing_blocks.base import ImageProcessingBlock
import cv2

class JafarBlock(ImageProcessingBlock):
    """
    Image processing block that runs clipseg on the image
    """
    def __init__(self, image_insize, device, models_dir):
        self.device = device
        self.input_size = image_insize

        # backbone_type = "vit_small_patch14_dinov2.lvd142m"
        backbone_type = "vit_small_patch14_dinov2.lvd142m"
        self.n_feats = 384
        # backbone_type = "radio_v2.5-b"
        # self.n_feats = 768

        if 'radio' not in backbone_type:
            self.model = PretrainedViTWrapper(backbone_type, norm=True).to(device).eval()
        else:
            self.model = RadioWrapper(backbone_type).to(device).eval()

        self.upsampler = JAFAR(v_dim=self.n_feats).to(device).eval()
        jafar_dir = '/home/tartandriver/tartandriver_ws/models/physics_atv_visual_mapping/jafar'
        model_path = os.path.join(jafar_dir, backbone_type + '.pth')
        
        self.upsampler.load_state_dict(torch.load(model_path, weights_only=False)["jafar"])


        self.jafar_type = backbone_type

        mean = torch.tensor([0.485, 0.456, 0.406], device=device)
        std = torch.tensor([0.229, 0.224, 0.225], device=device)
        self.transform = T.Compose(
            [
                # T.Resize(img_size),
                # T.CenterCrop(img_size),
                # T.ToTensor(),  # Convert to tensor
                T.Normalize(mean=mean, std=std),  # Normalize
            ]
        )

        ####
        self.save_samples = []
        self.save_count = 1

    def preprocess(self, img):
        assert len(img.shape) == 4, 'need to batch images'
        assert img.shape[1] == 3, 'expects channels-first'
        img = img.cuda().float()
        img = self.transform(img)
        img = torchvision.transforms.functional.resize(img,(self.input_size[1],self.input_size[0]))
        return img

    def run(self, image, intrinsics, image_orig):
        import time
        now = time.perf_counter()

        with torch.no_grad():
            # img = self.preprocess(torch.flip(image, dims=[1]))
            img = self.preprocess(image)
            lr_feats, _ = self.model(img) # 1, dim, lr_size, lr_size
            img_out = self.upsampler(img, lr_feats, self.input_size)

        # print(img_out.shape)
        ix = image.shape[3]
        dx = img_out.shape[3]
        iy = image.shape[2]
        dy = img_out.shape[2]

        # print(image.shape, img_out.shape)

        intrinsics[:, 0, 0] *= (dx/ix)
        intrinsics[:, 0, 2] *= (dx/ix)

        intrinsics[:, 1, 1] *= (dy/iy)
        intrinsics[:, 1, 2] *= (dy/iy)
        # torch.cuda.synchronize()
        # print(time.perf_counter() - now)

        # # ####img_out
        # save_samples = img_out[0,:,40:-20,:].cpu().flatten(1).numpy() #assuming 244x244
        # save_samples = save_samples.T
        # sample_ids = np.random.choice(len(save_samples), size=100)
        # save_samples = save_samples[sample_ids]
        # self.save_samples.append(save_samples)
        # if self.save_count % 10 == 0:
        #     np.save("/home/tartandriver/tartandriver_ws/loftup_dinov2s_224x224_feats", np.array(self.save_samples))
        #     print("_________________________________________", self.save_count)
        # self.save_count += 1

        return img_out, intrinsics

    @property
    def output_feature_keys(self):
        #n_layers = sum of layer.outchannels for layer in self.dino.layers
        # n_layers = sum(self.dino.dino_model.embed_dim for layer in self.dino.layers)
        n_layers = self.n_feats

        return FeatureKeyList(
            label=[f"{self.jafar_type}_{i}" for i in range(n_layers)],
            metainfo=["vfm" for i in range(n_layers)]
        )