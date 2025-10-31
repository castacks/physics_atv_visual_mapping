import os
import torch
import rospkg

from physics_atv_visual_mapping.image_processing.anyloc_utils import VLAD
from physics_atv_visual_mapping.image_processing.processing_blocks.base import (
    ImageProcessingBlock,
)
from physics_atv_visual_mapping.feature_key_list import FeatureKeyList

class VLADBlock(ImageProcessingBlock):
    def __init__(self, n_clusters, cache_dir, models_dir, desc_dim=None, device="cuda"):
        cache_dir = os.path.join(models_dir, cache_dir)
        self.vlad = VLAD(n_clusters, desc_dim=desc_dim, cache_dir=cache_dir)
        self.vlad.fit(None)

        # import numpy as np
        # descs = np.load('/home/tartandriver/tartandriver_ws/loftup_dinov2s_thermal_224x224_feats.npy')
        # descs = descs.reshape(-1, descs.shape[-1])
        # # descs = descs[::2]
        # descs = torch.from_numpy(descs).cuda()
        # self.vlad = VLAD(n_clusters, desc_dim=desc_dim, cache_dir=cache_dir)
        # self.vlad.fit(descs)

    def run(self, image, intrinsics, image_orig):
        img_outs = []

        for img in image:
            _img = img.permute(1, 2, 0)
            res = self.vlad.generate_res_vec(_img.view(-1, _img.shape[-1]))
            img_out = res.abs().sum(dim=2).view(_img.shape[0], _img.shape[1], -1)
            img_out = img_out.permute(2, 0, 1)
            img_outs.append(img_out)

        img_outs = torch.stack(img_outs, dim=0)

        return img_outs, intrinsics

    @property
    def output_feature_keys(self):
        return FeatureKeyList(
            label=[f"vlad_{i}" for i in range(self.vlad.num_clusters)],
            metainfo=["vfm" for i in range(self.vlad.num_clusters)]
        )