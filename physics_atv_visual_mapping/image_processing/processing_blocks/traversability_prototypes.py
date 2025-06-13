import os
import torch

from physics_atv_visual_mapping.image_processing.processing_blocks.base import (
    ImageProcessingBlock,
)


class TraversabilityPrototypesBlock(ImageProcessingBlock):
    """
    Block that applies a precomputed PCA to the image
    """

    def __init__(self, fp, models_dir, device):
        full_fp = os.path.join(models_dir, fp)
        prototypes = torch.load(full_fp)
        
        self.obstacle_keys = []
        self.obstacle_ptypes = []
        self.nonobstacle_keys = []
        self.nonobstacle_ptypes = []

        for pdata in prototypes["obstacle"]:
            self.obstacle_keys.append(pdata["label"])
            self.obstacle_ptypes.append(pdata["ptype"])

        for pdata in prototypes["nonobstacle"]:
            self.nonobstacle_keys.append(pdata["label"])
            self.nonobstacle_ptypes.append(pdata["ptype"])

        self.obstacle_ptypes = torch.stack(self.obstacle_ptypes, dim=0).to(device)
        self.nonobstacle_ptypes = torch.stack(self.nonobstacle_ptypes, dim=0).to(device)

    def run(self, image, intrinsics, image_orig):
        pos_csim, neg_csim = self.get_prototype_scores(image)

        img_out = torch.cat([pos_csim, neg_csim], dim=1)

        return img_out, intrinsics

    def get_prototype_scores(self, feat_img):
        """
        Args:
            feat_img: [B x C x W x H] Tensor of image data
        Returns:
            pos_csim: [B x P1 x W x H] Tensor of cosine similarity to this block's obstacle ptypes
                (P1 = num obstacle prototypes)
            neg_csim: [B x P2 x W x H] same as pos_csim but for the negative prototypes
        """
        pos_csim = get_feat_img_prototype_cosine_sim(feat_img, self.obstacle_ptypes)
        neg_csim = get_feat_img_prototype_cosine_sim(feat_img, self.nonobstacle_ptypes)

        return pos_csim, neg_csim

def get_feat_img_prototype_cosine_sim(feat_img, prototypes):
    """
    Args:
        feat_img: [B x C x W x H] Tensor of image data
        prototypes: [P x C] Tensor of prototypes
    Returns:
        csim: [B x P x W x H] Tensor of each pixel's cosine similarity to each prototype
    """
    _fimg = feat_img.view(feat_img.shape[0], 1, *feat_img.shape[1:]) #[B x P x C x W x H]
    _ptypes = prototypes.view(1, *prototypes.shape, 1, 1) #[B x P x C x W x H]

    _proj = (_fimg * _ptypes).sum(dim=2) #[B x P x W x H]
    _norm = torch.linalg.norm(_fimg, dim=2) * torch.linalg.norm(_ptypes, dim=2)

    csim = _proj / _norm

    return csim
