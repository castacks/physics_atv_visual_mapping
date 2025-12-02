import os
import torch
import torchvision.transforms.functional as F

from transformers import AutoModel

from physics_atv_visual_mapping.image_processing.processing_blocks.base import (
    ImageProcessingBlock,
)
from physics_atv_visual_mapping.feature_key_list import FeatureKeyList
from physics_atv_visual_mapping.utils import load_ontology

class Talk2DinoSegBlock(ImageProcessingBlock):
    """
    Perform semantic segmentation with Talk2Dino (Barselotti et al. 2025)
    """
    def __init__(self, ontology, image_insize, sharpness, return_logits, models_dir, device='cuda'):
        self.ontology = load_ontology(ontology)
        self.image_insize = image_insize
        self.sharpness = sharpness
        self.return_logits = return_logits
        self.device = device

        ##setup talk2dino
        self.talk2dino = AutoModel.from_pretrained(
            "lorebianchi98/Talk2DINO-ViTB",
            trust_remote_code=True
        ).to(self.device).eval()

        ##precompute text embeddings
        with torch.no_grad():
            self.text_embed = self.talk2dino.encode_text(self.ontology['prompts'])

    def run(self, image, intrinsics, image_orig):
        assert image.shape[1] == 3, "Talk2DinoSeg needs BGR inputs!"

        image_resize = F.resize(image, self.image_insize)
        image_in = image_resize[:, [2,1,0]] * 255. #1-scaled BGR -> 255-scaled RGB

        masks, _ = self.talk2dino.generate_masks(
            image_in,
            img_metas = None,
            text_emb = self.text_embed,
            classnames = ' '.join(self.ontology['labels']),
            apply_pamr = True
        )

        mask_logits = masks * self.sharpness

        img_out = mask_logits if self.return_logits else mask_logits.softmax(dim=1)

        ix = image.shape[3]
        dx = img_out.shape[3]
        iy = image.shape[2]
        dy = img_out.shape[2]

        intrinsics[:, 0, 0] *= dx / ix
        intrinsics[:, 0, 2] *= dx / ix

        intrinsics[:, 1, 1] *= dy / iy
        intrinsics[:, 1, 2] *= dy / iy

        return img_out, intrinsics

    @property
    def output_feature_keys(self):
        metainfo_key = "semantic_logits" if self.return_logits else "semantic_probs"
        return FeatureKeyList(
            label=self.ontology['labels'],
            metainfo=[metainfo_key for i in range(self.n_classes)]
        )

    @property
    def n_classes(self):
        return len(self.ontology['ids'])