import yaml
import torch
import torchvision.transforms.functional as F

from transformers import AutoModel

from physics_atv_visual_mapping.feature_key_list import FeatureKeyList
from physics_atv_visual_mapping.image_processing.processing_blocks.base import (
    ImageProcessingBlock,
)
from physics_atv_visual_mapping.utils import load_ontology


class Talk2DinoSegBlock(ImageProcessingBlock):
    """Semantic segmentation with Talk2DINO."""

    def __init__(
        self,
        ontology=None,
        ontology_fp=None,
        image_insize=(128, 224),
        sharpness=20.0,
        return_logits=True,
        input_color_order="bgr",
        models_dir=None,
        device="cuda",
    ):
        if ontology is None:
            if not ontology_fp:
                raise ValueError("Talk2DinoSegBlock requires ontology or ontology_fp")
            with open(ontology_fp, "r") as f:
                ontology = yaml.safe_load(f)

        self.ontology = load_ontology(ontology)
        self.image_insize = list(image_insize)
        self.sharpness = float(sharpness)
        self.return_logits = bool(return_logits)
        self.input_color_order = input_color_order.lower()
        self.device = device

        self.talk2dino = AutoModel.from_pretrained(
            "lorebianchi98/Talk2DINO-ViTB",
            trust_remote_code=True,
        ).to(self.device).eval()

        with torch.no_grad():
            self.text_embed = self.talk2dino.encode_text(self.ontology["prompts"])

    def run(self, image, intrinsics, image_orig):
        if image.shape[1] != 3:
            raise ValueError("Talk2DinoSegBlock expects 3-channel color images")

        image_resize = F.resize(image, self.image_insize)
        if self.input_color_order == "bgr":
            image_in = image_resize[:, [2, 1, 0]] * 255.0
        elif self.input_color_order == "rgb":
            image_in = image_resize * 255.0
        else:
            raise ValueError(f"Unsupported input_color_order={self.input_color_order}")

        with torch.no_grad():
            masks, _ = self.talk2dino.generate_masks(
                image_in,
                img_metas=None,
                text_emb=self.text_embed,
                classnames=" ".join(self.ontology["labels"]),
                apply_pamr=True,
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
            label=self.ontology["labels"],
            metainfo=[metainfo_key for _ in range(self.n_classes)],
        )

    @property
    def n_classes(self):
        return len(self.ontology["ids"])
