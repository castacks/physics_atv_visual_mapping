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
    def __init__(self, ontology, image_insize, sharpness, return_logits, models_dir, device='cuda', apply_pamr=True, use_fp16=False, tensorrt_engine=None):
        self.ontology = load_ontology(ontology)
        self.image_insize = image_insize
        self.sharpness = sharpness
        self.return_logits = return_logits
        self.device = device
        self.apply_pamr = apply_pamr
        self.use_fp16 = use_fp16
        self.tensorrt_engine = os.path.expandvars(tensorrt_engine or "")
        if "$" in self.tensorrt_engine:
            self.tensorrt_engine = ""
        elif self.tensorrt_engine and not os.path.isfile(self.tensorrt_engine):
            print(f"Talk2DINO TensorRT engine not found; using PyTorch: {self.tensorrt_engine}")
            self.tensorrt_engine = ""

        ##setup talk2dino
        self.talk2dino = AutoModel.from_pretrained(
            "lorebianchi98/Talk2DINO-ViTB",
            trust_remote_code=True
        ).to(self.device).eval()

        ##precompute text embeddings
        with torch.no_grad():
            self.text_embed = self.talk2dino.encode_text(self.ontology['prompts'])

        if self.tensorrt_engine:
            from physics_atv_visual_mapping.image_processing.processing_blocks.tensorrt_dino_backbone import (
                TensorRTDinoBackbone,
            )

            self.talk2dino.model = TensorRTDinoBackbone(
                self.tensorrt_engine,
                self.talk2dino.feats,
                device=self.device,
            ).eval()
            torch.cuda.empty_cache()
            print(f"Talk2DINO backbone backend: TensorRT ({self.tensorrt_engine})")
        else:
            print("Talk2DINO backbone backend: PyTorch")

    def run(self, image, intrinsics, image_orig):
        assert image.shape[1] == 3, "Talk2DinoSeg needs BGR inputs!"

        image_resize = F.resize(image, self.image_insize)
        image_in = image_resize[:, [2,1,0]] * 255. #1-scaled BGR -> 255-scaled RGB

        use_amp = self.use_fp16 and str(self.device).startswith('cuda')
        with torch.inference_mode():
            if use_amp:
                with torch.autocast(device_type='cuda', dtype=torch.float16):
                    masks, _ = self.talk2dino.generate_masks(
                        image_in,
                        img_metas = None,
                        text_emb = self.text_embed,
                        classnames = ' '.join(self.ontology['labels']),
                        apply_pamr = self.apply_pamr
                    )
            else:
                masks, _ = self.talk2dino.generate_masks(
                    image_in,
                    img_metas = None,
                    text_emb = self.text_embed,
                    classnames = ' '.join(self.ontology['labels']),
                    apply_pamr = self.apply_pamr
                )

        mask_logits = masks.float() * self.sharpness

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
