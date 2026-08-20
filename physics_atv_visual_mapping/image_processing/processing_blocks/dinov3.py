import os

import torch
import torchvision.transforms.functional as TF

from physics_atv_visual_mapping.feature_key_list import FeatureKeyList
from physics_atv_visual_mapping.image_processing.processing_blocks.base import ImageProcessingBlock


class Dinov3Block(ImageProcessingBlock):
    """
    Image processing block that runs DINOv3 on the image and returns the
    final intermediate feature map from the model.
    """

    def __init__(
        self,
        dino_type,
        image_insize,
        device,
        models_dir,
        num_layers,
        do_norm,
        weights=None,
    ):
        torch.hub.set_dir(os.path.join(models_dir, "torch_hub"))

        self.dino_type = dino_type
        self.num_layers = num_layers
        self.device = torch.device(device)
        self.input_size = tuple(image_insize)
        self.weights = weights
        self.do_norm = do_norm
        # self.dino_dir = self._resolve_dino_dir(models_dir)
        self.dino_dir = os.path.join(models_dir, "dinov3")

        if self.weights is not None and not os.path.isabs(self.weights):
            self.weights = os.path.join(models_dir, self.weights)

        self.dino = torch.hub.load(
            self.dino_dir,
            self.dino_type,
            source="local",
            weights=self.weights,
        )

        self.dino = self.dino.to(self.device).eval()

        self.output_channels = self._resolve_output_channels()

    # def _resolve_dino_dir(self, models_dir):
    #     candidates = [
    #         models_dir,
    #         os.path.join(models_dir, "torch_hub"),
    #         os.path.join(models_dir, "torch_hub", "dinov3"),
    #     ]
    #     for candidate in candidates:
    #         if os.path.isdir(candidate):
    #             return candidate

    #     raise FileNotFoundError(
    #         f"Could not resolve DINOv3 directory from models_dir={models_dir}"
    #     )

    def _resolve_output_channels(self):
        for attr in ("embed_dim", "dim", "head_dim", "hidden_dim", "hidden_size"):
            if hasattr(self.dino, attr):
                return int(getattr(self.dino, attr))

        dummy = self._extract_features(
            torch.zeros(1, 3, self.input_size[1], self.input_size[0], device=self.device)
        )
        return int(dummy.shape[1])

    def _preprocess(self, img: torch.Tensor) -> torch.Tensor:
        assert len(img.shape) == 4, "need to batch images"
        assert img.shape[1] == 3, "expects channels-first"
        img = img.to(self.device).float()
        return TF.resize(img, (self.input_size[1], self.input_size[0]))

    def _extract_features(self, img: torch.Tensor) -> torch.Tensor:
        img = self._preprocess(img)
        with torch.no_grad():
            features = self.dino.get_intermediate_layers(
                img,
                n=range(self.num_layers),
                reshape=True,
                norm=self.do_norm,
            )

        if isinstance(features, (list, tuple)):
            features = features[-1]

        return features

    def run(self, image, intrinsics, image_orig):
        img_out = self._extract_features(image)

        ix = image.shape[3]
        dx = img_out.shape[3]
        iy = image.shape[2]
        dy = img_out.shape[2]

        intrinsics = intrinsics.clone()
        intrinsics[:, 0, 0] *= dx / ix
        intrinsics[:, 0, 2] *= dx / ix
        intrinsics[:, 1, 1] *= dy / iy
        intrinsics[:, 1, 2] *= dy / iy

        return img_out, intrinsics

    @property
    def output_feature_keys(self):
        return FeatureKeyList(
            label=[f"{self.dino_type}_{i}" for i in range(self.output_channels)],
            metainfo=["vfm" for _ in range(self.output_channels)],
        )
