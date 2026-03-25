import torch
from PIL import Image
from torch import nn
from torch.nn import functional as F
from torchvision.transforms.functional import pil_to_tensor


class RadioWrapper(nn.Module):
    """
    RADIO backbone wrapper.
    Note: Input must be a PIL Image. RADIO expects input values in [0, 1] (float32).
    RADIO will automatically normalize to mean 0, std 1 internally.
    """

    def __init__(self, name="radio_v2.5-b", device="cuda"):
        super().__init__()
        self.name = name
        self.device = device
        self.model = torch.hub.load(
            "NVlabs/RADIO",
            "radio_model",
            version=name,
            progress=True,
            skip_validation=True,
        )
        self.config = {"mean": torch.tensor([0.0, 0, 0]), "std": torch.tensor([1.0, 1, 1])}  # RADIO normalizes internally
        self.model.to(self.device).eval()
        if name == "radio_v2.5-h":
            self.embed_dim = 1280
        elif name == "radio_v2.5-b":
            self.embed_dim = 768
        self.patch_size = 1

    def preprocess(self, img: Image.Image):
        x = img
        nearest_res = self.model.get_nearest_supported_resolution(*x.shape[-2:])
        x = F.interpolate(x, nearest_res, mode="bilinear", align_corners=False)
        if "e-radio" in self.name:
            self.model.model.set_optimal_window_size(x.shape[2:])
        return x

    @torch.no_grad()
    def forward(self, img: Image.Image):
        x = self.preprocess(img)
        # Only return spatial_features in NCHW format
        _, spatial_features = self.model(x, feature_fmt="NCHW")
        assert spatial_features.ndim == 4
        return spatial_features, None
