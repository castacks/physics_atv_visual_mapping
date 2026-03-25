import numpy as np
import torch
import torchvision.transforms as T
from einops import rearrange


def create_coordinate(h, w, start=0, end=1, device="cuda", dtype=torch.float32):
    # Create a grid of coordinates
    x = torch.linspace(start, end, h, device=device, dtype=dtype)
    y = torch.linspace(start, end, w, device=device, dtype=dtype)
    # Create a 2D map using meshgrid
    xx, yy = torch.meshgrid(x, y, indexing="ij")
    # Stack the x and y coordinates to create the final map
    coord_map = torch.stack([xx, yy], axis=-1)[None, ...]
    coords = rearrange(coord_map, "b h w c -> b (h w) c", h=h, w=w)
    return coords


def unnormalize(tensor, mean, std):
    """
    Unnormalizes a 4D tensor of shape (B, 3, H, W).
    """
    assert len(tensor.shape) == 4
    mean = mean.view(1, 3, 1, 1).to("cuda")  # Reshape to (1, C, 1, 1) for broadcasting
    std = std.view(1, 3, 1, 1).to("cuda")  # Reshape to (1, C, 1, 1) for broadcasting

    return tensor * std + mean  # Unnormalize


class PILToTensor:
    """Convert PIL Image to Tensor"""

    def __call__(self, image):
        image = T.functional.pil_to_tensor(image)
        return image
