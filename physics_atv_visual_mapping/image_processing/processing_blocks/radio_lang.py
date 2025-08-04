import os
import math
import torch
import torchvision
import torch.nn.functional as F

from torch import nn
from timm.layers import use_fused_attn

# from ptflops import get_model_complexity_info
# from thop import profile

from physics_atv_visual_mapping.image_processing.processing_blocks.base import (
    ImageProcessingBlock,
)
from physics_atv_visual_mapping.feature_key_list import FeatureKeyList

"""
Code heavily borrowed from https://github.com/RayFronts/RayFronts/blob/main/rayfronts/image_encoders/naradio.py
"""

class GaussKernelAttn(nn.Module):
  """Encompases the NACLIP attention mechanism."""

  def __init__(
    self,
    orig_attn,
    input_resolution: tuple,
    gauss_std: float,
    device,
    chosen_cls_id: int,
    dim: int,
    qk_norm: bool = False,
    num_prefix_tokens: int = 8,
  ) -> None:
    super().__init__()
    num_heads = orig_attn.num_heads
    assert dim % num_heads == 0, "dim should be divisible by num_heads"
    self.num_heads = num_heads
    self.head_dim = dim // num_heads
    self.scale = self.head_dim ** -0.5
    self.fused_attn = use_fused_attn()

    self.addition_cache = dict()
    self.input_resolution = input_resolution
    self.chosen_cls_id = chosen_cls_id
    self.gauss_std = gauss_std

    self.qkv = orig_attn.qkv
    self.q_norm = orig_attn.q_norm if qk_norm else nn.Identity()
    self.k_norm = orig_attn.k_norm if qk_norm else nn.Identity()
    self.attn_drop = orig_attn.attn_drop
    self.proj = orig_attn.proj
    self.proj_drop = orig_attn.proj_drop
    self.device = device
    self.num_prefix_tokens = num_prefix_tokens

  def forward(self, x: torch.Tensor, attn_mask=None) -> torch.Tensor:
    B, N, C = x.shape
    h, w = self.input_resolution
    n_patches = (w // 16, h //16)

    x_out = self.custom_attn(x.permute(1, 0, 2), n_patches, self.gauss_std)
    x_out = x_out.permute(1, 0, 2)

    return x_out

  @staticmethod
  def gaussian_window(dim1, dim2, std=5.):
    constant = 1 / (std * math.sqrt(2))
    ks = list()
    for dim in [dim1, dim2]:
      start = -(dim - 1) / 2.0
      k = torch.linspace(start=start * constant,
                         end=(start + (dim - 1)) * constant,
                         steps=dim,
                         dtype=torch.float)
      ks.append(k)
    dist_square_to_mu = (torch.stack(torch.meshgrid(
      *ks, indexing="ij")) ** 2).sum(0)

    return torch.exp(-dist_square_to_mu)

  @staticmethod
  def get_attention_addition(dim1, dim2, window, num_prefix_tokens=8):
    m = torch.einsum("ij,kl->ijkl", torch.eye(dim1), torch.eye(dim2))
    m = m.permute((0, 3, 1, 2)).contiguous()
    out = F.conv2d(m.view(-1, dim1, dim2).unsqueeze(1),
                   window.unsqueeze(0).unsqueeze(1),
                   padding='same').squeeze(1)

    out = out.view(dim1 * dim2, dim1 * dim2)
    if num_prefix_tokens > 0:
      v_adjusted = torch.vstack(
        [torch.zeros((num_prefix_tokens, dim1 * dim2)), out])
      out = torch.hstack(
        [torch.zeros((dim1 * dim2 + num_prefix_tokens, num_prefix_tokens)),
         v_adjusted])

    return out

  def custom_attn(self, x, n_patches, gauss_std):
    num_heads = self.num_heads
    num_tokens, bsz, embed_dim = x.size()
    head_dim = embed_dim // num_heads
    scale = head_dim ** -0.5

    q, k, v = self.qkv(x).chunk(3, dim=-1)
    q, k = self.q_norm(q), self.k_norm(k)

    q = q.contiguous().view(-1, bsz * num_heads, head_dim).transpose(0, 1)
    k = k.contiguous().view(-1, bsz * num_heads, head_dim).transpose(0, 1)
    v = v.contiguous().view(-1, bsz * num_heads, head_dim).transpose(0, 1)

    addition = self.addition_cache.get(n_patches)
    if addition is None:
      window_size = [side * 2 - 1 for side in n_patches] 
      window = GaussKernelAttn.gaussian_window(*window_size, std=gauss_std)
      addition = GaussKernelAttn.get_attention_addition(
        *n_patches, window, self.num_prefix_tokens
      ).unsqueeze(0).to(x.dtype).to(x.device)

      self.addition_cache[n_patches] = addition

    # kk.T vs kq.T has the most impact
    attn_weights = torch.bmm(k, k.transpose(1, 2)) * scale
    omega = addition

    # Gaussian attention seems to have minimal impact
    attn_weights += omega
    attn_weights = F.softmax(attn_weights, dim=-1)

    attn_output = torch.bmm(attn_weights, v)
    attn_output = attn_output.transpose(0, 1).contiguous().view(
      -1, bsz, embed_dim)
    attn_output = self.proj(attn_output)
    attn_output = self.proj_drop(attn_output)

    return attn_output

class RadioLangBlock(ImageProcessingBlock):
    """
    Image processing block that runs dino on the image
    """

    def __init__(self, radio_type, image_insize, models_dir, adaptor_type, device):
        self.input_size = image_insize
        self.radio_type = radio_type
        self.adaptor_type = adaptor_type
        self.device = device

        # call this to run from local
        torch.hub.set_dir(os.path.join(models_dir, "torch_hub"))
        radio_fp = os.path.join(models_dir, "torch_hub", "NVlabs_RADIO_main")
        radio = torch.hub.load(
            radio_fp,
            "radio_model",
            version=radio_type,
            progress=True,
            skip_validation=True,
            source="local",
            adaptor_names=self.adaptor_type
        )  #  force_reload=True

        self.radio = radio.to(device).eval()
        self.adaptor = radio.adaptors[self.adaptor_type]
        self.radio.adaptors = None

        self.output_size = (int(image_insize[0] / self.radio.patch_size), int(image_insize[1] / self.radio.patch_size))

        last_block = self.radio.model.blocks[-1]
        last_block.attn = GaussKernelAttn(
            last_block.attn,
            input_resolution=self.input_size,
            gauss_std=7.0,
            dim=self.radio.model.embed_dim,
            chosen_cls_id=self.adaptor.head_idx,
            device=self.device,
            num_prefix_tokens=self.radio.num_summary_tokens
        )

    def preprocess(self, img):
        assert len(img.shape) == 4, "need to batch images"
        assert img.shape[1] == 3, "expects channels-first"
        img = img.to(self.device).float()
        img = torchvision.transforms.functional.resize(
            img, (self.input_size[1], self.input_size[0])
        )
        return img

    def run(self, image, intrinsics, image_orig):
        with torch.no_grad():
            img = self.preprocess(image)

            #[BxWHxC]
            feats = self.radio(img).features
            lang_feats = self.adaptor.head_mlp(feats)

            #[BxCxWxH]
            img_out = lang_feats.view(img.shape[0], self.output_size[1], self.output_size[0], -1).permute(0,3,1,2)
            img_out = F.normalize(img_out, dim=-1)

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
        # self.radio.n_output_channels
        return FeatureKeyList(
            label=[f"{self.radio_type}_{self.adaptor_type}_{i}" for i in range(self.radio.embed_dim)],
            metainfo=["vfm" for i in range(self.radio.embed_dim)]
        )