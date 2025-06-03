import os
import torch

import math
import torch.nn.functional as F
from torch.nn.modules.utils import _pair
from types import MethodType

from physics_atv_visual_mapping.image_processing.anyloc_utils import (
    DinoV2ExtractFeatures,
)
from physics_atv_visual_mapping.image_processing.processing_blocks.base import (
    ImageProcessingBlock,
)

def get_model_params(dino_name: str):
        """Match a name like dinov2_vits14 / dinov2_vitg16_lc etc. to feature dim and patch size.

        :param dino_name: string of dino model name on torch hub
        :type dino_name: str
        :return: tuple of original patch size and hidden feature dimension
        :rtype: Tuple[int, int]
        """
        split_name = dino_name.split("_")
        model = split_name[1]
        arch, patch_size = model[3], int(model[4:])
        feat_dim_lookup = {"s": 384, "b": 768, "l": 1024, "g": 1536}
        feat_dim: int = feat_dim_lookup[arch]
        return feat_dim, patch_size

def _fix_pos_enc(patch_size: int, stride_hw):
        """Creates a method for position encoding interpolation, used to overwrite
        the original method in the DINO/DINOv2 vision transformer.
        Taken from https://github.com/ShirAmir/dino-vit-features/blob/main/extractor.py,
        added some bits from the Dv2 code in.

        :param patch_size: patch size of the model.
        :type patch_size: int
        :param stride_hw: A tuple containing the new height and width stride respectively.
        :type Tuple[int, int]
        :return: the interpolation method
        :rtype: Callable
        """
        def interpolate_pos_encoding(
            self, x: torch.Tensor, w: int, h: int
        ) -> torch.Tensor:
            previous_dtype = x.dtype
            npatch = x.shape[1] - 1
            N = self.pos_embed.shape[1] - 1
            if npatch == N and w == h:
                return self.pos_embed

            pos_embed = self.pos_embed.float()
            class_pos_embed = pos_embed[:, 0]
            patch_pos_embed = pos_embed[:, 1:]
            dim = x.shape[-1]

            # compute number of tokens taking stride into account
            w0: float = 1 + (w - patch_size) // stride_hw[1]
            h0: float = 1 + (h - patch_size) // stride_hw[0]

            assert (
                w0 * h0 == npatch
            ), f"""got wrong grid size for {h}x{w} with patch_size {patch_size} and
            #                               stride {stride_hw} got {h0}x{w0}={h0 * w0} expecting {npatch}"""

            # we add a small number to avoid floating point error in the interpolation
            # see discussion at https://github.com/facebookresearch/dino/issues/8
            w0, h0 = w0 + 0.1, h0 + 0.1
            patch_pos_embed = F.interpolate(
                patch_pos_embed.reshape(
                    1, int(math.sqrt(N)), int(math.sqrt(N)), dim
                ).permute(0, 3, 1, 2),
                scale_factor=(w0 / math.sqrt(N), h0 / math.sqrt(N)),
                mode="bicubic",
                align_corners=False,
                recompute_scale_factor=False,
            )
            
            assert (
                int(w0) == patch_pos_embed.shape[-2]
                and int(h0) == patch_pos_embed.shape[-1]
            )
            patch_pos_embed = patch_pos_embed.permute(0, 2, 3, 1).view(1, -1, dim)

            return torch.cat((class_pos_embed.unsqueeze(0), patch_pos_embed), dim=1).to(
                previous_dtype
            )

        return interpolate_pos_encoding

def set_model_stride(
        dino_model: torch.nn.Module, stride_l: int, ops, verbose: bool = False
    ) -> None:
        """Create new positional encoding interpolation method for $dino_model with
        supplied $stride, and set the stride of the patch embedding projection conv2D
        to $stride.

        :param dino_model: Dv2 model
        :type dino_model: DinoVisionTransformer
        :param new_stride: desired stride, usually stride < original_stride for higher res
        :type new_stride: int
        :return: None
        :rtype: None
        """

        new_stride_pair = torch.nn.modules.utils._pair(stride_l)

        # if new_stride_pair == self.stride:
        #    return  # early return as nothing to be done

        stride = new_stride_pair
        dino_model.patch_embed.proj.stride = new_stride_pair  # type: ignore

        if verbose:
            print(f"Setting stride to ({stride_l},{stride_l})")

        # if new_stride_pair == self.original_stride:
        # if resetting to original, return original method
        #    dino_model.interpolate_pos_encoding = self.original_pos_enc  # type: ignore
        # else:

        dino_model.interpolate_pos_encoding = MethodType(  # type: ignore
            _fix_pos_enc(ops, new_stride_pair),
            dino_model,
        )  # typed ignored as they can't type check reassigned methods (generally is poor practice)

class UpsampleDinov2Block(ImageProcessingBlock):
    """
    Image processing block that runs dino on the image
    """

    def __init__(
        self, dino_type, dino_layers, image_insize, desc_facet, stride, device, models_dir
    ):
        torch.hub.set_dir(os.path.join(models_dir, "torch_hub"))

        if "dino" in dino_type:
            dino_dir = os.path.join(models_dir, "torch_hub", "facebookresearch_dinov2_main")
        elif "radio" in dino_type:
            dino_dir = os.path.join(models_dir, "torch_hub", "NVlabs_RADIO_main")

        self.dino = DinoV2ExtractFeatures(
            dino_dir,
            dino_model=dino_type,
            layers=dino_layers,
            input_size=image_insize,
            facet=desc_facet,
            stride=stride,
            device=device,
        )
        
        patch = self.dino.dino_model.patch_size
        feat = self.dino.dino_model.embed_dim
        original_patch_size = patch
        original_stride = _pair(patch)
        stride = _pair(stride)

        set_model_stride(self.dino.dino_model, patch, original_patch_size)
        set_model_stride(self.dino.dino_model, stride, original_patch_size)

    def run(self, image, intrinsics, image_orig):
        img_out = self.dino(image)

        ix = image.shape[3]
        dx = img_out.shape[3]
        iy = image.shape[2]
        dy = img_out.shape[2]

        intrinsics[:, 0, 0] *= dx / ix
        intrinsics[:, 0, 2] *= dx / ix

        intrinsics[:, 1, 1] *= dy / iy
        intrinsics[:, 1, 2] *= dy / iy

        return img_out, intrinsics
