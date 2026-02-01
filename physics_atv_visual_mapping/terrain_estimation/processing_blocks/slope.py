import torch
import torch_scatter

from physics_atv_visual_mapping.terrain_estimation.processing_blocks.base import TerrainEstimationBlock
from physics_atv_visual_mapping.terrain_estimation.processing_blocks.utils import setup_kernel, apply_kernel
from physics_atv_visual_mapping.feature_key_list import FeatureKeyList

class Slope(TerrainEstimationBlock):
    """
    Compute a per-cell min and max height
    """
    def __init__(self, voxel_metadata, voxel_feature_keys, input_layer, radius, kernel_type, mask_layer=None, max_slope=1e10, device='cpu'):
        assert kernel_type in ['sobel', 'scharr']

        super().__init__(voxel_metadata, voxel_feature_keys, device)
        self.input_layer = input_layer
        self.mask_layer = mask_layer
        self.kernel_type = kernel_type

        #allow for optional cap of max slope to reduce noise/
        #be a better learning feature
        self.max_slope = max_slope

        self.gradient_x = setup_kernel(
            kernel_type=f"{kernel_type}_x",
            kernel_radius=radius,
            metadata=voxel_metadata,
        ).to(self.device)

        self.gradient_y = setup_kernel(
            kernel_type=f"{kernel_type}_y",
            kernel_radius=radius,
            metadata=voxel_metadata,
        ).to(self.device)

        self.box = torch.ones_like(self.gradient_x) / self.gradient_x.numel()

    def to(self, device):
        self.gradient_x = self.gradient_x.to(device)
        self.gradient_y = self.gradient_y.to(device)
        self.box = self.box.to(device)
        self.device = device
        return self

    @property
    def output_feature_keys(self):
        return FeatureKeyList(
            label = ["slope_x", "slope_y", "slope_magnitude"],
            metainfo = ["terrain_estimation"] * 3
        )

    def run(self, voxel_grid, bev_grid):
        terrain_idx = bev_grid.feature_keys.index(self.input_layer)
        terrain_data = bev_grid.data[..., terrain_idx].clone()

        #only take slopes if all convolved elements valid
        if self.mask_layer is None:
            valid_mask = torch.ones(*bev_grid.metadata.N, dtype=torch.bool, device=self.device)
        else:
            mask_idx = bev_grid.feature_keys.index(self.mask_layer)
            mask = bev_grid.data[..., mask_idx] > 1e-4
            valid_mask = apply_kernel(kernel=self.box, data=mask.float()) > 0.9999

        #correct by resolution to get slope as m/m instead of m/cell
        slope_x = apply_kernel(kernel=self.gradient_x, data=terrain_data) / bev_grid.metadata.resolution[0]
        slope_x[~valid_mask] = 0.
        slope_x = slope_x.clip(-self.max_slope, self.max_slope)
        slope_y = apply_kernel(kernel=self.gradient_y, data=terrain_data) / bev_grid.metadata.resolution[1]
        slope_y[~valid_mask] = 0.
        slope_y = slope_y.clip(-self.max_slope, self.max_slope)
        slope = torch.hypot(slope_x, slope_y)

        slope_x_idx = bev_grid.feature_keys.index(self.output_feature_keys.label[0])
        bev_grid.data[..., slope_x_idx] = slope_x

        slope_y_idx = bev_grid.feature_keys.index(self.output_feature_keys.label[1])
        bev_grid.data[..., slope_y_idx] = slope_y

        slope_idx = bev_grid.feature_keys.index(self.output_feature_keys.label[2])
        bev_grid.data[..., slope_idx] = slope

        return bev_grid