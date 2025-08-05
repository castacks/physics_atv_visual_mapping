import torch
import torch_scatter

from physics_atv_visual_mapping.terrain_estimation.processing_blocks.base import TerrainEstimationBlock
from physics_atv_visual_mapping.terrain_estimation.processing_blocks.utils import sobel_x_kernel, sobel_y_kernel, apply_kernel
from physics_atv_visual_mapping.feature_key_list import FeatureKeyList

class BEVTerrainDiff(TerrainEstimationBlock):
    """
    Compute a diff feature. Unlike the TerrainDiffBlock, compute using only the elevation map
    """
    def __init__(self, voxel_metadata, voxel_feature_keys, terrain_layer, max_elevation_layer, overhang, device):
        super().__init__(voxel_metadata, voxel_feature_keys, device)
        self.terrain_layer = terrain_layer
        self.max_elevation_layer = max_elevation_layer
        self.overhang = overhang

    def to(self, device):
        self.device = device
        return self

    @property
    def output_feature_keys(self):
        return FeatureKeyList(
            label = ["diff"],
            metainfo = ["terrain_estimation"]
        )

    def run(self, voxel_grid, bev_grid):
        terrain_idx = bev_grid.feature_keys.index(self.terrain_layer)
        terrain_data = bev_grid.data[..., terrain_idx].clone()

        max_elevation_idx = bev_grid.feature_keys.index(self.max_elevation_layer)
        max_elevation_data = bev_grid.data[..., max_elevation_idx].clone()

        diff = (max_elevation_data - terrain_data).clip(0., self.overhang)

        diff_idx = bev_grid.feature_keys.index(self.output_feature_keys.label[0])
        bev_grid.data[..., diff_idx] = diff

        return bev_grid