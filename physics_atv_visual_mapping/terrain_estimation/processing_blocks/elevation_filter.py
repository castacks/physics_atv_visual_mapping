import torch
import torch_scatter

from physics_atv_visual_mapping.terrain_estimation.processing_blocks.base import TerrainEstimationBlock
from physics_atv_visual_mapping.terrain_estimation.processing_blocks.utils import setup_kernel, apply_kernel

class ElevationFilter(TerrainEstimationBlock):
    """
    Compute a per-cell min and max height
    """
    def __init__(self, voxel_metadata, voxel_n_features, input_layer, cnt_layer, kernel_params, height_low_thresh, height_high_thresh, device):
        super().__init__(voxel_metadata, voxel_n_features, device)
        self.input_layer = input_layer
        self.cnt_layer = cnt_layer

        self.height_low_thresh = height_low_thresh
        self.height_high_thresh = height_high_thresh

        self.kernel = setup_kernel(**kernel_params, metadata=voxel_metadata).to(device)
        
    def to(self, device):
        self.kernel = self.kernel.to(device)
        self.device = device
        return self

    @property
    def output_keys(self):
        return ["{}_filtered".format(self.input_layer), "{}_filtered_mask".format(self.input_layer)]

    def run(self, voxel_grid, bev_grid):
        kmid = self.kernel.shape[0]//2
        kernel = self.kernel.clone()
        kernel[kmid, kmid] = 0.

        input_idx = bev_grid.feature_key_list.index(self.input_layer)
        input_data = bev_grid.data[..., input_idx].clone()

        cnt_idx = bev_grid.feature_key_list.index(self.cnt_layer)
        valid_mask = bev_grid.data[..., cnt_idx] > 1e-4

        #empty placeholder must be 0 to count correctly
        input_data[~valid_mask] = 0.

        height_sum = apply_kernel(kernel=kernel, data=input_data)
        height_cnt = apply_kernel(kernel=kernel, data=valid_mask.float())
        height_avg = height_sum/height_cnt

        #filter out cells that are too far from the interpolated avg
        height_diff = input_data - height_avg

        valid_but_no_avg = valid_mask & (height_cnt == 0)
        valid_in_bounds = valid_mask & (height_diff < self.height_high_thresh) & (height_diff > self.height_low_thresh)

        valid_mask = valid_but_no_avg | valid_in_bounds
        
        input_data[~valid_mask] = 0.

        output_data_idx = bev_grid.feature_key_list.index(self.output_keys[0])
        output_mask_idx = bev_grid.feature_key_list.index(self.output_keys[1])

        bev_grid.data[..., output_data_idx] = input_data
        bev_grid.data[..., output_mask_idx] = valid_mask.float()

        return bev_grid