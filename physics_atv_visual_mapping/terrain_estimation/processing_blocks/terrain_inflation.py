import torch
import torch_scatter

from physics_atv_visual_mapping.terrain_estimation.processing_blocks.base import TerrainEstimationBlock
from physics_atv_visual_mapping.terrain_estimation.processing_blocks.utils import setup_kernel, apply_kernel

class TerrainInflation(TerrainEstimationBlock):
    """
    Compute a per-cell min and max height
    """
    def __init__(self, voxel_metadata, voxel_n_features, input_layer, mask_layer, kernel_params, thresh, device):
        """
        Args:
            thresh: at least this fraction of neighboring cells must be observed
        """
        super().__init__(voxel_metadata, voxel_n_features, device)
        self.input_layer = input_layer
        self.mask_layer = mask_layer
        
        self.kernel = setup_kernel(**kernel_params, metadata=voxel_metadata).to(device)
        self.thresh = thresh * self.kernel.sum()
        
    def to(self, device):
        self.device = device
        return self

    @property
    def output_keys(self):
        return ["{}_inflated".format(self.input_layer), "{}_inflated_mask".format(self.input_layer)]

    def run(self, voxel_grid, bev_grid):
        input_idx = bev_grid.feature_keys.index(self.input_layer)
        input_data = bev_grid.data[..., input_idx].clone()

        mask_idx = bev_grid.feature_keys.index(self.mask_layer)
        valid_mask = bev_grid.data[..., mask_idx] > 1e-4

        input_data[~valid_mask] = 0.

        height_sum = apply_kernel(kernel=self.kernel, data=input_data)
        height_cnt = apply_kernel(kernel=self.kernel, data=valid_mask.float())
        height_avg = height_sum/height_cnt

        output_valid_mask = height_cnt > self.thresh
        height_avg[~output_valid_mask] = 0.

        #only copy values where the interpolation is valid and there is no data
        copy_mask = ~valid_mask & output_valid_mask
        input_data[copy_mask] = height_avg[copy_mask]

        output_data_idx = bev_grid.feature_keys.index(self.output_keys[0])
        output_mask_idx = bev_grid.feature_keys.index(self.output_keys[1])

        bev_grid.data[..., output_data_idx] = input_data
        bev_grid.data[..., output_mask_idx] = output_valid_mask.float()

        return bev_grid