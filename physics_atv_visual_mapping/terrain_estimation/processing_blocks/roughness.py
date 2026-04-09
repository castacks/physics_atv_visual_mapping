import torch
import torch_scatter

from physics_atv_visual_mapping.terrain_estimation.processing_blocks.base import TerrainEstimationBlock
from physics_atv_visual_mapping.terrain_estimation.processing_blocks.utils import setup_kernel, apply_kernel
from physics_atv_visual_mapping.feature_key_list import FeatureKeyList

class Roughness(TerrainEstimationBlock):
    """
    Compute a per-cell min and max height
    """
    def __init__(self, voxel_metadata, voxel_feature_keys, input_layer, mask_layer, kernel_params, thresh, device):
        """
        Args:
            thresh: at least this fraction of neighboring cells must be observed
        """
        super().__init__(voxel_metadata, voxel_feature_keys, device)
        self.input_layer = input_layer
        self.mask_layer = mask_layer
        
        self.kernel = setup_kernel(**kernel_params, metadata=voxel_metadata).to(device)
        self.thresh = thresh * self.kernel.sum()
        
    def to(self, device):
        self.device = device
        return self

    @property
    def output_feature_keys(self):
        return FeatureKeyList(
            label = ["roughness"],
            metainfo = ["terrain_estimation"]
        )

    def run(self, voxel_grid, bev_grid):
        input_idx = bev_grid.feature_keys.index(self.input_layer)
        input_data = bev_grid.data[..., input_idx].clone()

        mask_idx = bev_grid.feature_keys.index(self.mask_layer)
        valid_mask = bev_grid.data[..., mask_idx] > 1e-4

        input_data[~valid_mask] = 0.
        input_data_sq = torch.pow(input_data, 2)

        sum_x = apply_kernel(kernel=self.kernel, data=input_data)      
        sum_x2 = apply_kernel(kernel=self.kernel, data=input_data_sq)  
        count = apply_kernel(kernel=self.kernel, data=valid_mask.float()) 

        roughness = torch.zeros_like(input_data)

        # var = E[x^2] - E[x]^2
        E_x2 = sum_x2[valid_mask] / count[valid_mask]
        E_x = sum_x[valid_mask] / count[valid_mask]

        roughness[valid_mask] = torch.sqrt(E_x2 - (torch.pow(E_x, 2)))

        output_data_idx = bev_grid.feature_keys.index(self.output_feature_keys.label[0])

        bev_grid.data[..., output_data_idx] = roughness

        return bev_grid
