import torch

from physics_atv_visual_mapping.terrain_estimation.processing_blocks.base import TerrainEstimationBlock
from physics_atv_visual_mapping.feature_key_list import FeatureKeyList

class TerrainNormalsGradient(TerrainEstimationBlock):
    """
    Compute terrain normals from slope values
    """
    def __init__(self, voxel_metadata, voxel_feature_keys, slope_x_layer, slope_y_layer, mask_layer, device='cpu'):
        super().__init__(voxel_metadata, voxel_feature_keys, device)
        self.slope_x_layer = slope_x_layer
        self.slope_y_layer = slope_y_layer
        self.mask_layer = mask_layer

    def run(self, voxel_grid, bev_grid):
        """
        Args:
            voxel_grid: the input voxel grid
            bev_grid: the input BEV grid
        
        Returns:
            bev_grid: the output BEV grid 
        """

        # Index and extract the slope in the x-direction
        slope_x_idx = bev_grid.feature_keys.index(self.slope_x_layer)
        slope_x_data = bev_grid.data[..., slope_x_idx].clone()

        # Index and extract the slope in the y-direction
        slope_y_idx = bev_grid.feature_keys.index(self.slope_y_layer)
        slope_y_data = bev_grid.data[..., slope_y_idx].clone()

        # Index and apply a threshold to the mask layer to identify valid regions
        mask_idx = bev_grid.feature_keys.index(self.mask_layer)
        mask = bev_grid.data[..., mask_idx] > 1e-4

        # Construct a unit vector in x-slope direction: (1, 0, dz/dx)
        ones_dim = torch.ones_like(slope_x_data)
        zero_dim = torch.zeros_like(slope_x_data)
        slope_x_vec = torch.stack((ones_dim, zero_dim, slope_x_data), dim=-1)

        # Construct a unit vector in y-slope direction: (0, 1, dz/dy)
        slope_y_vec = torch.stack((zero_dim, ones_dim, slope_y_data), dim=-1)

        # Use the cross product of the two slope vectors to compute normal vector
        normals = torch.linalg.cross(slope_x_vec, slope_y_vec)

        # Normalize the normal vectors to unit length
        normals = normals / torch.linalg.vector_norm(normals, dim=-1, keepdim=True)

        # Extract x-component and zero out values outside the mask
        normals_x = normals[..., 0]
        normals_x[~mask] = 0

        # Extract y-component and zero out values outside the mask
        normals_y = normals[..., 1]
        normals_y[~mask] = 0

        # Extract z-component and zero out values outside the mask
        normals_z = normals[..., 2]
        normals_z[~mask] = 0

        # Store computed normals into BEV grid output
        normals_x_idx = bev_grid.feature_keys.index(self.output_feature_keys.label[0])
        bev_grid.data[..., normals_x_idx] = normals_x

        normals_y_idx = bev_grid.feature_keys.index(self.output_feature_keys.label[1])
        bev_grid.data[..., normals_y_idx] = normals_y

        normals_z_idx = bev_grid.feature_keys.index(self.output_feature_keys.label[2])
        bev_grid.data[..., normals_z_idx] = normals_z

        return bev_grid

    @property
    def output_feature_keys(self):
        """
        define the layer keys to output to 
        Note that for some layers such as BEVSplat, the output keys depend on the input
        """
        return FeatureKeyList(
            label = ['normal_x','normal_y','normal_z'],
            metainfo = ['terrain_estimation'] * 3
        )

    def to(self, device):
        self.device = device
        return self