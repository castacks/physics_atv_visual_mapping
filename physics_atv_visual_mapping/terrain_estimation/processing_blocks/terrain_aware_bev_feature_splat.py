import torch
import torch_scatter

from physics_atv_visual_mapping.terrain_estimation.processing_blocks.base import TerrainEstimationBlock

class TerrainAwareBEVFeatureSplat(TerrainEstimationBlock):
    """
    Compute a per-cell min and max height
    """
    def __init__(self, voxel_metadata, voxel_n_features, terrain_layer, terrain_mask_layer, output_key, overhang, device):
        super().__init__(voxel_metadata, voxel_n_features, device)
        self.terrain_layer = terrain_layer
        self.terrain_mask_layer = terrain_mask_layer
        self.output_key = output_key
        self.overhang = overhang
        
    def to(self, device):
        self.device = device
        return self

    @property
    def output_keys(self):
        return ["{}_{}".format(self.output_key, i) for i in range(self.voxel_n_features)]

    def run(self, voxel_grid, bev_grid):
        terrain_idx = bev_grid.feature_keys.index(self.terrain_layer)
        terrain_data = bev_grid.data[..., terrain_idx].clone()

        mask_idx = bev_grid.feature_keys.index(self.terrain_mask_layer)
        terrain_mask = bev_grid.data[..., mask_idx] > 1e-4

        voxel_grid_idxs = voxel_grid.raster_indices_to_grid_indices(voxel_grid.feature_raster_indices)
        voxel_grid_pts = voxel_grid.grid_indices_to_pts(voxel_grid_idxs, centers=True)

        voxel_terrain_height = terrain_data[voxel_grid_idxs[:, 0], voxel_grid_idxs[:, 1]]
        voxel_terrain_mask = terrain_mask[voxel_grid_idxs[:, 0], voxel_grid_idxs[:, 1]]
        voxel_hdiff = voxel_grid_pts[:, 2] - voxel_terrain_height
        voxel_valid_mask = voxel_terrain_mask & (voxel_hdiff < self.overhang)

        #bev grid idxs are the first 2 dims of the voxel idxs assuming matching metadata
        raster_idxs = voxel_grid_idxs[:, 0] * bev_grid.metadata.N[1] + voxel_grid_idxs[:, 1]

        idxs_to_scatter = raster_idxs[voxel_valid_mask]
        features_to_scatter = voxel_grid.features[voxel_valid_mask]
    
        num_cells = (bev_grid.metadata.N[0] * bev_grid.metadata.N[1]).item()

        bev_features = torch_scatter.scatter(src=features_to_scatter, index=idxs_to_scatter, dim_size=num_cells, dim=0, reduce='mean')
        bev_features = bev_features.view(*bev_grid.metadata.N, self.voxel_n_features)

        bev_feature_idxs = [bev_grid.feature_keys.index(k) for k in self.output_keys]
        bev_grid.data[..., bev_feature_idxs] = bev_features

        return bev_grid