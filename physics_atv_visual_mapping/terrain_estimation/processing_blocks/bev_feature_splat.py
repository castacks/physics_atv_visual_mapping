import torch
import torch_scatter

from physics_atv_visual_mapping.terrain_estimation.processing_blocks.base import TerrainEstimationBlock

class BEVFeatureSplat(TerrainEstimationBlock):
    """
    Compute a per-cell min and max height
    """
    def __init__(self, output_key, voxel_metadata, voxel_n_features, device):
        super().__init__(voxel_metadata, voxel_n_features, device)
        self.output_key = output_key

    def to(self, device):
        self.device = device
        return self

    @property
    def output_keys(self):
        return ["{}_{}".format(self.output_key, i) for i in range(self.voxel_n_features)]

    def run(self, voxel_grid, bev_grid):

        voxel_grid_idxs = voxel_grid.raster_indices_to_grid_indices(voxel_grid.indices)

        #bev grid idxs are the first 2 dims of the voxel idxs assuming matching metadata
        raster_idxs = voxel_grid_idxs[:, 0] * bev_grid.metadata.N[1] + voxel_grid_idxs[:, 1]
    
        num_cells = (bev_grid.metadata.N[0] * bev_grid.metadata.N[1]).item()

        bev_features = torch_scatter.scatter(src=voxel_grid.features, index=raster_idxs, dim_size=num_cells, dim=0, reduce='mean')
        bev_features = bev_features.view(*bev_grid.metadata.N, self.voxel_n_features)

        bev_feature_idxs = [bev_grid.feature_key_list.index(k) for k in self.output_keys]
        bev_grid.data[..., bev_feature_idxs] = bev_features

        return bev_grid
