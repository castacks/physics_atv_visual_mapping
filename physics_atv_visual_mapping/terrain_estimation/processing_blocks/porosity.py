import torch
import torch_scatter

from physics_atv_visual_mapping.terrain_estimation.processing_blocks.base import TerrainEstimationBlock

class Porosity(TerrainEstimationBlock):
    """
    Compute porosity (avg voxel passthrough in a column)
    """
    def __init__(self, voxel_metadata, voxel_n_features, device, reduce="mean"):
        super().__init__(voxel_metadata, voxel_n_features, device)
        self.reduce = reduce
        
    def to(self, device):
        self.device = device
        return self

    @property
    def output_keys(self):
        return ["porosity"]

    def run(self, voxel_grid, bev_grid):
        #keys can vary so we need to recompute them here
        porosity_idx = bev_grid.feature_keys.index(self.output_keys[0])

        #get grid idxs and coordinates of voxel grid
        voxel_grid_idxs = voxel_grid.raster_indices_to_grid_indices(voxel_grid.raster_indices)
        voxel_grid_pts = voxel_grid.grid_indices_to_pts(voxel_grid_idxs, centers=True)

        #bev grid idxs are the first 2 dims of the voxel idxs assuming matching metadata
        raster_idxs = voxel_grid_idxs[:, 0] * bev_grid.metadata.N[1] + voxel_grid_idxs[:, 1]
        porosity = voxel_grid.misses / (voxel_grid.hits + voxel_grid.misses)

        #scatter heights into grid
        num_cells = bev_grid.metadata.N[0] * bev_grid.metadata.N[1]

        bev_porosity = torch_scatter.scatter(src=porosity, index=raster_idxs, dim_size=num_cells, reduce=self.reduce)
        bev_grid.data[..., porosity_idx] = bev_porosity.view(*bev_grid.metadata.N)

        return bev_grid