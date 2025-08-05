import torch
import torch_scatter

from physics_atv_visual_mapping.terrain_estimation.processing_blocks.base import TerrainEstimationBlock
from physics_atv_visual_mapping.feature_key_list import FeatureKeyList

class ElevationStats(TerrainEstimationBlock):
    """
    Compute a per-cell min and max height
    """
    def __init__(self, voxel_metadata, voxel_feature_keys, use_voxel_centers=False, device='cpu'):
        """
        Args:
            use_voxel_centers: if False, use the min/max z stored in the voxels to compute elev. if True, use the voxel center
                (in 99% of cases, False is better)
        """
        super().__init__(voxel_metadata, voxel_feature_keys, device)
        self.use_voxel_centers = use_voxel_centers

    def to(self, device):
        self.device = device
        return self

    @property
    def output_feature_keys(self):
        return FeatureKeyList(
            label = ["min_elevation", "mean_elevation", "max_elevation", "num_voxels"],
            metainfo = ["terrain_estimation"] * 4
        )

    def run(self, voxel_grid, bev_grid):
        #keys can vary so we need to recompute them here
        min_height_idx = bev_grid.feature_keys.index(self.output_feature_keys.label[0])
        mean_height_idx = bev_grid.feature_keys.index(self.output_feature_keys.label[1])
        max_height_idx = bev_grid.feature_keys.index(self.output_feature_keys.label[2])
        num_voxels_idx = bev_grid.feature_keys.index(self.output_feature_keys.label[3])

        #get grid idxs and coordinates of voxel grid
        voxel_grid_idxs = voxel_grid.raster_indices_to_grid_indices(voxel_grid.raster_indices)
        voxel_grid_pts = voxel_grid.grid_indices_to_pts(voxel_grid_idxs, centers=True)

        #bev grid idxs are the first 2 dims of the voxel idxs assuming matching metadata
        raster_idxs = voxel_grid_idxs[:, 0] * bev_grid.metadata.N[1] + voxel_grid_idxs[:, 1]

        if self.use_voxel_centers:
            min_heights = voxel_grid_pts[:, 2]
            mean_heights = voxel_grid_pts[:, 2]
            max_heights = voxel_grid_pts[:, 2]
        else:
            min_heights = voxel_grid.min_coords[:, 2]
            mean_heights = voxel_grid.midpoints[:, 2]
            max_heights = voxel_grid.max_coords[:, 2]

        #scatter heights into grid
        num_cells = bev_grid.metadata.N[0] * bev_grid.metadata.N[1]

        min_height = torch_scatter.scatter(src=min_heights, index=raster_idxs, dim_size=num_cells, reduce='min')
        bev_grid.data[..., min_height_idx] = min_height.view(*bev_grid.metadata.N)

        mean_height = torch_scatter.scatter(src=mean_heights, index=raster_idxs, dim_size=num_cells, reduce='mean')
        bev_grid.data[..., mean_height_idx] = mean_height.view(*bev_grid.metadata.N)

        max_height = torch_scatter.scatter(src=max_heights, index=raster_idxs, dim_size=num_cells, reduce='max')
        bev_grid.data[..., max_height_idx] = max_height.view(*bev_grid.metadata.N)

        num_voxels = torch_scatter.scatter(src=torch.ones_like(min_heights), index=raster_idxs, dim_size=num_cells, reduce='sum')
        bev_grid.data[..., num_voxels_idx] = num_voxels.view(*bev_grid.metadata.N)

        return bev_grid