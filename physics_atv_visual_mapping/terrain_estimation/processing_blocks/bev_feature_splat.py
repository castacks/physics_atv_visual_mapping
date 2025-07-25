import torch
import torch_scatter

from physics_atv_visual_mapping.terrain_estimation.processing_blocks.base import TerrainEstimationBlock

class BEVFeatureSplat(TerrainEstimationBlock):
    """
    Compute a per-cell min and max height
    """
    def __init__(self, metainfo_key, n_features, voxel_metadata, voxel_feature_keys, device):
        super().__init__(voxel_metadata, voxel_feature_keys, device)

        self.metainfo_key = metainfo_key
        self.output_features = voxel_feature_keys.filter_metainfo(self.metainfo_key)
        self.n_features = len(self.output_features) if n_features == -1 else n_features
        self.output_features = self.output_features[:self.n_features]

        self.voxel_to_bev_idxs = torch.tensor([voxel_feature_keys.index(k) for k in self.output_features.label], device=self.device)
        assert len(self.voxel_to_bev_idxs) > 0, f"couldnt find metainfo key {self.metainfo_key} in voxel featurekeylist {voxel_feature_keys}"

    def to(self, device):
        self.device = device
        self.voxel_to_bev_idxs = self.voxel_to_bev_idxs.to(self.device)
        return self

    @property
    def output_feature_keys(self):
        return self.output_features

    def run(self, voxel_grid, bev_grid):

        voxel_grid_idxs = voxel_grid.raster_indices_to_grid_indices(voxel_grid.feature_raster_indices)

        #bev grid idxs are the first 2 dims of the voxel idxs assuming matching metadata
        raster_idxs = voxel_grid_idxs[:, 0] * bev_grid.metadata.N[1] + voxel_grid_idxs[:, 1]
    
        num_cells = (bev_grid.metadata.N[0] * bev_grid.metadata.N[1]).item()

        features_to_scatter = voxel_grid.features[:, self.voxel_to_bev_idxs]

        bev_features = torch_scatter.scatter(src=features_to_scatter, index=raster_idxs, dim_size=num_cells, dim=0, reduce='mean')
        bev_features = bev_features.view(*bev_grid.metadata.N, self.n_features)

        bev_feature_idxs = [bev_grid.feature_keys.index(k) for k in self.output_feature_keys.label]
        bev_grid.data[..., bev_feature_idxs] = bev_features

        return bev_grid
