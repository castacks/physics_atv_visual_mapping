import torch
import torch_scatter

from physics_atv_visual_mapping.terrain_estimation.processing_blocks.base import TerrainEstimationBlock
from physics_atv_visual_mapping.feature_key_list import FeatureKeyList

class TerrainAwareBEVFeatureSplat(TerrainEstimationBlock):
    """
    Compute a per-cell min and max height
    """
    def __init__(self, voxel_metadata, voxel_feature_keys, terrain_layer, terrain_mask_layer, metainfo_key, overhang, n_features=-1, reduce='mean', device='cpu'):
        super().__init__(voxel_metadata, voxel_feature_keys, device)
        self.terrain_layer = terrain_layer
        self.terrain_mask_layer = terrain_mask_layer

        self.metainfo_key = metainfo_key
        self.output_features = voxel_feature_keys.filter_metainfo(self.metainfo_key)
        self.n_features = len(self.output_features) if n_features == -1 else n_features
        self.output_features = self.output_features[:self.n_features]

        self.voxel_to_bev_idxs = torch.tensor([voxel_feature_keys.index(k) for k in self.output_features.label], device=self.device)
        assert len(self.voxel_to_bev_idxs) > 0, f"couldnt find metainfo key {self.metainfo_key} in voxel featurekeylist {voxel_feature_keys}"

        self.overhang = overhang
        self.reduce = reduce
        
    def to(self, device):
        self.device = device
        self.voxel_to_bev_idxs = self.voxel_to_bev_idxs.to(self.device)
        return self

    @property
    def output_feature_keys(self):
        return self.output_features

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

        features_to_scatter = features_to_scatter[:, self.voxel_to_bev_idxs]
    
        num_cells = (bev_grid.metadata.N[0] * bev_grid.metadata.N[1]).item()

        bev_features = torch_scatter.scatter(src=features_to_scatter, index=idxs_to_scatter, dim_size=num_cells, dim=0, reduce=self.reduce)
        bev_features = bev_features.view(*bev_grid.metadata.N, self.n_features)

        bev_feature_idxs = [bev_grid.feature_keys.index(k) for k in self.output_feature_keys.label]
        bev_grid.data[..., bev_feature_idxs] = bev_features

        return bev_grid