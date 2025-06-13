import torch
import torch_scatter

from physics_atv_visual_mapping.terrain_estimation.processing_blocks.base import TerrainEstimationBlock

class TraversabilityPrototypeScore(TerrainEstimationBlock):
    """
    Compute a per-cell min and max height
    """
    def __init__(self, voxel_metadata, voxel_n_features, vfm_feature_key, mask_key, prototype_fp, device='cpu'):
        super().__init__(voxel_metadata, voxel_n_features, device)
        self.vfm_feature_key = vfm_feature_key
        self.mask_key = mask_key

        prototypes = torch.load(prototype_fp)
        
        self.obstacle_keys = [pdata["label"] for pdata in prototypes["obstacle"]]
        self.nonobstacle_keys = [pdata["label"] for pdata in prototypes["nonobstacle"]]

        self.n_obstacle_ptypes = len(self.obstacle_keys)
        self.n_nonobstacle_ptypes = len(self.nonobstacle_keys)
        
    def to(self, device):
        self.device = device
        return self

    @property
    def output_keys(self):
        return ["obstacle_max_csim", "nonobstacle_max_csim", "cost"]

    def run(self, voxel_grid, bev_grid):
        vfm_fks = [x for x in bev_grid.feature_keys if self.vfm_feature_key in x]
        vfm_fidxs = [bev_grid.feature_keys.index(x) for x in vfm_fks]

        mask_idx = bev_grid.feature_keys.index(self.mask_key)

        res_idxs = [bev_grid.feature_keys.index(x) for x in self.output_keys]
        
        ptype_scores = bev_grid.data[..., vfm_fidxs]
        mask = bev_grid.data[..., mask_idx] > 1e-4

        obstacle_csim = ptype_scores[..., :self.n_obstacle_ptypes]
        nonobstacle_csim = ptype_scores[..., self.n_obstacle_ptypes:]

        obs_csim_max = obstacle_csim.max(dim=-1)[0]
        nonobs_csim_max = nonobstacle_csim.max(dim=-1)[0]

        #TODO figure out what the best rule here is
        cost = obs_csim_max - nonobs_csim_max

        cost[~mask] = cost.min()
        
        res_data = torch.stack([
            obs_csim_max,
            nonobs_csim_max,
            cost
        ], dim=-1)

        bev_grid.data[..., res_idxs] = res_data

        return bev_grid