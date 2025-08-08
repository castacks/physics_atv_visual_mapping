import torch
import torch_scatter

from scipy.ndimage import distance_transform_edt, binary_closing, binary_opening

from physics_atv_visual_mapping.terrain_estimation.processing_blocks.base import TerrainEstimationBlock
from physics_atv_visual_mapping.feature_key_list import FeatureKeyList

class SDF(TerrainEstimationBlock):
    """
    Compute signed distance function to a mask layer
        mask is computed as layer > thresh

    convention will be that stuff outside the mask has positive values and stuff
        inside the mask has negative values

    Also to my knowledge there is not a good implementation of SDF in torch, so
        we have to comvert to numpy and use scipy
    """
    def __init__(self, voxel_metadata, voxel_feature_keys, mask_layer, thresh, dilation, device):
        super().__init__(voxel_metadata, voxel_feature_keys, device)
        self.mask_layer = mask_layer
        self.thresh = thresh
        self.dilation = dilation

    def to(self, device):
        self.device = device
        return self

    @property
    def output_feature_keys(self):
        return FeatureKeyList(
            label = ["sdf"],
            metainfo = ["terrain_estimation"]
        )

    def run(self, voxel_grid, bev_grid):
        mask_idx = bev_grid.feature_keys.index(self.mask_layer)
        mask_data = bev_grid.data[..., mask_idx] > self.thresh
        mask_data = mask_data.cpu().numpy()

        if self.dilation:
            mask_data = binary_closing(mask_data)

        sampling = bev_grid.metadata.resolution.tolist()
        pos_sdf = distance_transform_edt(mask_data, sampling=sampling)
        neg_sdf = distance_transform_edt(1.-mask_data, sampling=sampling)

        neg_sdf[mask_data] = -pos_sdf[mask_data]

        sdf = torch.tensor(neg_sdf, dtype=torch.float, device=self.device)

        sdf_idx = bev_grid.feature_keys.index(self.output_feature_keys.label[0])
        bev_grid.data[..., sdf_idx] = sdf

        return bev_grid