import torch
import torch_scatter
import open3d as o3d

from numpy import pi as PI

from ros_torch_converter.datatypes.pointcloud import FeaturePointCloudTorch

from physics_atv_visual_mapping.localmapping.base import LocalMapper
from physics_atv_visual_mapping.localmapping.metadata import LocalMapperMetadata
from physics_atv_visual_mapping.feature_key_list import FeatureKeyList
from physics_atv_visual_mapping.utils import *

class VoxelLocalMapper(LocalMapper):
    """Class for local mapping voxels"""

    def __init__(self, metadata, feature_keys, ema, raytracer=None, n_features=-1, device='cpu'):
        super().__init__(metadata, device)
        assert metadata.ndims == 3, "VoxelLocalMapper requires 3d metadata"
        self.n_features = len(feature_keys) if n_features == -1 else n_features
        self.feature_keys = feature_keys[:self.n_features]

        assert ema >= 0. and ema <= 1.

        self.voxel_grid = VoxelGrid(self.metadata, self.feature_keys, device)
        self.raytracer = raytracer
        self.do_raytrace = self.raytracer is not None
        self.ema = ema
        self.assert_feat_match = True

    def update_pose(self, pose: torch.Tensor):
        """
        Args:
            pose: [N] Tensor (we will take the first two elements as the new pose)
        """
        #keep origin a multiple of resolution
        new_origin = torch.round((pose[:3] + self.base_metadata.origin)/self.base_metadata.resolution) * self.base_metadata.resolution
        self.voxel_grid.metadata = self.metadata

        px_shift = torch.round(
            (new_origin - self.metadata.origin) / self.metadata.resolution
        ).long()
        self.voxel_grid.shift(px_shift)
        self.metadata.origin = new_origin

    def add_feature_pc(self, pos: torch.Tensor, feat_pc: FeaturePointCloudTorch, do_raytrace=False, debug=False):
        voxel_grid_new = VoxelGrid.from_feature_pc(feat_pc, self.metadata, self.n_features, pos, strategy='mindist')

        if self.assert_feat_match:
            assert self.voxel_grid.feature_keys == voxel_grid_new.feature_keys, f"voxel feat key mismatch: mapper has {self.voxel_grid.feature_keys}, added pc has {voxel_grid_new.feature_keys}"

        if self.do_raytrace:
            # self.raytracer.raytrace(pos, voxel_grid_meas=voxel_grid_new, voxel_grid_agg=self.voxel_grid)
            self.raytracer.raytrace_but_better(pos, pc_meas=feat_pc, voxel_grid_agg=self.voxel_grid)

        #first map all indices with features
        all_raster_idxs = torch.cat([self.voxel_grid.raster_indices, voxel_grid_new.raster_indices])
        unique_raster_idxs, inv_idxs, counts = torch.unique(
            all_raster_idxs, return_inverse=True, return_counts=True, sorted=True
        )

        # we need an index into both the full set of idxs and also the feature buffer
        # note that since we're sorting by raster index, we can align the two buffers
        all_feature_raster_idxs = torch.cat([self.voxel_grid.feature_raster_indices, voxel_grid_new.feature_raster_indices])
        unique_feature_raster_idxs, feat_inv_idxs = torch.unique(all_feature_raster_idxs, return_inverse=True, sorted=True)

        # separate out idxs that are in 1 voxel grid vs both
        vg1_inv_idxs = inv_idxs[: self.voxel_grid.raster_indices.shape[0]] #index from vg1 idxs to aggregated buffer
        vg1_feat_inv_idxs = feat_inv_idxs[:self.voxel_grid.feature_raster_indices.shape[0]] #index from feature idxs into aggregated feature buffer
        vg1_has_feature = self.voxel_grid.feature_mask #this line requires that the voxel grid is sorted by raster idx

        vg2_inv_idxs = inv_idxs[self.voxel_grid.raster_indices.shape[0] :]
        vg2_feat_inv_idxs = feat_inv_idxs[self.voxel_grid.feature_raster_indices.shape[0]:]
        vg2_has_feature = voxel_grid_new.feature_mask #this line requires that the voxel grid is sorted by raster idx

        vg1_feat_buf = torch.zeros(
            unique_feature_raster_idxs.shape[0],
            self.voxel_grid.features.shape[-1],
            device=self.voxel_grid.device,
        )
        vg1_feat_buf_mask = torch.zeros(vg1_feat_buf.shape[0], dtype=torch.bool, device=self.voxel_grid.device)

        vg2_feat_buf = torch.zeros(
            unique_feature_raster_idxs.shape[0],
            voxel_grid_new.features.shape[-1],
            device=self.voxel_grid.device,
        )
        vg2_feat_buf_mask = vg1_feat_buf_mask.clone()

        #first copy over the original features
        vg1_feat_buf[vg1_feat_inv_idxs] += self.voxel_grid.features
        vg1_feat_buf_mask[vg1_feat_inv_idxs] = True

        vg2_feat_buf[vg2_feat_inv_idxs] += voxel_grid_new.features
        vg2_feat_buf_mask[vg2_feat_inv_idxs] = True

        #apply ema
        feat_buf = self._merge_voxel_features(vg1_feat_buf, vg1_feat_buf_mask, vg2_feat_buf, vg2_feat_buf_mask)

        #ok now i have the merged features and the final raster idxs. need to make the mask
        feature_mask = torch.zeros(unique_raster_idxs.shape[0], dtype=torch.bool, device=self.voxel_grid.device)
        feature_mask[vg1_inv_idxs] = (feature_mask[vg1_inv_idxs] | vg1_has_feature) 
        feature_mask[vg2_inv_idxs] = (feature_mask[vg2_inv_idxs] | vg2_has_feature)

        self.voxel_grid.raster_indices = unique_raster_idxs
        self.voxel_grid.features = feat_buf
        self.voxel_grid.feature_mask = feature_mask

        hit_buf = torch.zeros(
            unique_raster_idxs.shape[0],
            device=self.voxel_grid.device,
        )
        hit_buf[vg1_inv_idxs] += self.voxel_grid.hits
        hit_buf[vg2_inv_idxs] += voxel_grid_new.hits

        miss_buf = torch.zeros(
            unique_raster_idxs.shape[0],
            device=self.voxel_grid.device,
        )
        miss_buf[vg1_inv_idxs] += self.voxel_grid.misses
        miss_buf[vg2_inv_idxs] += voxel_grid_new.misses

        self.voxel_grid.hits = hit_buf
        self.voxel_grid.misses = miss_buf

        min_coords_buf = 1e10 * torch.ones(unique_raster_idxs.shape[0], 3, device=self.voxel_grid.device)
        min_coords_buf[vg1_inv_idxs] = torch.minimum(min_coords_buf[vg1_inv_idxs], self.voxel_grid.min_coords)
        min_coords_buf[vg2_inv_idxs] = torch.minimum(min_coords_buf[vg2_inv_idxs], voxel_grid_new.min_coords)
        self.voxel_grid.min_coords = min_coords_buf

        max_coords_buf = -1e10 * torch.ones(unique_raster_idxs.shape[0], 3, device=self.voxel_grid.device)
        max_coords_buf[vg1_inv_idxs] = torch.maximum(max_coords_buf[vg1_inv_idxs], self.voxel_grid.max_coords)
        max_coords_buf[vg2_inv_idxs] = torch.maximum(max_coords_buf[vg2_inv_idxs], voxel_grid_new.max_coords)
        self.voxel_grid.max_coords = max_coords_buf

        #compute passthrough rate
        passthrough_rate = self.voxel_grid.misses / (self.voxel_grid.hits + self.voxel_grid.misses)

        cull_mask = passthrough_rate > 0.75

        # print('culling {} voxels...'.format(cull_mask.sum()))

        new_grid_idxs = self.voxel_grid.raster_indices_to_grid_indices(self.voxel_grid.raster_indices)
        new_grid_idxs_bev = new_grid_idxs[:, :2]
        new_grid_idxs_z = new_grid_idxs[:, 2]
        new_grid_idxs_bev_raster = new_grid_idxs[:, 0] * self.voxel_grid.metadata.N[1] + new_grid_idxs[:, 1]
        new_minz = torch_scatter.scatter(new_grid_idxs_z, index=new_grid_idxs_bev_raster, reduce="min")
        bottom_voxel_mask = new_grid_idxs_z <= new_minz[new_grid_idxs_bev_raster]

        cull_mask = cull_mask & ~bottom_voxel_mask

        if debug:
            import open3d as o3d
            pts = self.voxel_grid.grid_indices_to_pts(self.voxel_grid.raster_indices_to_grid_indices(self.voxel_grid.raster_indices))
            #solid=black, porous=green, cull=red
            colors = torch.stack([torch.zeros_like(passthrough_rate), passthrough_rate, torch.zeros_like(passthrough_rate)], dim=-1)
            colors[cull_mask] = torch.tensor([1., 0., 0.], device=self.device)
            porosity_pc = o3d.geometry.PointCloud()
            porosity_pc.points = o3d.utility.Vector3dVector(pts.cpu().numpy())
            porosity_pc.colors = o3d.utility.Vector3dVector(colors.cpu().numpy())
            o3d.visualization.draw_geometries([porosity_pc])

        self.voxel_grid.hits = self.voxel_grid.hits[~cull_mask]
        self.voxel_grid.misses = self.voxel_grid.misses[~cull_mask]
        self.voxel_grid.min_coords = self.voxel_grid.min_coords[~cull_mask]
        self.voxel_grid.max_coords = self.voxel_grid.max_coords[~cull_mask]
        self.voxel_grid.raster_indices = self.voxel_grid.raster_indices[~cull_mask]

        feat_cull_mask = cull_mask[self.voxel_grid.feature_mask]
        self.voxel_grid.features = self.voxel_grid.features[~feat_cull_mask]
        self.voxel_grid.feature_mask = self.voxel_grid.feature_mask[~cull_mask]

    def _merge_voxel_features(self, vg1_feats, vg1_mask, vg2_feats, vg2_mask):
        """
        Merge voxel grid features together with EMA
        Args:
            vg1_feats: [NxF] Tensor of raster feats from voxel grid 1. Note that this should be raster-padded w/ vg2
                (i.e. this is after the cat+uniq op)
            vg1_mask:  [N] Tensor which is True iff. that raster idx has a feature in vg1
            vg2_feats: Same as vg1_feats for vg2
            vg2_mask:  Same as vg1_mask for vg2
        Returns:
            feats_out: [NxF] Tensor of merged features for all unique raster indices
        """
        feats_out = vg1_feats.clone()
        merge_mask = vg1_mask & vg2_mask
        feats_out[vg1_mask & ~merge_mask] = vg1_feats[vg1_mask & ~merge_mask]
        feats_out[vg2_mask & ~merge_mask] = vg2_feats[vg2_mask & ~merge_mask]
        feats_out[merge_mask] = (1.-self.ema) * vg1_feats[merge_mask] + self.ema * vg2_feats[merge_mask]

        return feats_out

    def to(self, device):
        self.device = device
        self.voxel_grid = self.voxel_grid.to(device)
        self.metadata = self.metadata.to(device)
        if self.raytracer:
            self.raytracer = self.raytracer.to(device)
        return self

class VoxelCoordinateLocalMapper(VoxelLocalMapper):
    """
    Voxel localmapper class that maps image coordinates instead of features directly
    This requires us to implement a slightly different merge rule (list update instead of EMA)
    """
    def __init__(self, metadata, feature_keys, ema, raytracer=None, n_features=-1, max_n_coords=10, device='cpu'):
        """
        Args:
            max_n_coords: the maximum amount of image feats that a voxel can pull from
                note that with 0.5 ema, 7 coords is sufficient for >1% err
        """
        self.input_feature_keys = feature_keys
        self.output_feature_keys = []
        for k in ['seq', 'cam', 'u', 'v', 'w']:
            for i in range(max_n_coords):
                self.output_feature_keys.append(f"{k}_{i}")
        self.output_feature_keys = FeatureKeyList(
            label=self.output_feature_keys,
            metainfo=["img_coords"] * len(self.output_feature_keys)
        )

        super().__init__(metadata, self.output_feature_keys, ema, raytracer, n_features, device)
        self.assert_feat_match = False
        self.max_n_coords = max_n_coords

    def _merge_voxel_features(self, vg1_feats, vg1_mask, vg2_feats, vg2_mask):
        """
        Merge voxel grid features together with EMA
        Args:
            vg1_feats: [NxF] Tensor of raster feats from voxel grid 1. Note that this should be raster-padded w/ vg2
                (i.e. this is after the cat+uniq op)
            vg1_mask:  [N] Tensor which is True iff. that raster idx has a feature in vg1
            vg2_feats: Same as vg1_feats for vg2
            vg2_mask:  Same as vg1_mask for vg2
        Returns:
            feats_out: [NxF] Tensor of merged features for all unique raster indices
        """
        #[N x ncams x 5]
        vg1_coords = vg1_feats.view(vg1_feats.shape[0], 5, self.max_n_coords).permute(0,2,1)

        #[N x ncoords x 5]
        vg2_coords = vg2_feats.view(vg2_feats.shape[0], 5, -1).permute(0,2,1)
        
        #apply ema to weights (note that placeholder weight MUST be 0)
        vg1_coords[:, :, -1] = (vg1_coords[:, :, -1] * (1.-self.ema))
        vg2_coords[:, :, -1] = (vg2_coords[:, :, -1] * self.ema)

        #[N x (ncams+ncoords) x 5]
        cat_coords = torch.cat([vg1_coords, vg2_coords], dim=1)

        #[N x (ncams+ncoords)]
        idxs = cat_coords[:, :, 4].argsort(dim=-1, descending=True)

        idxs = idxs[:, :self.max_n_coords]

        _B = torch.arange(idxs.shape[0], device=self.device).unsqueeze(-1)
        cat_coords = cat_coords[_B, idxs] #[N x ncoords x 5]

        weight_sum = cat_coords[:, :, 4].sum(dim=-1, keepdim=True)
        cat_coords[:, :, 4] = cat_coords[:, :, 4] / weight_sum

        feats_out = cat_coords.permute(0,2,1).reshape(-1, 5*self.max_n_coords)

        return feats_out

class VoxelGrid:
    """
    Actual class that handles feature aggregation
    """
    def _get_voxel_features_mindist(voxelgrid, pts, features, pos):
        grid_idxs, valid_mask = voxelgrid.get_grid_idxs(pts)
        dists = torch.linalg.norm(pts - pos[:3].view(1,3), dim=-1)

        valid_grid_idxs = grid_idxs[valid_mask]
        valid_feats = features[valid_mask]
        valid_dists = dists[valid_mask]

        valid_raster_idxs = voxelgrid.grid_indices_to_raster_indices(valid_grid_idxs)
        
        sort_idxs = valid_raster_idxs.argsort()

        valid_raster_idxs = valid_raster_idxs[sort_idxs]
        valid_feats = valid_feats[sort_idxs]
        valid_dists = valid_dists[sort_idxs]
        feature_raster_idxs, inv_idxs = valid_raster_idxs.unique(return_inverse=True, sorted=True)

        raster_mindists = torch_scatter.scatter(
            src=valid_dists,
            index=inv_idxs,
            dim_size=feature_raster_idxs.shape[0],
            reduce='min'
        )

        voxel_is_mindist = (valid_dists - raster_mindists[inv_idxs]).abs() < 1e-16
        if voxel_is_mindist.sum() != feature_raster_idxs.shape[0]:
            if voxel_is_mindist.sum() < feature_raster_idxs.shape[0]:
                print("uh oh not all voxels have a mindist")
                import pdb;pdb.set_trace()

        voxel_is_mindist = voxel_is_mindist.float().unsqueeze(-1)

        feat_buf = torch_scatter.scatter(
            src=valid_feats * voxel_is_mindist,
            index=inv_idxs,
            dim_size=feature_raster_idxs.shape[0],
            reduce="sum",
            dim=0
        )

        ## need to do an extra scatter of the mask to handle cases where voxel_is_mindist True for multiple pts in voxel
        ## e.g. if the same point is in there twice
        ## hopefully same spatial pos -> same feature, otherwise tough luck :)
        cnt = torch_scatter.scatter(
            src=voxel_is_mindist,
            index=inv_idxs,
            dim_size=feature_raster_idxs.shape[0],
            reduce="sum",
            dim=0
        )

        voxel_feats = feat_buf / cnt

        return feature_raster_idxs, voxel_feats

    def _get_voxel_features_scatter(voxelgrid, pts, features):
        grid_idxs, valid_mask = voxelgrid.get_grid_idxs(pts)
        valid_grid_idxs = grid_idxs[valid_mask]
        valid_feats = features[valid_mask]

        valid_raster_idxs = voxelgrid.grid_indices_to_raster_indices(valid_grid_idxs)

        #NOTE: we need the voxel raster indices to be in ascending order (at least, within feat/no-feat) for stuff to work
        feature_raster_idxs, inv_idxs = torch.unique(
            valid_raster_idxs, return_inverse=True, sorted=True
        )
        
        feat_buf = torch_scatter.scatter(
            src=valid_feats, index=inv_idxs, dim_size=feature_raster_idxs.shape[0], reduce="mean", dim=0
        )

        return feature_raster_idxs, feat_buf

    def from_feature_pc(feat_pc, metadata, n_features=-1, pos=None, strategy='avg'):
        """
        Instantiate a VoxelGrid from a feauture pc

        Steps:
            1. separate out feature points and non-feature points
        """
        n_features = len(feat_pc.feature_keys) if n_features == -1 else n_features
        feat_keys = feat_pc.feature_keys[:n_features]

        voxelgrid = VoxelGrid(metadata, feat_keys, feat_pc.device)

        feature_pts = feat_pc.feature_pts.clone()
        feature_pts_features = feat_pc.features[:, :n_features].clone()

        #first scatter and average the feature points
        if strategy == 'avg':
            feature_raster_idxs, voxel_features = VoxelGrid._get_voxel_features_scatter(
                voxelgrid, feature_pts, feature_pts_features
            )
        elif strategy == 'mindist':
            feature_raster_idxs, voxel_features = VoxelGrid._get_voxel_features_mindist(
                voxelgrid, feature_pts, feature_pts_features, pos
                # voxelgrid.to('cpu'), feature_pts.to('cpu'), feature_pts_features.to('cpu'), pos.to('cpu')
            )

        # voxelgrid = voxelgrid.to('cuda')
        # feature_raster_idxs = feature_raster_idxs.to('cuda')
        # voxel_features = voxel_features.to('cuda')

        #then add in non-feature points
        non_feature_pts = feat_pc.non_feature_pts.clone()
        grid_idxs, valid_mask = voxelgrid.get_grid_idxs(non_feature_pts)
        valid_grid_idxs = grid_idxs[valid_mask]
        valid_raster_idxs = voxelgrid.grid_indices_to_raster_indices(valid_grid_idxs)
        non_feature_raster_idxs = torch.unique(valid_raster_idxs)

        _raster_idxs_cnt_in = torch.cat([feature_raster_idxs, feature_raster_idxs, non_feature_raster_idxs])
        _raster_idxs, _raster_idx_cnts = torch.unique(_raster_idxs_cnt_in, return_counts=True)
        non_feature_raster_idxs = _raster_idxs[_raster_idx_cnts == 1]

        #store in voxel grid
        n_feat_voxels = feature_raster_idxs.shape[0]
        all_raster_idxs = torch.cat([feature_raster_idxs, non_feature_raster_idxs])
        feat_mask = torch.zeros(all_raster_idxs.shape[0], dtype=torch.bool, device=feat_pc.device)
        feat_mask[:n_feat_voxels] = True

        #I need raster idxs to be sorted to do other downstream ops
        all_raster_idxs, idxs = torch.sort(all_raster_idxs)
        feat_mask = feat_mask[idxs]

        voxelgrid.raster_indices = all_raster_idxs
        voxelgrid.features = voxel_features
        voxelgrid.feature_mask = feat_mask

        voxelgrid.hits = torch.ones(all_raster_idxs.shape[0], device=voxelgrid.device)
        voxelgrid.misses = torch.zeros(all_raster_idxs.shape[0], device=voxelgrid.device)

        #scatter min/max coords
        all_pts = feat_pc.pts
        all_grid_idxs, valid_mask = voxelgrid.get_grid_idxs(all_pts)
        all_valid_pts = all_pts[valid_mask]
        all_pts_raster_idxs = voxelgrid.grid_indices_to_raster_indices(all_grid_idxs[valid_mask])

        pt_raster_idxs, inv_idxs = torch.unique(all_pts_raster_idxs, return_inverse=True, sorted=True)

        voxelgrid.min_coords = torch_scatter.scatter(
            src=all_valid_pts, index=inv_idxs, dim_size=all_raster_idxs.shape[0], reduce="min", dim=0
        )

        voxelgrid.max_coords = torch_scatter.scatter(
            src=all_valid_pts, index=inv_idxs, dim_size=all_raster_idxs.shape[0], reduce="max", dim=0
        )

        return voxelgrid

    def __init__(self, metadata, feature_keys, device):
        self.metadata = metadata
        self.feature_keys = feature_keys
        self.n_features = len(self.feature_keys)

        #raster indices of all points in voxel grid
        self.raster_indices = torch.zeros(0, dtype=torch.long, device=device)

        #list of features for all points in grid with features
        self.features = torch.zeros(0, self.n_features, dtype=torch.float, device=device)

        #mapping from indices to features (i.e. raster_indices[mask] = features)
        self.feature_mask = torch.zeros(0, dtype=torch.bool, device=device)

        self.hits = torch.zeros(0, dtype=torch.float, device=device) + 1e-8
        self.misses = torch.zeros(0, dtype=torch.float, device=device)

        #store min/max coords per voxel for more accurate reconstruction
        self.min_coords = torch.zeros(0, 3, dtype=torch.float, device=device)
        self.max_coords = torch.zeros(0, 3, dtype=torch.float, device=device)

        self.device = device

    @property
    def non_feature_raster_indices(self):
        return self.raster_indices[~self.feature_mask]

    @property
    def feature_raster_indices(self):
        return self.raster_indices[self.feature_mask]

    @property
    def midpoints(self):
        """
        Return the midpoints of all occupied voxels
            (i.e. (max coords + min_coords) / 2.)
        """
        return 0.5 * (self.min_coords + self.max_coords)

    @property
    def feature_midpoints(self):
        return 0.5 * (self.min_coords[self.feature_mask] + self.max_coords[self.feature_mask])

    @property
    def non_feature_midpoints(self):
        return 0.5 * (self.min_coords[~self.feature_mask] + self.max_coords[~self.feature_mask])

    def get_grid_idxs(self, pts):
        """
        Get indexes for positions given map metadata
        """
        gidxs = torch.div((pts[:, :3] - self.metadata.origin.view(1,3)), self.metadata.resolution.view(1,3), rounding_mode='floor').long()
        mask = (gidxs >=  0).all(dim=-1) & (gidxs < self.metadata.N.view(1,3)).all(dim=-1) 
        return gidxs, mask

    def shift(self, px_shift):
        """
        Apply a pixel shift to the map

        Args:
            px_shift: Tensor of [dx, dy, dz], where the ORIGIN of the map is moved by this many cells
                e.g. if px_shift is [-3, 5], the new origin is 3*res units left and 5*res units up
                        note that this means the data is shifted 3 cells right and 5 cells down
        """
        #shift feature indices
        grid_indices = self.raster_indices_to_grid_indices(self.raster_indices)
        grid_indices = grid_indices - px_shift.view(1, 3)
        mask = self.grid_idxs_in_bounds(grid_indices)
        self.raster_indices = self.grid_indices_to_raster_indices(grid_indices[mask])

        self.features = self.features[mask[self.feature_mask]]
        self.feature_mask = self.feature_mask[mask]
        self.hits = self.hits[mask]
        self.misses = self.misses[mask]
        self.min_coords = self.min_coords[mask]
        self.max_coords = self.max_coords[mask]

        self.metadata.origin += px_shift * self.metadata.resolution

    def pts_in_bounds(self, pts):
        """Check if points are in bounds

        Args:
            pts: [Nx3] Tensor of coordinates

        Returns:
            valid: [N] mask of whether point is within voxel grid
        """
        _min = self.metadata.origin.view(1, 3)
        _max = (self.metadata.origin + self.metadata.length).view(1, 3)

        low_check = (pts >= _min).all(axis=-1)
        high_check = (pts < _max).all(axis=-1)
        return low_check & high_check

    def grid_idxs_in_bounds(self, grid_idxs):
        """Check if grid idxs are in bounds

        Args:
            grid_idxs: [Nx3] Long Tensor of grid idxs

        Returns:
            valid: [N] mask of whether idx is within voxel grid
        """
        _min = torch.zeros_like(self.metadata.N).view(1, 3)
        _max = self.metadata.N.view(1, 3)

        low_check = (grid_idxs >= _min).all(axis=-1)
        high_check = (grid_idxs < _max).all(axis=-1)

        return low_check & high_check

    def grid_indices_to_pts(self, grid_indices, centers=True):
        """Convert a set of grid coordinates to cartesian coordinates

        Args:
            grid_indices: [Nx3] Tensor of grid coordinates
            centers: Set this flag to false to return voxel lower-bottom-left, else return voxel centers
        """
        coords = grid_indices * self.metadata.resolution.view(
            1, 3
        ) + self.metadata.origin.view(1, 3)

        if centers:
            coords += (self.metadata.resolution / 2.0).view(1, 3)

        return coords

    def grid_indices_to_raster_indices(self, grid_idxs):
        """Convert a set of grid indices to raster indices

        Args:
            grid_idxs: [Nx3] Long Tensor of grid indices

        Returns:
            raster_idxs: [N] Long Tensor of raster indices
        """
        _N1 = self.metadata.N[1] * self.metadata.N[2]
        _N2 = self.metadata.N[2]

        return _N1 * grid_idxs[:, 0] + _N2 * grid_idxs[:, 1] + grid_idxs[:, 2]

    def raster_indices_to_grid_indices(self, raster_idxs):
        """Convert a set of raster indices to grid indices

        Args:
            raster_idxs: [N] Long Tensor of raster indices

        Returns:
            grid_idxs: [Nx3] Long Tensor of grid indices
        """
        _N1 = self.metadata.N[1] * self.metadata.N[2]
        _N2 = self.metadata.N[2]

        xs = torch.div(raster_idxs, _N1, rounding_mode="floor").long()
        ys = torch.div(raster_idxs % _N1, _N2, rounding_mode="floor").long()
        zs = raster_idxs % _N2

        return torch.stack([xs, ys, zs], axis=-1)

    def random_init(device='cpu'):
        metadata = LocalMapperMetadata.random_init(ndim=3, device=device)

        fks = FeatureKeyList(
            label=[f'rand_{i}' for i in range(5)],
            metainfo=['feat'] * 5
        )

        maxn = torch.prod(metadata.N)
        nvox = int(maxn/2)
        nfeats = int(nvox/2)

        raster_idxs = torch.randperm(maxn, device=device)[:nvox]
        feats = torch.randn(nfeats, 5, device=device)

        mask_idxs = torch.randperm(nvox)[:nfeats]
        mask = torch.zeros(len(raster_idxs), dtype=torch.bool, device=device)
        mask[mask_idxs] = True

        hits = (10. * torch.rand(nvox, device=device)).long()
        misses = (10. * torch.rand(nvox, device=device)).long()

        voxel_grid = VoxelGrid(metadata, fks, device=device)
        voxel_grid.raster_indices = raster_idxs
        voxel_grid.feature_mask = mask
        voxel_grid.features = feats
        voxel_grid.hits = hits
        voxel_grid.misses = misses

        voxel_centers = voxel_grid.grid_indices_to_pts(voxel_grid.raster_indices_to_grid_indices(raster_idxs))
        voxel_grid.min_coords = voxel_centers - metadata.resolution/2.
        voxel_grid.max_coords = voxel_centers + metadata.resolution/2.

        return voxel_grid

    def __eq__(self, other):
        if self.feature_keys != other.feature_keys:
            return False

        if self.metadata != other.metadata:
            return False

        if not (self.raster_indices == other.raster_indices).all():
            return False

        if not (self.feature_mask == other.feature_mask).all():
            return False

        if not torch.allclose(self.features, other.features):
            return False

        if not (self.hits == other.hits).all():
            return False

        if not (self.misses == other.misses).all():
            return False

        if not torch.allclose(self.min_coords, other.min_coords):
            return False

        if not torch.allclose(self.max_coords, other.max_coords):
            return False

        return True

    def visualize(self, viz_all=True, midpoints=True, sample_frac=1.0):
        pc = o3d.geometry.PointCloud()
        if midpoints:
            pts = self.feature_midpoints
        else:
            pts = self.grid_indices_to_pts(
                self.raster_indices_to_grid_indices(self.feature_raster_indices)
            )
        colors = normalize_dino(self.features[:, :3])

        #all_indices is a superset of indices
        if viz_all:
            if midpoints:
                non_colorized_pts = self.non_feature_midpoints
            else:
                non_colorized_idxs = self.non_feature_raster_indices

                non_colorized_pts = self.grid_indices_to_pts(
                    self.raster_indices_to_grid_indices(non_colorized_idxs)
                )

            color_placeholder = 0.3 * torch.ones(non_colorized_pts.shape[0], 3, device=non_colorized_pts.device)

            pts = torch.cat([pts, non_colorized_pts], dim=0)
            colors = torch.cat([colors, color_placeholder], dim=0)

        pc.points = o3d.utility.Vector3dVector(pts.cpu().numpy())
        pc.colors = o3d.utility.Vector3dVector(colors.cpu().numpy())

        if sample_frac < 1.0:
            pc = pc.uniform_down_sample(every_k_points=int(1/sample_frac))

        return pc
        # o3d.visualization.draw_geometries([pc])

    def to(self, device):
        self.device = device
        self.metadata = self.metadata.to(device)
        self.raster_indices = self.raster_indices.to(device)
        self.feature_mask = self.feature_mask.to(device)
        self.features = self.features.to(device)
        self.hits = self.hits.to(device)
        self.misses = self.misses.to(device)
        self.min_coords = self.min_coords.to(device)
        self.max_coords = self.max_coords.to(device)
        return self
