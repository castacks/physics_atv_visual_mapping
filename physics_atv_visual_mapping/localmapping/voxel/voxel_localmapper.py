import torch
import torch_scatter
import open3d as o3d

from numpy import pi as PI

from ros_torch_converter.datatypes.pointcloud import FeaturePointCloudTorch

from physics_atv_visual_mapping.localmapping.base import LocalMapper
from physics_atv_visual_mapping.utils import *

class VoxelLocalMapper(LocalMapper):
    """Class for local mapping voxels"""

    def __init__(self, metadata, n_features, ema, raytracer=None, device='cpu'):
        super().__init__(metadata, device)
        assert metadata.ndims == 3, "VoxelLocalMapper requires 3d metadata"
        self.voxel_grid = VoxelGrid(self.metadata, n_features, device)
        self.n_features = n_features
        self.raytracer = raytracer
        self.do_raytrace = self.raytracer is not None
        self.ema = ema

    def update_pose(self, pose: torch.Tensor):
        """
        Args:
            pose: [N] Tensor (we will take the first two elements as the new pose)
        """
        new_origin = (
            (pose[:3] + self.base_metadata.origin) // self.base_metadata.resolution
        ) * self.base_metadata.resolution
        self.voxel_grid.metadata = self.metadata

        px_shift = torch.round(
            (new_origin - self.metadata.origin) / self.metadata.resolution
        ).long()
        self.voxel_grid.shift(px_shift)
        self.metadata.origin = new_origin

    def add_feature_pc(self, pos: torch.Tensor, feat_pc: FeaturePointCloudTorch, do_raytrace=False, debug=False):
        voxel_grid_new = VoxelGrid.from_feature_pc(feat_pc, self.metadata, self.n_features)

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

        vg2_feat_buf = vg1_feat_buf.clone()
        vg2_feat_buf_mask = vg1_feat_buf_mask.clone()

        feat_buf = vg1_feat_buf.clone()

        #first copy over the original features
        vg1_feat_buf[vg1_feat_inv_idxs] += self.voxel_grid.features
        vg1_feat_buf_mask[vg1_feat_inv_idxs] = True

        vg2_feat_buf[vg2_feat_inv_idxs] += voxel_grid_new.features
        vg2_feat_buf_mask[vg2_feat_inv_idxs] = True

        #apply ema
        ema_mask = vg1_feat_buf_mask & vg2_feat_buf_mask
        feat_buf[vg1_feat_buf_mask & ~ema_mask] = vg1_feat_buf[vg1_feat_buf_mask & ~ema_mask]
        feat_buf[vg2_feat_buf_mask & ~ema_mask] = vg2_feat_buf[vg2_feat_buf_mask & ~ema_mask]
        feat_buf[ema_mask] = (1.-self.ema) * vg1_feat_buf[ema_mask] + self.ema * vg2_feat_buf[ema_mask]

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

        #compute passthrough rate
        passthrough_rate = self.voxel_grid.misses / (self.voxel_grid.hits + self.voxel_grid.misses)

        cull_mask = passthrough_rate > 0.75

        # print('culling {} voxels...'.format(cull_mask.sum()))

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
        self.voxel_grid.raster_indices = self.voxel_grid.raster_indices[~cull_mask]

        feat_cull_mask = cull_mask[self.voxel_grid.feature_mask]
        self.voxel_grid.features = self.voxel_grid.features[~feat_cull_mask]
        self.voxel_grid.feature_mask = self.voxel_grid.feature_mask[~cull_mask]

    def to(self, device):
        self.device = device
        self.voxel_grid = self.voxel_grid.to(device)
        self.metadata = self.metadata.to(device)
        if self.raytracer:
            self.raytracer = self.raytracer.to(device)
        return self


class VoxelGrid:
    """
    Actual class that handles feature aggregation
    """

    def from_feature_pc(feat_pc, metadata, n_features=-1):
        """
        Instantiate a VoxelGrid from a feauture pc

        Steps:
            1. separate out feature points and non-feature points
        """
        n_features = feat_pc.features.shape[-1] if n_features == -1 else n_features

        voxelgrid = VoxelGrid(metadata, n_features, feat_pc.device)

        feature_pts = feat_pc.feature_pts
        feature_pts_features = feat_pc.features[:, :n_features]
        non_feature_pts = feat_pc.non_feature_pts

        #first scatter and average the feature points

        grid_idxs, valid_mask = voxelgrid.get_grid_idxs(feature_pts)
        valid_grid_idxs = grid_idxs[valid_mask]
        valid_feats = feature_pts_features[valid_mask]

        valid_raster_idxs = voxelgrid.grid_indices_to_raster_indices(valid_grid_idxs)

        #NOTE: we need the voxel raster indices to be in ascending order (at least, within feat/no-feat) for stuff to work
        feature_raster_idxs, inv_idxs = torch.unique(
            valid_raster_idxs, return_inverse=True, sorted=True
        )
        
        feat_buf = torch_scatter.scatter(
            src=valid_feats, index=inv_idxs, dim_size=feature_raster_idxs.shape[0], reduce="mean", dim=0
        )

        #then add in non-feature points
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

        voxelgrid.raster_indices = all_raster_idxs
        voxelgrid.features = feat_buf
        voxelgrid.feature_mask = feat_mask

        voxelgrid.hits = torch.ones(all_raster_idxs.shape[0], device=voxelgrid.device)
        voxelgrid.misses = torch.zeros(all_raster_idxs.shape[0], device=voxelgrid.device)

        return voxelgrid

    # def from_pc(pts, metadata):
    #     voxelgrid = VoxelGrid(metadata, 0, pts.device)

    #     grid_idxs, valid_mask = voxelgrid.get_grid_idxs(pts)
    #     valid_grid_idxs = grid_idxs[valid_mask]

    #     valid_raster_idxs = voxelgrid.grid_indices_to_raster_indices(valid_grid_idxs)

    #     unique_raster_idxs, inv_idxs = torch.unique(
    #         valid_raster_idxs, return_inverse=True
    #     )

    #     voxelgrid.all_indices = unique_raster_idxs

    #     voxelgrid.hits = torch.ones(unique_raster_idxs.shape[0], device=voxelgrid.device)
    #     voxelgrid.misses = torch.zeros(unique_raster_idxs.shape[0], device=voxelgrid.device)

    #     return voxelgrid

    def __init__(self, metadata, n_features, device):
        self.metadata = metadata

        #raster indices of all points in voxel grid
        self.raster_indices = torch.zeros(0, dtype=torch.long, device=device)

        #list of features for all points in grid with features
        self.features = torch.zeros(0, n_features, dtype=torch.float, device=device)

        #mapping from indices to features (i.e. raster_indices[mask] = features)
        self.feature_mask = torch.zeros(0, dtype=torch.bool, device=device)

        self.hits = torch.zeros(0, dtype=torch.float, device=device) + 1e-8
        self.misses = torch.zeros(0, dtype=torch.float, device=device)

        self.device = device

    @property
    def non_feature_raster_indices(self):
        return self.raster_indices[~self.feature_mask]

    @property
    def feature_raster_indices(self):
        return self.raster_indices[self.feature_mask]

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

    def visualize(self, viz_all=True):
        pc = o3d.geometry.PointCloud()
        pts = self.grid_indices_to_pts(
            self.raster_indices_to_grid_indices(self.feature_raster_indices)
        )
        colors = normalize_dino(self.features[:, :3])

        #all_indices is a superset of indices
        if viz_all:
            non_colorized_idxs = self.non_feature_raster_indices

            non_colorized_pts = self.grid_indices_to_pts(
                self.raster_indices_to_grid_indices(non_colorized_idxs)
            )

            color_placeholder = 0.3 * torch.ones(non_colorized_pts.shape[0], 3, device=non_colorized_pts.device)

            pts = torch.cat([pts, non_colorized_pts], dim=0)
            colors = torch.cat([colors, color_placeholder], dim=0)

        pc.points = o3d.utility.Vector3dVector(pts.cpu().numpy())
        pc.colors = o3d.utility.Vector3dVector(colors.cpu().numpy())
        o3d.visualization.draw_geometries([pc])

    def to(self, device):
        self.device = device
        self.raster_indices = self.raster_indices.to(device)
        self.features = self.features.to(device)
        self.hits = self.hits.to(device)
        self.misses = self.misses.to(device)
        return self
