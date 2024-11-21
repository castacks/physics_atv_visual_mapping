import torch
import torch_scatter
import open3d as o3d

from physics_atv_visual_mapping.localmapping.base import LocalMapper
from physics_atv_visual_mapping.utils import *


class VoxelLocalMapper(LocalMapper):
    """Class for local mapping voxels"""

    def __init__(self, metadata, n_features, ema, device):
        super().__init__(metadata, device)
        assert metadata.ndims == 3, "VoxelLocalMapper requires 3d metadata"
        self.voxel_grid = VoxelGrid(self.metadata, n_features, device)
        self.n_features = n_features
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

    def add_feature_pc(self, pts: torch.Tensor, features: torch.Tensor):
        voxel_grid_new = VoxelGrid.from_feature_pc(pts, features, self.metadata)

        all_raster_idxs = torch.cat([self.voxel_grid.indices, voxel_grid_new.indices])
        unique_idxs, inv_idxs, counts = torch.unique(
            all_raster_idxs, return_inverse=True, return_counts=True
        )
        feat_buf = torch.zeros(
            unique_idxs.shape[0],
            self.voxel_grid.features.shape[-1],
            device=self.voxel_grid.device,
        )

        # separate out idxs that are in 1 voxel grid vs both
        vg1_inv_idxs = inv_idxs[: self.voxel_grid.indices.shape[0]]
        vg1_is_collision = counts[vg1_inv_idxs] == 2
        vg2_inv_idxs = inv_idxs[self.voxel_grid.indices.shape[0] :]
        vg2_is_collision = counts[vg2_inv_idxs] == 2

        # passthrough features in just 1 grid
        feat_buf[vg1_inv_idxs[~vg1_is_collision]] += self.voxel_grid.features[
            ~vg1_is_collision
        ]
        feat_buf[vg2_inv_idxs[~vg2_is_collision]] += voxel_grid_new.features[
            ~vg2_is_collision
        ]

        # merge features in both with EMA
        feat_buf[vg1_inv_idxs[vg1_is_collision]] += (
            1.0 - self.ema
        ) * self.voxel_grid.features[vg1_is_collision]
        feat_buf[vg2_inv_idxs[vg2_is_collision]] += (
            self.ema * voxel_grid_new.features[vg2_is_collision]
        )

        self.voxel_grid.indices = unique_idxs
        self.voxel_grid.features = feat_buf

    def to(self, device):
        self.device = device
        self.bev_grid = self.bev_grid.to(device)
        self.metadata = self.metadata.to(device)
        return self


class VoxelGrid:
    """
    Actual class that handles feature aggregation
    """

    def from_feature_pc(pts, features, metadata):
        """
        Instantiate a VoxelGrid from a feauture pc
        """
        voxelgrid = VoxelGrid(metadata, features.shape[-1], features.device)

        grid_idxs, valid_mask = voxelgrid.get_grid_idxs(pts)
        valid_grid_idxs = grid_idxs[valid_mask]
        valid_feats = features[valid_mask]

        valid_raster_idxs = voxelgrid.grid_indices_to_raster_indices(valid_grid_idxs)

        unique_raster_idxs, inv_idxs = torch.unique(
            valid_raster_idxs, return_inverse=True
        )
        feat_buf = torch.zeros(
            unique_raster_idxs.shape[0], features.shape[-1], device=features.device
        )
        torch_scatter.scatter(
            src=valid_feats, index=inv_idxs, out=feat_buf, reduce="mean", dim=0
        )

        voxelgrid.indices = unique_raster_idxs
        voxelgrid.features = feat_buf

        return voxelgrid

    def __init__(self, metadata, n_features, device):
        self.metadata = metadata

        self.indices = torch.zeros(0, dtype=torch.long, device=device)
        self.features = torch.zeros(0, n_features, dtype=torch.float, device=device)

        self.device = device

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
        grid_indices = self.raster_indices_to_grid_indices(self.indices)
        grid_indices = grid_indices - px_shift.view(1, 3)
        mask = self.grid_idxs_in_bounds(grid_indices)
        self.indices = self.grid_indices_to_raster_indices(grid_indices[mask])
        self.features = self.features[mask]

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

    def visualize(self):
        pc = o3d.geometry.PointCloud()
        pts = self.grid_indices_to_pts(
            self.raster_indices_to_grid_indices(self.indices)
        )
        colors = normalize_dino(self.features[:, :3])
        pc.points = o3d.utility.Vector3dVector(pts.cpu().numpy())
        pc.colors = o3d.utility.Vector3dVector(colors.cpu().numpy())
        o3d.visualization.draw_geometries([pc])

    def to(self, device):
        self.device = device
        self.indices = self.indices.to(device)
        self.features = self.features.to(device)
        return self
