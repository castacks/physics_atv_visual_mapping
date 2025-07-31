import torch
import torch_scatter
import matplotlib.pyplot as plt

from ros_torch_converter.datatypes.pointcloud import FeaturePointCloudTorch

from physics_atv_visual_mapping.localmapping.base import LocalMapper
from physics_atv_visual_mapping.feature_key_list import FeatureKeyList
from physics_atv_visual_mapping.utils import *

#similar to Fankhauser et al 2018
ELEVATION_MAP_FEATURE_KEYS = FeatureKeyList(
    label=["min_elevation", "max_elevation", "elevation_var"],
    metainfo=["terrain_estimation"] * 3
)

class BEVLocalMapper(LocalMapper):
    """Class for local mapping in BEV"""

    def __init__(self, metadata, feature_keys, n_features, ema, overhang, device,):
        super().__init__(metadata, device)
        assert metadata.ndims == 2, "BEVLocalMapper requires 2d metadata"
        self.n_features = len(feature_keys) if n_features == -1 else n_features
        self.feature_keys = feature_keys[:self.n_features]

        self.bev_feature_grid = BEVGrid(self.metadata, self.feature_keys, device=device)

        self.elevation_grid = BEVGrid(self.metadata, ELEVATION_MAP_FEATURE_KEYS, device=device)

        #need nonzero placeholders for elev
        self.elev_data_placeholder = torch.tensor([1e10, -1e10, 0], device=self.device)
        self.elevation_grid.data += self.elev_data_placeholder.view(1,1,3)

        self.ema = ema
        self.overhang = overhang

    def get_bev_map(self):
        """
        Combine BEVGrids into one
        """
        mask_keys = FeatureKeyList(
            label=["num_points", "num_feature_points"],
            metainfo=["terrain_estimation"] * 2
        )
        all_fks = self.elevation_grid.feature_keys + self.bev_feature_grid.feature_keys + mask_keys
        bev_grid = BEVGrid(self.metadata, all_fks, device=self.device)

        n1 = len(self.elevation_grid.feature_keys)
        n2 = len(self.bev_feature_grid.feature_keys)

        bev_grid.data[..., :n1] = self.elevation_grid.data
        bev_grid.data[..., n1:n1+n2] = self.bev_feature_grid.data
        bev_grid.data[..., -2] = self.elevation_grid.hits
        bev_grid.data[..., -1] = self.bev_feature_grid.hits

        bev_grid.hits = self.elevation_grid.hits
        bev_grid.misses = self.elevation_grid.misses

        return bev_grid

    def update_pose(self, pose: torch.Tensor):
        """
        Args:
            pose: [N] Tensor (we will take the first two elements as the new pose)
        """
        new_origin = (
            torch.div(pose[:2] + self.base_metadata.origin, self.base_metadata.resolution, rounding_mode='floor')
        ) * self.base_metadata.resolution
        
        px_shift = torch.round(
            (new_origin - self.metadata.origin) / self.metadata.resolution
        ).long()

        self.bev_feature_grid.shift(px_shift)
        self.elevation_grid.shift(px_shift, placeholder=self.elev_data_placeholder)
        self.metadata.origin = new_origin

    def add_feature_pc(self, pos: torch.Tensor, feat_pc: FeaturePointCloudTorch):
        ## update elevation ##
        elevation_grid_new = self.elevation_grid_from_pc(feat_pc, self.metadata)
        min_elev_idx = ELEVATION_MAP_FEATURE_KEYS.index("min_elevation")
        max_elev_idx = ELEVATION_MAP_FEATURE_KEYS.index("max_elevation")

        self.elevation_grid.data[..., min_elev_idx] = torch.minimum(
            self.elevation_grid.data[..., min_elev_idx],
            elevation_grid_new.data[..., min_elev_idx]
        )

        self.elevation_grid.data[..., max_elev_idx] = torch.maximum(
            self.elevation_grid.data[..., max_elev_idx],
            elevation_grid_new.data[..., max_elev_idx]
        )

        #TODO variance

        self.elevation_grid.hits += elevation_grid_new.hits
        self.elevation_grid.misses += elevation_grid_new.misses

        feat_pc_filtered = self.filter_pc_with_terrain_map(feat_pc)

        ## merge features ##
        bev_grid_new = BEVGrid.from_feature_pc(feat_pc_filtered, self.metadata, self.n_features)

        to_add = bev_grid_new.known & ~self.bev_feature_grid.known
        to_merge = bev_grid_new.known & self.bev_feature_grid.known

        self.bev_feature_grid.hits += bev_grid_new.hits
        self.bev_feature_grid.misses += bev_grid_new.misses

        self.bev_feature_grid.data[to_add] = bev_grid_new.data[to_add]
        self.bev_feature_grid.data[to_merge] = (1.0 - self.ema) * self.bev_feature_grid.data[
            to_merge
        ] + self.ema * bev_grid_new.data[to_merge]

    def filter_pc_with_terrain_map(self, feat_pc):
        """
        Given a feature pc, return a fpc that only contains points below terrain+overhang
        """
        pt_zs = feat_pc.pts[:, 2]

        min_elev_idx = ELEVATION_MAP_FEATURE_KEYS.index("min_elevation")
        min_height = self.elevation_grid.data[..., min_elev_idx]

        grid_idxs, valid_mask = self.elevation_grid.get_grid_idxs(feat_pc.pts)
        grid_idxs[~valid_mask] = 0

        #note that cells w/o height vals default to large pos and won't be filtered
        pt_cmp_heights = min_height[grid_idxs[:, 0], grid_idxs[:, 1]] + self.overhang
        height_mask = pt_zs < pt_cmp_heights

        return feat_pc.apply_mask(height_mask)

    def elevation_grid_from_pc(self, pc, metadata):
        """
        Create an elevation grid (min/max elev + optional variance)
            from a pc
        """
        elev_grid = BEVGrid(metadata, ELEVATION_MAP_FEATURE_KEYS, self.device)

        grid_idxs, valid_mask = elev_grid.get_grid_idxs(pc.pts)
        raster_idxs = elev_grid.grid_indices_to_raster_indices(grid_idxs)
        zs = pc.pts[:, 2]

        min_height = torch.ones(metadata.N[0]*metadata.N[1], device=self.device) * 1e10
        torch_scatter.scatter(
            zs[valid_mask],
            raster_idxs[valid_mask],
            dim=0,
            out=min_height,
            reduce='min'
        )
        elev_grid.data[..., ELEVATION_MAP_FEATURE_KEYS.index("min_elevation")] = min_height.view(*metadata.N)

        max_height = torch.ones(metadata.N[0]*metadata.N[1], device=self.device) * -1e10
        torch_scatter.scatter(
            zs[valid_mask],
            raster_idxs[valid_mask],
            dim=0,
            out=max_height,
            reduce='max'
        )
        elev_grid.data[..., ELEVATION_MAP_FEATURE_KEYS.index("max_elevation")] = max_height.view(*metadata.N)

        hits = torch_scatter.scatter(
            torch.ones(valid_mask.sum(), dtype=torch.long, device=self.device),
            raster_idxs[valid_mask],
            dim=0,
            dim_size=metadata.N[0] * metadata.N[1],
            reduce="sum",
        )
        elev_grid.hits = hits.view(*metadata.N)

        return elev_grid

    def to(self, device):
        self.device = device
        self.bev_grid = self.bev_grid.to(device)
        self.metadata = self.metadata.to(device)
        return self

class BEVGrid:
    """
    Actual class that handles feature aggregation
    """
    def from_feature_pc(feat_pc, metadata, n_features=-1):
        """
        Instantiate a BEVGrid from a feature pc
        """
        n_features = len(feat_pc.feature_keys) if n_features == -1 else n_features
        feat_keys = feat_pc.feature_keys[:n_features]

        bevgrid = BEVGrid(metadata, feat_keys, feat_pc.device)

        #scatter pts with features
        grid_idxs, valid_mask = bevgrid.get_grid_idxs(feat_pc.feature_pts)
        raster_idxs = bevgrid.grid_indices_to_raster_indices(grid_idxs)
        raster_map = torch_scatter.scatter(
            feat_pc.features[valid_mask][:, :n_features],
            raster_idxs[valid_mask],
            dim=0,
            dim_size=metadata.N[0] * metadata.N[1],
            reduce="mean",
        )
        bevgrid.data = raster_map.view(*metadata.N, -1)

        #scatter hits
        hits = torch_scatter.scatter(
            torch.ones(valid_mask.sum(), dtype=torch.long, device=feat_pc.device),
            raster_idxs[valid_mask],
            dim=0,
            dim_size=metadata.N[0] * metadata.N[1],
            reduce="sum",
        )
        bevgrid.hits = hits.view(*metadata.N)

        return bevgrid

    def __init__(self, metadata, feature_keys, device='cpu'):
        self.metadata = metadata
        self.feature_keys = feature_keys
        self.n_features = len(self.feature_keys)

        self.data = torch.zeros(*metadata.N, self.n_features, device=device)
        self.hits = torch.zeros(*metadata.N, device=device, dtype=torch.long)
        self.misses = torch.zeros(*metadata.N, device=device, dtype=torch.long)
        self.device = device

    @property
    def known(self):
        return self.hits > 0

    def get_grid_idxs(self, pts):
        """
        Get indexes for positions given map metadata
        """
        gidxs = torch.div(pts[:, :2] - self.metadata.origin, self.metadata.resolution, rounding_mode='floor').long()
        mask = (gidxs >= 0).all(dim=-1) & (gidxs < self.metadata.N.view(1,2)).all(dim=-1)
        return gidxs, mask

    def shift(self, px_shift, placeholder=None):
        """
        Apply a pixel shift to the map

        Args:
            px_shift: Tensor of [dx, dy], where the ORIGIN of the bev map is moved by this many cells
                e.g. if px_shift is [-3, 5], the new origin is 3*res units left and 5*res units up
                        note that this means the data is shifted 3 cells right and 5 cells down
        """
        if placeholder == None:
            _ph = torch.zeros(self.n_features, device=self.device)
        else:
            _ph = placeholder

        dgx, dgy = px_shift
        self.data = torch.roll(self.data, shifts=[-dgx, -dgy], dims=[0, 1])
        self.hits = torch.roll(self.hits, shifts=[-dgx, -dgy], dims=[0, 1])
        self.misses = torch.roll(self.misses, shifts=[-dgx, -dgy], dims=[0, 1])

        if dgx > 0:
            self.data[-dgx:] = _ph
            self.hits[-dgx:] = False
            self.misses[-dgx:] = False
        elif dgx < 0:
            self.data[:-dgx] = _ph
            self.hits[:-dgx] = False
            self.misses[:-dgx] = False
        if dgy > 0:
            self.data[:, -dgy:] = _ph
            self.hits[:, -dgy:] = False
            self.misses[:, -dgy:] = False
        elif dgy < 0:
            self.data[:, :-dgy] = _ph
            self.hits[:, :-dgy] = False
            self.misses[:, :-dgy] = False

        # update metadata
        self.metadata.origin += px_shift * self.metadata.resolution[0]

    def pts_in_bounds(self, pts):
        """Check if points are in bounds

        Args:
            pts: [Nx3] Tensor of coordinates

        Returns:
            valid: [N] mask of whether point is within voxel grid
        """
        _min = self.metadata.origin.view(1, 2)
        _max = (self.metadata.origin + self.metadata.length).view(1, 2)

        low_check = (pts[..., :2] >= _min).all(axis=-1)
        high_check = (pts[..., :2] < _max).all(axis=-1)
        return low_check & high_check

    def grid_idxs_in_bounds(self, grid_idxs):
        """Check if grid idxs are in bounds

        Args:
            grid_idxs: [Nx2] Long Tensor of grid idxs

        Returns:
            valid: [N] mask of whether idx is within voxel grid
        """
        _min = torch.zeros_like(self.metadata.N).view(1, 2)
        _max = self.metadata.N.view(1, 2)

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
            grid_idxs: [Nx2] Long Tensor of grid indices

        Returns:
            raster_idxs: [N] Long Tensor of raster indices
        """
        return grid_idxs[:, 0] * self.metadata.N[1] + grid_idxs[:, 1]

    def raster_indices_to_grid_indices(self, raster_idxs):
        """Convert a set of raster indices to grid indices

        Args:
            raster_idxs: [N] Long Tensor of raster indices

        Returns:
            grid_idxs: [Nx2] Long Tensor of grid indices
        """
        xs = torch.div(raster_idxs, self.metadata.N[1], rounding_mode="floor").long()
        ys = raster_idxs % self.metadata.N[1]

        return torch.stack([xs, ys], axis=-1)

    def visualize(self, fig=None, axs=None):
        if fig is None or axs is None:
            fig, axs = plt.subplots(1, 2)

        extent = (
            self.metadata.origin[0].item(),
            self.metadata.origin[0].item() + self.metadata.length[0].item(),
            self.metadata.origin[1].item(),
            self.metadata.origin[1].item() + self.metadata.length[1].item(),
        )

        axs[0].imshow(
            normalize_dino(self.data[..., :3]).permute(1, 0, 2).cpu().numpy(),
            origin="lower",
            extent=extent,
        )
        axs[1].imshow(self.mask.T.cpu().numpy(), origin="lower", extent=extent)

        axs[0].set_title("features")
        axs[1].set_title("known")

        axs[0].set_xlabel("X(m)")
        axs[0].set_ylabel("Y(m)")

        axs[1].set_xlabel("X(m)")
        axs[1].set_ylabel("Y(m)")

        return fig, axs

    def to(self, device):
        self.device = device
        self.data = self.data.to(device)
        self.hits = self.hits.to(device)
        self.misses = self.misses.to(device)
        return self
