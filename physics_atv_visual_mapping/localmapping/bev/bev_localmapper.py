import torch
import torch_scatter
import matplotlib.pyplot as plt

from ros_torch_converter.datatypes.pointcloud import FeaturePointCloudTorch

from physics_atv_visual_mapping.localmapping.base import LocalMapper
from physics_atv_visual_mapping.localmapping.metadata import LocalMapperMetadata
from physics_atv_visual_mapping.feature_key_list import FeatureKeyList
from physics_atv_visual_mapping.utils import *

#similar to Fankhauser et al 2018
ELEVATION_MAP_FEATURE_KEYS = FeatureKeyList(
    label=["min_elevation", "max_elevation", "elevation_var"],
    metainfo=["terrain_estimation"] * 3
)

class BEVLocalMapper(LocalMapper):
    """Class for local mapping in BEV"""

    def __init__(self, metadata, feature_keys, n_features, ema, overhang, do_overhang_cleanup, device):
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
        self.do_overhang_cleanup = do_overhang_cleanup

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
        feat_pc_filtered = self.filter_pc_with_terrain_map(feat_pc)

        elevation_grid_new = self.elevation_grid_from_pc(feat_pc_filtered, self.metadata)
        min_elev_idx = ELEVATION_MAP_FEATURE_KEYS.index("min_elevation")
        max_elev_idx = ELEVATION_MAP_FEATURE_KEYS.index("max_elevation")

        self.elevation_grid.data[..., min_elev_idx] = torch.minimum(
            self.elevation_grid.data[..., min_elev_idx],
            elevation_grid_new.data[..., min_elev_idx]
        )

        #do an extra check to remove extraneous overhangs
        #if prev diff > overhang and curr_diff < overhang, just overwrite curr max
        prev_diff = self.elevation_grid.data[..., max_elev_idx] - self.elevation_grid.data[..., min_elev_idx]
        curr_diff = elevation_grid_new.data[..., max_elev_idx] - elevation_grid_new.data[..., min_elev_idx]
        mask = (prev_diff > self.overhang) & (curr_diff < self.overhang)

        max_elev1 = elevation_grid_new.data[..., max_elev_idx]

        self.elevation_grid.data[..., max_elev_idx] = torch.maximum(
            self.elevation_grid.data[..., max_elev_idx],
            elevation_grid_new.data[..., max_elev_idx]
        )

        if self.do_overhang_cleanup:
            self.elevation_grid.data[..., max_elev_idx][mask] = max_elev1[mask]

        #TODO variance

        self.elevation_grid.hits += elevation_grid_new.hits
        self.elevation_grid.misses += elevation_grid_new.misses

        ## merge features ##
        bev_grid_new = self.bev_grid_from_fpc(feat_pc_filtered)

        self.bev_feature_grid = self.merge_bev_grids(self.bev_feature_grid, bev_grid_new)

    def bev_grid_from_fpc(self, fpc):
        return BEVGrid.from_feature_pc(fpc, self.metadata, self.n_features)

    def merge_bev_grids(self, bev_grid1, bev_grid2):
        to_add = bev_grid2.known & ~bev_grid1.known
        to_merge = bev_grid2.known & bev_grid1.known

        bev_grid1.hits += bev_grid2.hits
        bev_grid1.misses += bev_grid2.misses

        bev_grid1.data[to_add] = bev_grid2.data[to_add]
        bev_grid1.data[to_merge] = (1.0 - self.ema) * bev_grid1.data[
            to_merge
        ] + self.ema * bev_grid2.data[to_merge]

        return bev_grid1

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

class BEVCoordinateLocalMapper(BEVLocalMapper):
    """
    Voxel localmapper class that maps image coordinates instead of features directly
    This requires us to implement a slightly different merge rule (list update instead of EMA)
    """
    def __init__(self, metadata, feature_keys, n_features, ema, overhang, do_overhang_cleanup, max_n_coords=10, device='cpu'):
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

        super().__init__(metadata, self.output_feature_keys, n_features, ema, overhang, do_overhang_cleanup, device)
        self.assert_feat_match = False
        self.max_n_coords = max_n_coords

    def bev_grid_from_fpc(self, fpc):
        """
        have to handle this one differently as an instantaneous BEV cell
            can draw from many pixels. Also note that unfortunately, the
            coordinate mapping trick is far less effective for BEV
        """
        bevgrid = BEVGrid(self.metadata, self.output_feature_keys, self.device)

        grid_idxs, valid_mask = bevgrid.get_grid_idxs(fpc.feature_pts)
        raster_idxs = bevgrid.grid_indices_to_raster_indices(grid_idxs)
        P = valid_mask.sum()
        ncams = fpc.features.shape[-1] // 5

        valid_raster_idxs = raster_idxs[valid_mask]
        #[P x ncam x 5]
        valid_coords = fpc.features[valid_mask].reshape(P,5,ncams).permute(0,2,1)

        valid_raster_idxs, sort_idxs = valid_raster_idxs.sort()
        valid_coords = valid_coords[sort_idxs]

        uniq_raster_idxs, inv_idxs, cnts = torch.unique(valid_raster_idxs, return_inverse=True, return_counts=True, sorted=True)
        n_cells = uniq_raster_idxs.shape[0]
        max_pts = cnts.max()

        """
        Ok I think the final algo looks somethign like this
            1. create an index tensor of shape [ncells x pmax]
            2. populate each with arange and clamp to count
            3. if inputs are sorted, we can calculate the offset to index in
            3. then we mask
        """
        #this only works if the input raster idxs are sorted
        offsets = torch.cat([torch.zeros(1, dtype=torch.long, device=self.device), cnts[:-1].cumsum(dim=0)])
        buf_idxs = torch.arange(max_pts, device=self.device).view(1,-1).tile(n_cells, 1)
        mask = (buf_idxs < cnts.view(-1, 1))
        buf_idxs += offsets.view(-1, 1)
        buf_idxs[~mask] = -1
        
        coord_data = valid_coords[buf_idxs]
        coord_data[~mask] = 0.

        coord_data = coord_data.reshape(n_cells, max_pts*ncams, 5)

        weights = coord_data[:, :, -1]
        #add jitter to encourage random sampling of pts in a cell
        sort_idxs = torch.argsort(weights + 1e-4*torch.rand_like(weights), dim=-1, descending=True)
        sort_idxs = sort_idxs[:, :self.max_n_coords]
        ibs = torch.arange(n_cells, device=self.device).view(-1, 1).tile(1, self.max_n_coords)
        coord_data = coord_data[ibs, sort_idxs]

        coord_data[:, :, -1] /= coord_data[:, :, -1].sum(dim=-1, keepdims=True)

        coord_data = coord_data.permute(0,2,1).reshape(n_cells, -1)

        grid_idxs = bevgrid.raster_indices_to_grid_indices(uniq_raster_idxs)
        bevgrid.data[grid_idxs[:, 0], grid_idxs[:, 1]] = coord_data
        bevgrid.hits[grid_idxs[:, 0], grid_idxs[:, 1]] = cnts

        return bevgrid
        
    def merge_bev_grids(self, bev_grid1, bev_grid2):
        nx, ny = bev_grid1.metadata.N

        bev_grid1.hits += bev_grid2.hits
        bev_grid1.misses += bev_grid2.misses

        bg1_coord_data = bev_grid1.data.view(nx, ny, 5, self.max_n_coords).permute(0,1,3,2)
        bg2_coord_data = bev_grid2.data.view(nx, ny, 5, self.max_n_coords).permute(0,1,3,2)

        bg1_coord_data[:, :, :, -1] *= (1.0 - self.ema)
        bg2_coord_data[:, :, :, -1] *= self.ema

        all_coord_data = torch.cat([bg1_coord_data, bg2_coord_data], dim=2)
        sort_idxs = all_coord_data[:, :, :, -1].argsort(dim=2, descending=True)[:, :, :self.max_n_coords]
        ixs, iys = torch.meshgrid(torch.arange(nx, device=self.device), torch.arange(ny, device=self.device), indexing='ij')
        ixs = ixs.unsqueeze(-1).tile(1,1,sort_idxs.shape[-1])
        iys = iys.unsqueeze(-1).tile(1,1,sort_idxs.shape[-1])
        all_coord_data = all_coord_data[ixs, iys, sort_idxs]
        weight_sum = all_coord_data[:, :, :, -1].sum(dim=-1, keepdims=True).clamp(1e-8, 1.)
        all_coord_data[:, :, :, -1] /= weight_sum

        bev_grid1.data = all_coord_data.permute(0,1,3,2).reshape(nx, ny, 5*self.max_n_coords)

        return bev_grid1

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
        self.metadata = metadata[:2]
        self.feature_keys = feature_keys
        self.n_features = len(self.feature_keys)

        self.data = torch.zeros(*self.metadata.N, self.n_features, device=device)
        self.hits = torch.zeros(*self.metadata.N, device=device, dtype=torch.long)
        self.misses = torch.zeros(*self.metadata.N, device=device, dtype=torch.long)
        self.device = device

    def random_init(device='cpu'):
        metadata = LocalMapperMetadata.random_init(ndim=2, device=device)

        fks = FeatureKeyList(
            label=[f'rand_{i}' for i in range(5)],
            metainfo=['feat'] * 5
        )

        bevgrid = BEVGrid(metadata, fks)
        bevgrid.data = torch.randn_like(bevgrid.data)
        bevgrid.hits = (10. * torch.rand(*metadata.N, device=device)).long()
        bevgrid.misses = (10. * torch.rand(*metadata.N, device=device)).long()

        return bevgrid

    def __eq__(self, other):
        if self.feature_keys != other.feature_keys:
            return False

        if self.metadata != other.metadata:
            return False

        if not torch.allclose(self.data, other.data):
            return False

        #TODO we need to save/load these from kitti
        # if not (self.hits == other.hits).all():
        #     return False

        # if not (self.misses == other.misses).all():
        #     return False

        return True

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
        self.metadata = self.metadata.to(device)
        self.data = self.data.to(device)
        self.hits = self.hits.to(device)
        self.misses = self.misses.to(device)
        return self
