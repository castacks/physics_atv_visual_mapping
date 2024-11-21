import torch
import torch_scatter
import matplotlib.pyplot as plt

from physics_atv_visual_mapping.localmapping.base import LocalMapper
from physics_atv_visual_mapping.utils import *


class BEVLocalMapper(LocalMapper):
    """Class for local mapping in BEV"""

    def __init__(self, metadata, n_features, ema, device, feature_key='feature', feature_keys=None):
        super().__init__(metadata, device)
        assert metadata.ndims == 2, "BEVLocalMapper requires 2d metadata"
        if feature_keys is None:
            self.feature_keys = ['{}_{}'.format(feature_key, i) for i in range(n_features)]
        else:
            self.feature_keys = feature_keys

        self.bev_grid = BEVGrid(self.metadata, n_features, self.feature_keys, device)
        self.n_features = n_features
        self.ema = ema

    def update_pose(self, pose: torch.Tensor):
        """
        Args:
            pose: [N] Tensor (we will take the first two elements as the new pose)
        """
        new_origin = (
            torch.div(pose[:2] + self.base_metadata.origin, self.base_metadata.resolution, rounding_mode='floor')
        ) * self.base_metadata.resolution
        self.bev_grid.metadata = self.metadata

        px_shift = torch.round(
            (new_origin - self.metadata.origin) / self.metadata.resolution
        ).long()
        self.bev_grid.shift(px_shift)
        self.metadata.origin = new_origin

    def add_feature_pc(self, pts: torch.Tensor, features: torch.Tensor):
        bev_grid_new = BEVGrid.from_feature_pc(pts, features, self.feature_keys, self.metadata)

        to_add = bev_grid_new.known & ~self.bev_grid.known
        to_merge = bev_grid_new.known & self.bev_grid.known

        self.bev_grid.known = self.bev_grid.known | bev_grid_new.known

        self.bev_grid.data[to_add] = bev_grid_new.data[to_add]
        self.bev_grid.data[to_merge] = (1.0 - self.ema) * self.bev_grid.data[
            to_merge
        ] + self.ema * bev_grid_new.data[to_merge]

    def to(self, device):
        self.device = device
        self.bev_grid = self.bev_grid.to(device)
        self.metadata = self.metadata.to(device)
        return self


class BEVGrid:
    """
    Actual class that handles feature aggregation
    """

    def from_feature_pc(pts, features, feature_keys, metadata):
        """
        Instantiate a BEVGrid from a feauture pc
        """
        bevgrid = BEVGrid(metadata, features.shape[-1], feature_keys, features.device)
        grid_idxs, valid_mask = bevgrid.get_grid_idxs(pts)
        raster_idxs = grid_idxs[:, 0] * metadata.N[1] + grid_idxs[:, 1]
        res_map = torch.zeros(*metadata.N, features.shape[-1], device=features.device)
        known_map = torch.zeros(*metadata.N, device=features.device)
        raster_map = res_map.view(-1, features.shape[-1])
        raster_known_map = known_map.view(-1)

        torch_scatter.scatter(
            features[valid_mask],
            raster_idxs[valid_mask],
            dim=0,
            out=raster_map,
            reduce="mean",
        )

        torch_scatter.scatter(
            torch.ones(valid_mask.sum(), device=features.device),
            raster_idxs[valid_mask],
            dim=0,
            out=raster_known_map,
            reduce="mean",
        )

        bevgrid.data = res_map
        bevgrid.known = known_map > 1e-8 #cant scatter bool so convert here

        return bevgrid

    def __init__(self, metadata, n_features, feature_keys, device='cpu'):
        self.metadata = metadata
        self.feature_keys = feature_keys
        self.data = torch.zeros(*metadata.N, n_features, device=device)
        self.known = torch.zeros(*metadata.N, device=device, dtype=torch.bool)
        self.device = device

    def get_grid_idxs(self, pts):
        """
        Get indexes for positions given map metadata
        """
        gidxs = torch.div(pts[:, :2] - self.metadata.origin, self.metadata.resolution, rounding_mode='floor').long()
        mask = (gidxs >= 0).all(dim=-1) & (gidxs < self.metadata.N.view(1,2)).all(dim=-1)
        return gidxs, mask

    def shift(self, px_shift):
        """
        Apply a pixel shift to the map

        Args:
            px_shift: Tensor of [dx, dy], where the ORIGIN of the bev map is moved by this many cells
                e.g. if px_shift is [-3, 5], the new origin is 3*res units left and 5*res units up
                        note that this means the data is shifted 3 cells right and 5 cells down
        """
        dgx, dgy = px_shift
        self.data = torch.roll(self.data, shifts=[-dgx, -dgy], dims=[0, 1])
        self.known = torch.roll(self.known, shifts=[-dgx, -dgy], dims=[0, 1])

        if dgx > 0:
            self.data[-dgx:] = 0.0
            self.known[-dgx:] = False
        elif dgx < 0:
            self.data[:-dgx] = 0.0
            self.known[:-dgx] = False
        if dgy > 0:
            self.data[:, -dgy:] = 0.0
            self.known[:, -dgy:] = False
        elif dgy < 0:
            self.data[:, :-dgy] = 0.0
            self.known[:, :-dgy] = False

        # update metadata
        self.metadata.origin += px_shift * self.metadata.resolution[0]

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
        axs[1].imshow(self.known.T.cpu().numpy(), origin="lower", extent=extent)

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
        self.known = self.known.to(device)
        return self
