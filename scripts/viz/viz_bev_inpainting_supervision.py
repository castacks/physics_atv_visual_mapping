import os
import argparse

import torch
import open3d as o3d
import matplotlib.pyplot as plt

from tartandriver_utils.os_utils import kitti_n_frames
from tartandriver_utils.o3d_viz_utils import make_bev_mesh, normalize_dino

from ros_torch_converter.datatypes.bev_grid import BEVGridTorch
from ros_torch_converter.datatypes.voxel_grid import VoxelGridTorch

def make_bev_viz(bev_grid):
    height_idx = bev_grid.feature_keys.index('terrain')
    height = bev_grid.data[:, :, height_idx]

    mask_idx = bev_grid.feature_keys.index('min_elevation_filtered_inflated_mask')
    mask = bev_grid.data[:, :, mask_idx]

    vfm_idxs = [i for i,k in enumerate(bev_grid.feature_keys.metainfo) if k == 'vfm']
    colors = normalize_dino(bev_grid.data[:, :, vfm_idxs])

    mesh = make_bev_mesh(
        metadata=bev_grid.metadata,
        height=height,
        mask=mask,
        colors=colors
    )

    return mesh

def make_bev_cmp_viz(base_bev, inpaint_bev):
    base_nvox_idx = base_bev.feature_keys.index('num_voxels')
    inpaint_nvox_idx = inpaint_bev.feature_keys.index('num_voxels')

    base_terrain_idx = base_bev.feature_keys.index('terrain')
    inpaint_terrain_idx = inpaint_bev.feature_keys.index('terrain')

    base_vfm_idxs = [i for i,k in enumerate(base_bev.feature_keys.metainfo) if k == 'vfm']
    inpaint_vfm_idxs = [i for i,k in enumerate(inpaint_bev.feature_keys.metainfo) if k == 'vfm']

    extent = (
        base_bev.metadata.origin[0].item(),
        base_bev.metadata.origin[0].item() + base_bev.metadata.length[0].item(),
        base_bev.metadata.origin[1].item(),
        base_bev.metadata.origin[1].item() + base_bev.metadata.length[1].item()
    )

    base_voxels = base_bev.data[..., base_nvox_idx]
    inpaint_voxels = inpaint_bev.data[..., inpaint_nvox_idx]

    base_terrain = base_bev.data[..., base_terrain_idx]
    inpaint_terrain = inpaint_bev.data[..., inpaint_terrain_idx]

    base_vfm = base_bev.data[..., base_vfm_idxs]
    inpaint_vfm = inpaint_bev.data[..., inpaint_vfm_idxs]

    fig, axs = plt.subplots(3, 3, figsize=(18, 12))

    axs[0, 0].set_title('base voxel count')
    axs[1, 0].set_title('inpaint voxel count')
    axs[2, 0].set_title('voxel diff')

    axs[0, 1].set_title('base terrain')
    axs[1, 1].set_title('inpaint terrain')
    axs[2, 1].set_title('terrain diff')

    axs[0, 2].set_title('base features')
    axs[1, 2].set_title('inpaint features')
    axs[2, 2].set_title('feature diff')

    for ax in axs.flatten():
        ax.set_xlabel('X(m)')
        ax.set_ylabel('Y(m)')

    voxel_vmax = inpaint_voxels.max()
    voxel_diff = (inpaint_voxels - base_voxels).abs()

    terrain_vmin = inpaint_terrain.min()
    terrain_vmax = inpaint_terrain.max()
    terrain_diff = (inpaint_terrain - base_terrain).abs()

    vfm_diff = torch.linalg.norm(inpaint_vfm - base_vfm, dim=-1)

    axs[0, 0].imshow(base_voxels.T.cpu().numpy(), vmin=0., vmax=voxel_vmax, cmap='jet', extent=extent)
    axs[1, 0].imshow(inpaint_voxels.T.cpu().numpy(), vmin=0., vmax=voxel_vmax, cmap='jet', extent=extent)
    axs[2, 0].imshow(voxel_diff.T.cpu().numpy(), cmap='jet', extent=extent)

    axs[0, 1].imshow(base_terrain.T.cpu().numpy(), vmin=terrain_vmin, vmax=terrain_vmax, cmap='jet', extent=extent)
    axs[1, 1].imshow(inpaint_terrain.T.cpu().numpy(), vmin=terrain_vmin, vmax=terrain_vmax, cmap='jet', extent=extent)
    axs[2, 1].imshow(terrain_diff.T.cpu().numpy(), cmap='jet', extent=extent)

    axs[0, 2].imshow(normalize_dino(base_vfm).permute(1,0,2).cpu().numpy(), extent=extent)
    axs[1, 2].imshow(normalize_dino(inpaint_vfm).permute(1,0,2).cpu().numpy(), extent=extent)
    axs[2, 2].imshow(vfm_diff.T.cpu().numpy(), cmap='jet', extent=extent)

    return fig, axs

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--run_dir', type=str, required=True)
    parser.add_argument('--n', type=int, required=False, default=10)
    args = parser.parse_args()

    idxs = torch.randperm(kitti_n_frames(args.run_dir))[:args.n]

    voxel_dir = os.path.join(args.run_dir, 'voxel_map')
    inpaint_voxel_dir = os.path.join(args.run_dir, 'voxel_map_inpaint')
    bev_dir = os.path.join(args.run_dir, 'bev_map_reduce')
    inpaint_bev_dir = os.path.join(args.run_dir, 'bev_map_inpaint_reduce')

    for i in idxs:
        base_vg = VoxelGridTorch.from_kitti(voxel_dir, i).voxel_grid
        base_vg_viz = base_vg.visualize()

        inpaint_vg = VoxelGridTorch.from_kitti(inpaint_voxel_dir, i).voxel_grid
        inpaint_vg_viz = inpaint_vg.visualize()

        base_bev = BEVGridTorch.from_kitti(bev_dir, i).bev_grid
        base_bev_viz = make_bev_viz(base_bev)

        inpaint_bev = BEVGridTorch.from_kitti(inpaint_bev_dir, i).bev_grid
        inpaint_bev_viz = make_bev_viz(inpaint_bev)

        ## check overlap stats
        all_idxs = torch.cat([base_vg.raster_indices, inpaint_vg.raster_indices])
        base_n = base_vg.raster_indices.shape[0]
        _uniq, _inv, _cnt = torch.unique(all_idxs, return_inverse=True, return_counts=True)

        base_cnts = _cnt[_inv[:base_n]]
        inpaint_cnts = _cnt[_inv[base_n:]]

        print(f"|base - inpaint| = {(base_cnts==1).sum()} (should be small)")
        print(f"|inpaint - base| = {(inpaint_cnts==1).sum()} (should be big)")

        ## 2d viz
        fig, axs = make_bev_cmp_viz(base_bev, inpaint_bev)
        plt.show()

        o3d.visualization.draw_geometries([base_vg_viz, base_bev_viz], window_name='base')
        o3d.visualization.draw_geometries([inpaint_vg_viz, inpaint_bev_viz], window_name='gt')
        o3d.visualization.draw_geometries([base_vg_viz, inpaint_bev_viz], window_name='supervision')