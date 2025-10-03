import os
import argparse

import torch
import open3d as o3d

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

        o3d.visualization.draw_geometries([base_vg_viz, base_bev_viz], window_name='base')
        o3d.visualization.draw_geometries([inpaint_vg_viz, inpaint_bev_viz], window_name='gt')
        o3d.visualization.draw_geometries([base_vg_viz, inpaint_bev_viz], window_name='supervision')