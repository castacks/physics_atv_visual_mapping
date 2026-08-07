import os
import yaml
import tqdm
import argparse
from pathlib import Path

import copy
import torch
import open3d as o3d

from ros_torch_converter.datatypes.pointcloud import FeaturePointCloudTorch
from ros_torch_converter.datatypes.voxel_grid import VoxelGridTorch

from tartandriver_utils.os_utils import kitti_n_frames
from torch_coordinator.stream_control import wait_for_stream_ack

from physics_atv_visual_mapping.localmapping.metadata import LocalMapperMetadata
from physics_atv_visual_mapping.localmapping.hindsight import compute_overlaps
from physics_atv_visual_mapping.localmapping.voxel.voxel_localmapper import VoxelGrid, VoxelLocalMapper

"""
Script for computing psuedo-ground-truth voxel grids to compare against
Goal is that these voxel grids will contain all past/future voxels for a run
that intersect the current timestep's mapping volume
"""

def get_metadatas(voxel_dir, device):
    """
    faster to just load all the yamls than to load and discard the voxel data
    """
    metadata_fps = sorted([x for x in os.listdir(voxel_dir) if 'metadata.yaml' in x])

    metadatas = []
    for mfp in metadata_fps:
        mconfig = yaml.safe_load(open(os.path.join(voxel_dir, mfp), 'r'))
        metadata = LocalMapperMetadata(
            origin = mconfig['origin'],
            length = mconfig['length'],
            resolution = mconfig['resolution'],
        ).to(device)
        metadatas.append(metadata)

    return metadatas

def crop_voxel_grid(voxel_grid, metadata_new):
    """
    Crop a voxel grid to the metadata provided
    """
    assert torch.allclose(voxel_grid.metadata.resolution, metadata_new.resolution)
    
    grid_idxs = voxel_grid.raster_indices_to_grid_indices(voxel_grid.raster_indices)

    px_shift = torch.round(
            (metadata_new.origin - voxel_grid.metadata.origin) / voxel_grid.metadata.resolution
        ).long()

    grid_idxs_new = grid_idxs - px_shift.view(1,3)

    voxel_grid.metadata = metadata_new
    mask = voxel_grid.grid_idxs_in_bounds(grid_idxs_new)

    voxel_grid.raster_indices = voxel_grid.grid_indices_to_raster_indices(grid_idxs_new[mask])
    voxel_grid.features = voxel_grid.features[mask[voxel_grid.feature_mask]]
    voxel_grid.feature_mask = voxel_grid.feature_mask[mask]
    voxel_grid.hits = voxel_grid.hits[mask]
    voxel_grid.misses = voxel_grid.misses[mask]
    voxel_grid.min_coords = voxel_grid.min_coords[mask]
    voxel_grid.max_coords = voxel_grid.max_coords[mask]

    return voxel_grid

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--run_dir', type=str, required=True, help='run dir to generate supervision for')
    parser.add_argument('--voxel_dir', type=str, required=False, default='mapping/voxel_map')
    parser.add_argument('--output_dir', type=str, required=False, default='inpainting/voxel_map_inpaint', help='dir to save to (default=inpainting/voxel_map_inpaint)')
    parser.add_argument('--device', type=str, required=False, default='cuda')
    parser.add_argument('--stream-control-dir', type=Path, help='directory for batch ready/ack markers')
    parser.add_argument('--stream-batch-size', type=int, help='pause after this many output frames')
    args = parser.parse_args()

    if (args.stream_control_dir is None) != (args.stream_batch_size is None):
        parser.error('--stream-control-dir and --stream-batch-size must be used together')
    if args.stream_batch_size is not None and args.stream_batch_size < 1:
        parser.error('--stream-batch-size must be positive')
    if args.stream_control_dir is not None:
        args.stream_control_dir.mkdir(parents=True, exist_ok=True)

    N = kitti_n_frames(args.run_dir)

    voxel_dir = os.path.join(args.run_dir, args.voxel_dir)
    output_dir = os.path.join(args.run_dir, args.output_dir if args.output_dir else args.voxel_dir + '_inpaint')
    os.makedirs(output_dir, exist_ok=True)

    print(f'processing {voxel_dir} -> {output_dir} ({N} frames)')

    all_metadatas = get_metadatas(voxel_dir, args.device)
    overlaps = compute_overlaps(all_metadatas)

    ## now that we have the last frame, re-aggregate the voxels

    ## guaranteed to have all pts if we 3x the mapping volume
    agg_metadata = LocalMapperMetadata(
        origin = -0.5 * all_metadatas[0].length - all_metadatas[0].length,
        length = all_metadatas[0].length * 3,
        resolution = all_metadatas[0].resolution.clone()
    ).to(args.device)
    voxel_grid = VoxelGridTorch.from_kitti(voxel_dir, 0, args.device).voxel_grid
    agg_voxel_grid = VoxelGrid(metadata=agg_metadata, feature_keys=voxel_grid.feature_keys, device=args.device)

    ## by default we probably just want to copy latest if there's a feature
    localmapper = VoxelLocalMapper(
        metadata = agg_metadata,
        feature_keys = voxel_grid.feature_keys,
        raytracer = None,
        ema = 1.0,
        device = args.device
    )

    saved_frames = []

    for ii in range(N):
        curr_voxel_grid = VoxelGridTorch.from_kitti(voxel_dir, ii, args.device).voxel_grid
        pose = curr_voxel_grid.metadata.origin + curr_voxel_grid.metadata.length/2.

        fpc = FeaturePointCloudTorch.from_torch(
            pts = curr_voxel_grid.midpoints,
            features = curr_voxel_grid.features,
            mask = curr_voxel_grid.feature_mask,
            feature_keys = curr_voxel_grid.feature_keys
        )

        localmapper.update_pose(pose)
        localmapper.add_feature_pc(pose, fpc)

        ## TODO extract the aggregated voxel grid from the current metadata
        save_idxs = torch.argwhere(ii == overlaps).flatten()

        print(f'itr {ii}/{N}, saving {save_idxs.shape[0]} frames', end='\r')

        for si in save_idxs:
            saved_frames.append(si)

            query_metadata = all_metadatas[si]

            inpaint_vg = crop_voxel_grid(
                copy.deepcopy(localmapper.voxel_grid),
                query_metadata
            )

            vgt_out = VoxelGridTorch.from_voxel_grid(inpaint_vg)
            vgt_out.to_kitti(output_dir, si)

            if args.stream_control_dir is not None:
                wait_for_stream_ack(
                    args.stream_control_dir,
                    args.stream_batch_size,
                    N,
                    len(saved_frames),
                )

            base_vg = VoxelGridTorch.from_kitti(voxel_dir, si, device=localmapper.device).voxel_grid

            print(f'saved frame {si} ({base_vg.raster_indices.shape[0]}->{inpaint_vg.raster_indices.shape[0]} voxels)')

    ## verify
    out_idxs = torch.tensor(saved_frames)
    assert (out_idxs.sort()[0] == torch.arange(N)).all()

    # for i, (metadata, overlap) in enumerate(zip(all_metadatas, overlaps)):
    #     idxs = torch.arange(i, overlap)

    #     ## debug viz 
    #     if i % 100 == 0:
    #         viz = []
    #         for ii in idxs[::10]:
    #             overlap_metadata = all_metadatas[ii]
    #             overlap_viz = overlap_metadata.visualize()
    #             overlap_viz.paint_uniform_color([0., 0., 1.])
    #             viz.append(overlap_viz)

    #         voxels = VoxelGridTorch.from_kitti(voxel_dir, i).voxel_grid.visualize()
    #         viz.append(voxels)

    #         o3d.visualization.draw_geometries(viz)
