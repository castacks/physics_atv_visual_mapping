import os
import yaml
import tqdm
import argparse

import copy
import torch
import open3d as o3d

from ros_torch_converter.datatypes.pointcloud import FeaturePointCloudTorch
from ros_torch_converter.datatypes.voxel_grid import VoxelGridTorch

from tartandriver_utils.os_utils import kitti_n_frames

from physics_atv_visual_mapping.localmapping.metadata import LocalMapperMetadata
from physics_atv_visual_mapping.localmapping.voxel.voxel_localmapper import VoxelGrid, VoxelLocalMapper

"""
Script for computing psuedo-ground-truth voxel grids to compare against
Goal is that these voxel grids will contain all past/future voxels for a run
that intersect the current timestep's mapping volume
"""

def get_metadatas(voxel_dir):
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
        )
        metadatas.append(metadata)

    return metadatas

def compute_overlaps(metadatas):
    """
    Args:
        metadatas: [N] list of LocalMapperMetadatas defining the mapping volume
    Returns:
        overlaps: [N] list of indices for the last volume intersecting with the volume at that timestep
    """
    N = len(metadatas)
    maxidx = []

    for i in range(N):
        curr_metadata = metadatas[i]
        curr_maxidx = i
        for ii in range(i, N):
            check_metadata = metadatas[ii]

            if curr_metadata.intersects(check_metadata):
                curr_maxidx = ii
            else:
                break

        maxidx.append(curr_maxidx)

    return maxidx

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--run_dir', type=str, required=True, help='run dir to generate supervision for')
    parser.add_argument('--voxel_dir', type=str, required=False, default='voxel_map')
    parser.add_argument('--output_dir', type=str, required=False, default=None, help='dir to save to (default=<voxel_dir>_inpaint)')
    parser.add_argument('--device', type=str, required=False, default='cuda')
    args = parser.parse_args()

    N = kitti_n_frames(args.run_dir)

    voxel_dir = os.path.join(args.run_dir, args.voxel_dir)
    output_dir = os.path.join(args.run_dir, args.output_dir if args.output_dir else args.voxel_dir + '_inpaint')

    print(f'processing {voxel_dir} -> {output_dir} ({N} frames)')

    all_metadatas = get_metadatas(voxel_dir)
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

    localmapper = VoxelLocalMapper(
        metadata = agg_metadata,
        feature_keys = voxel_grid.feature_keys,
        raytracer = None,
        ema = 1.0,
        device = args.device
    )

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

        if ii % 100 == 0:
            o3d.visualization.draw_geometries([
                localmapper.voxel_grid.metadata.visualize(),
                localmapper.voxel_grid.visualize(),
                curr_voxel_grid.metadata.visualize()
            ])

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