import os
import yaml
import tqdm
import argparse

import cv2
import torch
import numpy as np
import matplotlib.pyplot as plt

from tartandriver_utils.geometry_utils import TrajectoryInterpolator

from physics_atv_visual_mapping.image_processing.image_pipeline import (
    setup_image_pipeline,
)
from physics_atv_visual_mapping.pointcloud_colorization.torch_color_pcl_utils import *
from physics_atv_visual_mapping.utils import *

from physics_atv_visual_mapping.localmapping.bev.bev_localmapper import BEVLocalMapper
from physics_atv_visual_mapping.localmapping.voxel.voxel_localmapper import (
    VoxelLocalMapper,
)
from physics_atv_visual_mapping.localmapping.metadata import LocalMapperMetadata

from physics_atv_visual_mapping.terrain_estimation.terrain_estimation_pipeline import setup_terrain_estimation_pipeline

"""
Run rgb mapping offline on the kitti-formatted dataset
"""

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--run_dir", type=str, required=True, help="path to dataset")
    parser.add_argument("--config", type=str, required=True, help="path to config")
    parser.add_argument(
        "--odom", type=str, required=False, default="odom", help="name of odom folder"
    )
    parser.add_argument(
        "--pcl", type=str, required=False, default="pcl", help="name of pcl folder"
    )
    parser.add_argument(
        "--image",
        type=str,
        required=False,
        default="image_left_color",
        help="name of image folder",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        required=False,
        default="local_visual_map",
        help="name of result folder",
    )
    parser.add_argument(
        "--timestamp_check",
        action="store_true",
        help="set this flag to double-check timestamps on data",
    )
    args = parser.parse_args()

    config = yaml.safe_load(open(args.config, "r"))

    # setup io
    os.makedirs(os.path.join(args.run_dir, args.output_dir), exist_ok=True)

    intrinsics = (
        torch.tensor(config["intrinsics"]["K"]).reshape(3, 3).to(config["device"])
    )
    extrinsics = pose_to_htm(
        np.concatenate(
            [np.array(config["extrinsics"]["p"]), np.array(config["extrinsics"]["q"])],
            axis=-1,
        )
    ).to(config["device"])

    # image_pipeline = setup_image_pipeline(config)

    # setup localmapper
    localmapper_metadata = LocalMapperMetadata(**config["localmapping"]["metadata"])
    # localmapper = BEVLocalMapper(
    localmapper = VoxelLocalMapper(
        localmapper_metadata,
        n_features=config["localmapping"]["n_features"],
        ema=config["localmapping"]["ema"],
        device=config["device"],
    )

    # setup terrain estimation
    do_terrain_estimation = 'terrain_estimation' in config.keys()
    if do_terrain_estimation:
        terrain_estimator = setup_terrain_estimation_pipeline(config)

    ## first create the trajectory interpolator
    odom_dir = os.path.join(args.run_dir, args.odom)
    poses = np.loadtxt(os.path.join(odom_dir, "data.txt"))
    pose_ts = np.loadtxt(os.path.join(odom_dir, "timestamps.txt"))
    mask = np.ones_like(pose_ts).astype(bool)
    mask[1:] = np.abs(pose_ts[1:] - pose_ts[:-1]) > 1e-4
    poses = poses[mask]
    pose_ts = pose_ts[mask] 

    traj_interp = TrajectoryInterpolator(pose_ts, poses, tol=0.5)

    ## next compute gridmap sample times
    pcl_dir = os.path.join(args.run_dir, args.pcl)
    image_dir = os.path.join(args.run_dir, args.image)

    pcl_ts = np.loadtxt(os.path.join(pcl_dir, "timestamps.txt"))
    image_ts = np.loadtxt(os.path.join(image_dir, "timestamps.txt"))

    # timestamp check
    if args.timestamp_check:
        x = np.arange(len(pose_ts)) 
        pcl_ts = pcl_ts[mask]
        image_ts = image_ts[mask]
        plt.scatter(x, pose_ts, label="pose ({} samples)".format(pose_ts.shape[0]))
        plt.scatter(x, pcl_ts, label="pcl ({} samples)".format(pcl_ts.shape[0]))
        plt.scatter(x, image_ts, label="image ({} samples)".format(image_ts.shape[0]))
        plt.legend()
        plt.title("timestamp check (all colors should overlap)")
        plt.show()

    # sync image times to pcl times (note that I'm choosing to break causality for accuracy)
    image_to_pcl_tdiffs = np.abs(image_ts.reshape(1, -1) - pcl_ts.reshape(-1, 1))
    pcl_idxs = np.argmin(image_to_pcl_tdiffs, axis=0)
    pcl_errs = np.min(image_to_pcl_tdiffs, axis=0)

    print(
        "pcl->image timesync errs: mean: {:.4f}s, max: {:.4f}s".format(
            pcl_errs.mean(), pcl_errs.max()
        )
    )

    for ii in tqdm.tqdm(range(len(image_ts))):
        pcl_fp = os.path.join(args.run_dir, args.pcl, "{:08d}.npy".format(pcl_idxs[ii]))
        image_fp = os.path.join(args.run_dir, args.image, "{:08d}.png".format(ii))

        pcl = torch.from_numpy(np.load(pcl_fp)).float().to(config["device"])
        img = cv2.imread(image_fp)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB) / 255.0
        img = torch.tensor(img).to(config["device"])#[H,W,C]

        I = get_intrinsics(intrinsics).to(config["device"])
        E = get_extrinsics(extrinsics).to(config["device"])

        P = obtain_projection_matrix(I, E)
        pixel_coordinates = get_pixel_from_3D_source(pcl[:, :3], P)
        (
            lidar_points_in_frame,
            pixels_in_frame,
            ind_in_frame,
        ) = get_points_and_pixels_in_frame(
            pcl[:, :3], pixel_coordinates, img.shape[0], img.shape[1]
        )

        rgb_colors = img[pixels_in_frame[:, 1], pixels_in_frame[:, 0]]
        rgb_colors = rgb_colors.to(config["device"]).float()
        rgb_pcl = torch.cat([pcl[ind_in_frame][:, :3], rgb_colors], dim=-1)

        pose = traj_interp(image_ts[ii])
        pose = pose_to_htm(pose).to(config["device"])
        rgb_pcl = transform_points(rgb_pcl, pose)

        no_feature_pcl = transform_points(pcl, pose)
        localmapper.update_pose(pose[:3, -1])
        localmapper.add_feature_pc(pts=rgb_pcl[:, :3], features=rgb_pcl[:, 3:])

        #also add non-colorized points
        localmapper.add_pc(pts=no_feature_pcl[:, :3])

        if do_terrain_estimation:
            bev_features = terrain_estimator.run(localmapper.voxel_grid)
            torch.cuda.synchronize()
   
        if (ii+1) % 100 == 0:
            # localmapper.bev_grid.visualize();plt.show()
            localmapper.voxel_grid.visualize()

            if do_terrain_estimation:
                fig, axs = plt.subplots(4, 4, figsize=(16, 16))
                axs = axs.flatten()
                for i in range(min(len(axs), len(bev_features.feature_keys))):
                    k = bev_features.feature_keys[i]
                    data = bev_features.data[..., i]
                    axs[i].imshow(data.T.cpu().numpy(), origin='lower', cmap='jet', interpolation='none')
                    axs[i].set_title(k)

                fig.suptitle('showing 16 of {} features'.format(len(bev_features.feature_keys)))
                plt.show()
