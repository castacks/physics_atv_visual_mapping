import os
import cv2
import tqdm
import yaml
import torch
import numpy as np
import argparse
import matplotlib.pyplot as plt


from tartandriver_utils.geometry_utils import TrajectoryInterpolator

from physics_atv_visual_mapping.image_processing.image_pipeline import (
    setup_image_pipeline,
)
from physics_atv_visual_mapping.pointcloud_colorization.torch_color_pcl_utils import *
from physics_atv_visual_mapping.utils import pose_to_htm, transform_points

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, required=True, help='path to local mapping config')
    parser.add_argument(
        "--data_dir",
        type=str,
        required=True,
        help="path to KITTI-formatted dataset to process",
    )
    parser.add_argument(
        "--save_to", type=str, required=True, help="path to save PCA to"
    )
    parser.add_argument('--pc_in_local', action='store_true', help='set this flag if the pc is in the sensor frame, otherwise assume in odom frame')
    parser.add_argument('--pc_lim', type=float, nargs=2, required=False, default=[5., 100.], help='limit on range (m) of pts to consider')
    parser.add_argument('--pca_nfeats', type=int, required=False, default=64, help='number of pca feats to use')
    args = parser.parse_args()

    config = yaml.safe_load(open(args.config, "r"))
    print(config)

    ##get extrinsics and intrinsics
    lidar_to_cam = np.concatenate(
        [
            np.array(config["extrinsics"]["p"]),
            np.array(config["extrinsics"]["q"]),
        ],
        axis=-1,
    )
    extrinsics = pose_to_htm(lidar_to_cam)

    intrinsics = get_intrinsics(torch.tensor(config["intrinsics"]["K"]).reshape(3, 3))
    # dont combine because we need to recalculate given dino

    pipeline = setup_image_pipeline(config)

    dino_buf = []

    # check to see if single run or dir of runs
    run_dirs = []
    if config['odometry']['folder'] in os.listdir(args.data_dir):
        run_dirs = [args.data_dir]
    else:
        run_dirs = [os.path.join(args.data_dir, x) for x in os.listdir(args.data_dir)]

    print("computing for {} run dirs".format(len(run_dirs)))

    ## viz loop ##
    for ddir in run_dirs:
        odom_dir = os.path.join(ddir, config["odometry"]["folder"])
        poses = np.loadtxt(os.path.join(odom_dir, "data.txt"))
        pose_ts = np.loadtxt(os.path.join(odom_dir, "timestamps.txt"))
        mask = np.abs(pose_ts[1:] - pose_ts[:-1]) > 1e-4
        poses = poses[1:][mask]
        pose_ts = pose_ts[1:][mask]

        traj_interp = TrajectoryInterpolator(pose_ts, poses)

        img_dir = os.path.join(ddir, config["image"]["folder"])
        img_ts = np.loadtxt(os.path.join(img_dir, "timestamps.txt"))

        pcl_dir = os.path.join(ddir, config["pointcloud"]["folder"])
        pcl_ts = np.loadtxt(os.path.join(pcl_dir, "timestamps.txt"))

        pcl_img_dists = np.abs(img_ts.reshape(1, -1) - pcl_ts.reshape(-1, 1))
        pcl_img_mindists = np.min(pcl_img_dists, axis=-1)
        pcl_img_argmin = np.argmin(pcl_img_dists, axis=-1)

        pcl_valid_mask = (
            (pcl_ts > pose_ts[0]) & (pcl_ts < pose_ts[-1]) & (pcl_img_mindists < 0.1)
        )
        pcl_valid_idxs = np.argwhere(pcl_valid_mask).flatten()

        print("found {} valid pcl-image pairs".format(pcl_valid_mask.sum()))

        for pcl_idx in pcl_valid_idxs[::100]:
            pcl_fp = os.path.join(pcl_dir, "{:08d}.npy".format(pcl_idx))
            pcl = torch.from_numpy(np.load(pcl_fp)).to(config["device"]).float()
            pcl_t = pcl_ts[pcl_idx]

            if not args.pc_in_local:
                pose = traj_interp(pcl_t)
                H = pose_to_htm(pose).to(config["device"]).float()
                pcl = transform_points(pcl, torch.linalg.inv(H))

            pcl_dists = torch.linalg.norm(pcl[:, :3], dim=-1)
            pcl_mask = (pcl_dists > args.pc_lim[0]) & (
                pcl_dists < args.pc_lim[1]
            )
            pcl = pcl[pcl_mask]
            pcl_dists = pcl_dists[pcl_mask]

            pcl = pcl[:, :3]  # assume first three are [x,y,z]

            img_idx = pcl_img_argmin[pcl_idx]
            img_fp = os.path.join(img_dir, "{:08d}.png".format(img_idx))
            img = cv2.imread(img_fp)
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB) / 255.0

            P = obtain_projection_matrix(intrinsics, extrinsics).to(
                config["device"]
            )
            pcl_pixel_coords = get_pixel_from_3D_source(pcl, P)
            (
                pcl_in_frame,
                pixels_in_frame,
                ind_in_frame,
            ) = get_points_and_pixels_in_frame(
                pcl, pcl_pixel_coords, img.shape[0], img.shape[1]
            )

            pcl_px_in_frame = pcl_pixel_coords[ind_in_frame]
            pcl_dists = pcl_dists[ind_in_frame]
            pc_z = (pcl_dists / pcl_dists.max()).cpu().numpy()

            fig, axs = plt.subplots(1, 2, figsize=(40, 24))
            axs = axs.flatten()

            axs[0].imshow(img)
            #            axs[0].imshow(dino_viz.cpu(), alpha=0.5, extent=extent)

            axs[1].imshow(img)
            axs[1].scatter(pcl_px_in_frame[:, 0].cpu(), pcl_px_in_frame[:, 1].cpu(), c=pc_z, s=1., alpha=0.5, cmap='jet')

            plt.show()
