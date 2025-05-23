import os
import cv2
import tqdm
import yaml
import torch
import numpy as np
import argparse
import matplotlib.pyplot as plt
from termcolor import colored

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
    parser.add_argument('--n_frames', type=int, required=False, default=3000, help='process this many frames for the pca')
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

    #if config already has a pca, remove it.
    image_processing_config = []
    for ip_block in config['image_processing']:
        if ip_block['type'] == 'pca':
            print(colored('WARNING: found an existing PCA block in the image processing config. removing...', 'yellow'))
            break
        image_processing_config.append(ip_block)

    config['image_processing'] = image_processing_config

    pipeline = setup_image_pipeline(config)

    dino_buf = []

    # check to see if single run or dir of runs
    run_dirs = []
    if config['odometry']['folder'] in os.listdir(args.data_dir):
        run_dirs = [args.data_dir]
    else:
        run_dirs = [os.path.join(args.data_dir, x) for x in os.listdir(args.data_dir)]

    N_samples = 0
    for ddir in run_dirs:
        odom_dir = os.path.join(ddir, config["odometry"]["folder"])
        poses = np.loadtxt(os.path.join(odom_dir, "data.txt"))
        N_samples += poses.shape[0]

    n_frames = min(args.n_frames, N_samples)
    proc_every = int(N_samples / n_frames)

    print("computing for {} run dirs ({} samples, proc every {}th frame.)".format(len(run_dirs), N_samples, proc_every))

    for ddir in tqdm.tqdm(run_dirs):
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

        for pcl_idx in tqdm.tqdm(pcl_valid_idxs):
            if pcl_idx % proc_every == 0:
                pcl_fp = os.path.join(pcl_dir, "{:08d}.npy".format(pcl_idx))
                pcl_t = pcl_ts[pcl_idx]
                pcl = torch.from_numpy(np.load(pcl_fp)).to(config["device"]).float()

                if not args.pc_in_local:
                    pose = traj_interp(pcl_t)
                    H = pose_to_htm(pose).to(config["device"]).float()
                    pcl = transform_points(pcl, torch.linalg.inv(H))

                pcl_dists = torch.linalg.norm(pcl[:, :3], dim=-1)
                pcl_mask = (pcl_dists > args.pc_lim[0]) & (
                    pcl_dists < args.pc_lim[1]
                )
                pcl = pcl[pcl_mask]

                pcl = pcl[:, :3]  # assume first three are [x,y,z]

                img_idx = pcl_img_argmin[pcl_idx]
                img_fp = os.path.join(img_dir, "{:08d}.png".format(img_idx))
                img = cv2.imread(img_fp)
                img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB) / 255.0
                img = torch.tensor(img).permute(2, 0, 1).float()

                dino_feats, dino_intrinsics = pipeline.run(
                    img.unsqueeze(0), intrinsics.unsqueeze(0)
                )
                dino_feats = dino_feats[0].permute(1, 2, 0)
                dino_intrinsics = dino_intrinsics[0]

                extent = (0, dino_feats.shape[1], 0, dino_feats.shape[0])

                P = obtain_projection_matrix(dino_intrinsics, extrinsics).to(
                    config["device"]
                )
                pcl_pixel_coords = get_pixel_from_3D_source(pcl, P)
                (
                    pcl_in_frame,
                    pixels_in_frame,
                    ind_in_frame,
                ) = get_points_and_pixels_in_frame(
                    pcl, pcl_pixel_coords, dino_feats.shape[0], dino_feats.shape[1]
                )

                #ok to just int cast here
                pcl_pixel_coords = pcl_pixel_coords.long()

                pcl_px_in_frame = pcl_pixel_coords[ind_in_frame]
                dino_idxs = pcl_px_in_frame.unique(
                    dim=0
                )  # only get feats with a lidar return

                # mask_dino_feats = dino_feats[dino_idxs[:, 1], dino_idxs[:, 0]]

                mask_dino_feats = dino_feats.view(-1, dino_feats.shape[-1]).cpu()
                dino_buf.append(mask_dino_feats)

            """
            if pcl_idx % 100 == 0:
                fig, axs = plt.subplots(1, 3)
                dino_viz = dino_feats[..., :3]
                vmin = dino_viz.view(-1, 3).min(dim=0)[0].view(1,1,3)
                vmax = dino_viz.view(-1, 3).max(dim=0)[0].view(1,1,3)
                dino_viz = (dino_viz-vmin)/(vmax-vmin)

                axs[0].imshow(img.permute(1,2,0), extent=extent)
                axs[0].imshow(dino_viz.cpu(), alpha=0.5, extent=extent)
                axs[0].scatter(pcl_px_in_frame[:, 0].cpu(), dino_feats.shape[0]-pcl_px_in_frame[:, 1].cpu(), s=1., alpha=0.5)

                mask = torch.zeros_like(dino_viz[..., 0])
                mask[dino_idxs[:, 1], dino_idxs[:, 0]] = 1.
                axs[1].imshow(mask.cpu())

                plt.show()
            """

    dino_buf = torch.cat(dino_buf, dim=0)
    feat_mean = dino_buf.mean(dim=0)
    dino_feats_norm = dino_buf - feat_mean.unsqueeze(0)

    U, S, V = torch.pca_lowrank(dino_feats_norm, q=args.pca_nfeats)

    pca_res = {"mean": feat_mean.cpu(), "V": V.cpu()}
    torch.save(pca_res, args.save_to)

    dino_feats_proj = dino_feats_norm @ V
    total_feat_norm = torch.linalg.norm(dino_feats_norm, dim=-1)
    proj_feat_norm = torch.linalg.norm(dino_feats_proj, dim=-1)
    residual_feat_norm = torch.sqrt(total_feat_norm ** 2 - proj_feat_norm ** 2)
    individual_feat_norm = dino_feats_proj.abs()

    avg_feat_norm = individual_feat_norm.mean(dim=0).cpu().numpy()
    avg_residual = residual_feat_norm.mean(dim=0).cpu().numpy()
    avg_total_feat_norm = total_feat_norm.mean().cpu().numpy()
    avg_proj_feat_norm = proj_feat_norm.mean().cpu().numpy()

    plt.title(
        "Raw Data Norm: {:.4f} Projection Norm: {:.4f} Residual Norm: {:.4f}".format(
            avg_total_feat_norm, avg_proj_feat_norm, avg_residual
        )
    )
    plt.bar(
        np.arange(avg_feat_norm.shape[0]),
        np.cumsum(avg_feat_norm/avg_total_feat_norm),
        color="b",
        label="pca component norm",
    )
    plt.bar([avg_feat_norm.shape[0]], avg_residual / avg_total_feat_norm, color="r", label="residual norm")
    plt.legend()
    plt.show()

    feat_mean = feat_mean.cuda()
    V = V.cuda()

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

            pcl = pcl[:, :3]  # assume first three are [x,y,z]

            img_idx = pcl_img_argmin[pcl_idx]
            img_fp = os.path.join(img_dir, "{:08d}.png".format(img_idx))
            img = cv2.imread(img_fp)
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB) / 255.0
            img = torch.tensor(img).permute(2, 0, 1)

            dino_feats, dino_intrinsics = pipeline.run(
                img.unsqueeze(0), intrinsics.unsqueeze(0)
            )
            dino_feats = dino_feats[0].permute(1, 2, 0)
            dino_intrinsics = dino_intrinsics[0]

            extent = (0, dino_feats.shape[1], 0, dino_feats.shape[0])

            dino_feats_norm = dino_feats.view(
                -1, dino_feats.shape[-1]
            ) - feat_mean.view(1, -1)
            dino_feats_pca = dino_feats_norm.unsqueeze(1) @ V.unsqueeze(0)
            dino_feats = dino_feats_pca.view(
                dino_feats.shape[0], dino_feats.shape[1], args.pca_nfeats
            )

            P = obtain_projection_matrix(dino_intrinsics, extrinsics).to(
                config["device"]
            )
            pcl_pixel_coords = get_pixel_from_3D_source(pcl, P)
            (
                pcl_in_frame,
                pixels_in_frame,
                ind_in_frame,
            ) = get_points_and_pixels_in_frame(
                pcl, pcl_pixel_coords, dino_feats.shape[0], dino_feats.shape[1]
            )

            #ok to just int cast here
            pcl_pixel_coords = pcl_pixel_coords.long()

            pcl_px_in_frame = pcl_pixel_coords[ind_in_frame]
            dino_idxs = pcl_px_in_frame.unique(
                dim=0
            )  # only get feats with a lidar return

            fig, axs = plt.subplots(2, 3, figsize=(32, 24))
            axs = axs.flatten()
            dino_viz = dino_feats[..., :9]
            vmin = dino_viz.view(-1, 9).min(dim=0)[0].view(1, 1, 9)
            vmax = dino_viz.view(-1, 9).max(dim=0)[0].view(1, 1, 9)
            dino_viz = (dino_viz - vmin) / (vmax - vmin)

            img = img.permute(1, 2, 0).cpu().numpy()

            axs[0].imshow(img, extent=extent)

            axs[1].imshow(img, extent=extent)
            axs[1].imshow(dino_viz[..., :3].cpu(), alpha=0.5, extent=extent)

            mask = torch.zeros_like(dino_viz[..., 0])
            mask[dino_idxs[:, 1], dino_idxs[:, 0]] = 1.0
            axs[2].imshow(mask.cpu())

            axs[3].imshow(dino_viz[..., :3].cpu(), alpha=1.0, extent=extent)
            axs[4].imshow(dino_viz[..., 3:6].cpu(), alpha=1.0, extent=extent)
            axs[5].imshow(dino_viz[..., 6:].cpu(), alpha=1.0, extent=extent)

            plt.show()
