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
from physics_atv_visual_mapping.utils import pose_to_htm

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, required=True, help="path to config")
    parser.add_argument(
        "--data_dir",
        type=str,
        required=True,
        help="path to KITTI-formatted dataset to process",
    )
    parser.add_argument(
        "--save_to", type=str, required=True, help="path to save PCA to"
    )
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
    if config["odometry"]["folder"] in os.listdir(args.data_dir):
        run_dirs = [args.data_dir]
    else:
        run_dirs = [os.path.join(args.data_dir, x) for x in os.listdir(args.data_dir)]

    print("computing for {} run dirs".format(len(run_dirs)))

    for ddir in tqdm.tqdm(run_dirs):
        odom_dir = os.path.join(ddir, config["odometry"]["folder"])
        poses = np.load(os.path.join(odom_dir, "odometry.npy"))
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
            if pcl_idx % 10 == 0:
                pcl_fp = os.path.join(pcl_dir, "{:06d}.npy".format(pcl_idx))
                pcl = torch.from_numpy(np.load(pcl_fp)).to(config["device"]).float()

                pcl_dists = torch.linalg.norm(pcl[:, :3], dim=-1)
                pcl_mask = (pcl_dists > config["pcl_mindist"]) & (
                    pcl_dists < config["pcl_maxdist"]
                )
                pcl = pcl[pcl_mask]

                pcl = pcl[:, :3]  # assume first three are [x,y,z]

                img_idx = pcl_img_argmin[pcl_idx]
                img_fp = os.path.join(img_dir, "{:06d}.png".format(img_idx))
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

                pcl_px_in_frame = pcl_pixel_coords[ind_in_frame]
                dino_idxs = pixels_in_frame.unique(
                    dim=0
                )  # only get feats with a lidar return

                mask_dino_feats = dino_feats[dino_idxs[:, 1], dino_idxs[:, 0]]
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

    U, S, V = torch.pca_lowrank(dino_feats_norm, q=config["pca_nfeats"])

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
        avg_feat_norm,
        color="b",
        label="pca component norm",
    )
    plt.bar([avg_feat_norm.shape[0]], avg_residual, color="r", label="residual norm")
    plt.legend()
    plt.show()

    ## viz loop ##
    for ddir in run_dirs:
        odom_dir = os.path.join(ddir, config["odometry"]["folder"])
        poses = np.load(os.path.join(odom_dir, "odometry.npy"))
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
            pcl_fp = os.path.join(pcl_dir, "{:06d}.npy".format(pcl_idx))
            pcl = torch.from_numpy(np.load(pcl_fp)).to(config["device"]).float()

            pcl_dists = torch.linalg.norm(pcl[:, :3], dim=-1)
            pcl_mask = (pcl_dists > config["pcl_mindist"]) & (
                pcl_dists < config["pcl_maxdist"]
            )
            pcl = pcl[pcl_mask]

            pcl = pcl[:, :3]  # assume first three are [x,y,z]

            img_idx = pcl_img_argmin[pcl_idx]
            img_fp = os.path.join(img_dir, "{:06d}.png".format(img_idx))
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
                dino_feats.shape[0], dino_feats.shape[1], config["pca_nfeats"]
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

            pcl_px_in_frame = pcl_pixel_coords[ind_in_frame]
            dino_idxs = pixels_in_frame.unique(
                dim=0
            )  # only get feats with a lidar return

            fig, axs = plt.subplots(2, 2, figsize=(32, 24))
            axs = axs.flatten()
            dino_viz = dino_feats[..., :3]
            vmin = dino_viz.view(-1, 3).min(dim=0)[0].view(1, 1, 3)
            vmax = dino_viz.view(-1, 3).max(dim=0)[0].view(1, 1, 3)
            dino_viz = (dino_viz - vmin) / (vmax - vmin)

            img = img.permute(1, 2, 0).cpu().numpy()

            axs[0].imshow(img, extent=extent)
            #            axs[0].imshow(dino_viz.cpu(), alpha=0.5, extent=extent)
            #            axs[0].scatter(pcl_px_in_frame[:, 0].cpu(), dino_feats.shape[0]-pcl_px_in_frame[:, 1].cpu(), s=1., alpha=0.5)

            axs[1].imshow(img, extent=extent)
            axs[1].imshow(dino_viz.cpu(), alpha=0.5, extent=extent)

            mask = torch.zeros_like(dino_viz[..., 0])
            mask[dino_idxs[:, 1], dino_idxs[:, 0]] = 1.0
            axs[2].imshow(mask.cpu())

            axs[3].imshow(dino_viz.cpu(), alpha=1.0, extent=extent)

            plt.show()
