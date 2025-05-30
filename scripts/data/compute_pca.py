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
from physics_atv_visual_mapping.utils import pose_to_htm, transform_points, normalize_dino

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

     #setup proj stuff
    image_keys = list(config['images'].keys())
    image_intrinsics = []
    image_extrinsics = []

    for ik in image_keys:
        lidar_to_cam = np.concatenate([
            np.array(config['images'][ik]['extrinsics']['p']),
            np.array(config['images'][ik]['extrinsics']['q']),
        ], axis=-1)
        extrinsics = pose_to_htm(lidar_to_cam)

        intrinsics = get_intrinsics(np.array(config['images'][ik]['intrinsics']['P']))

        image_extrinsics.append(extrinsics)
        image_intrinsics.append(intrinsics)

    image_intrinsics = torch.stack(image_intrinsics, dim=0).to(config['device'])
    image_extrinsics = torch.stack(image_extrinsics, dim=0).to(config['device'])

    image_Ps = get_projection_matrix(image_intrinsics, image_extrinsics).to(config["device"])

    #if config already has a pca, remove it.
    image_processing_config = []
    for ip_block in config['image_processing']:
        if ip_block['type'] == 'pca':
            print(colored('WARNING: found an existing PCA block in the image processing config. removing...', 'yellow'))
            break
        image_processing_config.append(ip_block)

    config['image_processing'] = image_processing_config

    image_pipeline = setup_image_pipeline(config)

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
        N_samples += poses.shape[0] * len(image_keys)

    n_frames = min(args.n_frames, N_samples)
    proc_every = int(N_samples / n_frames)

    print("computing for {} run dirs ({} samples, proc every {}th frame.)".format(len(run_dirs), N_samples, proc_every))

    for ddir in run_dirs:
        odom_dir = os.path.join(ddir, config["odometry"]["folder"])
        poses = np.loadtxt(os.path.join(odom_dir, "data.txt"))
        pose_ts = np.loadtxt(os.path.join(odom_dir, "timestamps.txt"))
        mask = np.abs(pose_ts[1:] - pose_ts[:-1]) > 1e-4
        poses = poses[1:][mask]
        pose_ts = pose_ts[1:][mask]

        traj_interp = TrajectoryInterpolator(pose_ts, poses)

        pcl_dir = os.path.join(ddir, config["pointcloud"]["folder"])
        pcl_ts = np.loadtxt(os.path.join(pcl_dir, "timestamps.txt"))

        for pcl_idx in range(len(pcl_ts))[10::proc_every]:
            pcl_fp = os.path.join(pcl_dir, "{:08d}.npy".format(pcl_idx))
            pcl_t = pcl_ts[pcl_idx]        

            pose = traj_interp(pcl_t)
            pose = pose_to_htm(pose).to(config["device"])

            pcl = torch.from_numpy(np.load(pcl_fp)).float().to(config["device"])
            
            if args.pc_in_local:
                pcl_odom = transform_points(pcl.clone(), pose)
                pcl_base = pcl.clone()
            else:
                pcl_odom = pcl.clone()
                pcl_base = transform_points(pcl.clone(), torch.linalg.inv(pose))

            ## setup images ##
            images = []
            for ii, ik in enumerate(image_keys):
                img_dir = config['images'][ik]['folder']
                img_fp = os.path.join(ddir, img_dir, '{:08d}.png'.format(pcl_idx))
                
                img = cv2.imread(img_fp)
                img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB) / 255.0
                img = torch.tensor(img).float().to(config['device'])
                images.append(img)

            images = torch.stack(images, dim=0)
            images = images.permute(0, 3, 1, 2)

            feature_images, feature_intrinsics = image_pipeline.run(
                images, image_intrinsics
            )

            #re-calc projection matrices
            image_Ps = []
            for ii, ik in enumerate(image_keys):
                img_dir = config['images'][ik]['folder']
                img_ts = np.loadtxt(os.path.join(ddir, img_dir, 'timestamps.txt'))
                img_t = img_ts[pcl_idx]

                # pre-multiply by the tf from pc odom to img odom
                odom_pc_H = pose_to_htm(traj_interp(pcl_t)).to(config['device'])
                odom_img_H = pose_to_htm(traj_interp(img_t)).to(config['device'])
                pc_img_H = odom_img_H @ torch.linalg.inv(odom_pc_H)
                E = pc_img_H @ image_extrinsics[ii]
                I = feature_intrinsics[ii]
                image_Ps.append(get_projection_matrix(I, E))
            
            image_Ps = torch.stack(image_Ps)
            # move back to channels-last
            feature_images = feature_images.permute(0, 2, 3, 1)

            coords, valid_mask = get_pixel_projection(pcl_base[:, :3], image_Ps, feature_images)

            for img, coors, vmask in zip(feature_images, coords, valid_mask):
                coors = coors[vmask].long()
                coors = torch.unique(coors, dim=0)

                feats = img[coors[:, 1], coors[:, 0]]
                dino_buf.append(feats)

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

    #add new pca to config
    pca_conf = {
        'type': 'pca',
        'args': {
            'fp': args.save_to
        }
    }
    config['image_processing'].append(pca_conf)

    image_pipeline = setup_image_pipeline(config)

    ## viz loop ##
    for ddir in run_dirs:
        odom_dir = os.path.join(ddir, config["odometry"]["folder"])
        poses = np.loadtxt(os.path.join(odom_dir, "data.txt"))
        pose_ts = np.loadtxt(os.path.join(odom_dir, "timestamps.txt"))
        mask = np.abs(pose_ts[1:] - pose_ts[:-1]) > 1e-4
        poses = poses[1:][mask]
        pose_ts = pose_ts[1:][mask]

        traj_interp = TrajectoryInterpolator(pose_ts, poses)

        pcl_dir = os.path.join(ddir, config["pointcloud"]["folder"])
        pcl_ts = np.loadtxt(os.path.join(pcl_dir, "timestamps.txt"))

        for pcl_idx in np.random.choice(np.arange(len(pcl_ts)), size=(10,)):
            pcl_fp = os.path.join(pcl_dir, "{:08d}.npy".format(pcl_idx))
            pcl_t = pcl_ts[pcl_idx]        

            pose = traj_interp(pcl_t)
            pose = pose_to_htm(pose).to(config["device"])

            pcl = torch.from_numpy(np.load(pcl_fp)).float().to(config["device"])
            
            if args.pc_in_local:
                pcl_odom = transform_points(pcl.clone(), pose)
                pcl_base = pcl.clone()
            else:
                pcl_odom = pcl.clone()
                pcl_base = transform_points(pcl.clone(), torch.linalg.inv(pose))

            ## setup images ##
            images = []
            for ii, ik in enumerate(image_keys):
                img_dir = config['images'][ik]['folder']
                img_fp = os.path.join(ddir, img_dir, '{:08d}.png'.format(pcl_idx))
                
                img = cv2.imread(img_fp)
                img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB) / 255.0
                img = torch.tensor(img).float().to(config['device'])
                images.append(img)

            images = torch.stack(images, dim=0)
            images = images.permute(0, 3, 1, 2)

            feature_images, feature_intrinsics = image_pipeline.run(
                images, image_intrinsics
            )

            #re-calc projection matrices
            image_Ps = []
            for ii, ik in enumerate(image_keys):
                img_dir = config['images'][ik]['folder']
                img_ts = np.loadtxt(os.path.join(ddir, img_dir, 'timestamps.txt'))
                img_t = img_ts[pcl_idx]

                # pre-multiply by the tf from pc odom to img odom
                odom_pc_H = pose_to_htm(traj_interp(pcl_t)).to(config['device'])
                odom_img_H = pose_to_htm(traj_interp(img_t)).to(config['device'])
                pc_img_H = odom_img_H @ torch.linalg.inv(odom_pc_H)
                E = pc_img_H @ image_extrinsics[ii]
                I = feature_intrinsics[ii]
                image_Ps.append(get_projection_matrix(I, E))
            
            image_Ps = torch.stack(image_Ps)
            # move back to channels-last
            feature_images = feature_images.permute(0, 2, 3, 1)
            images = images.permute(0, 2, 3, 1)

            coords, valid_mask = get_pixel_projection(pcl_base[:, :3], image_Ps, feature_images)
            img_masks = []

            for img, coors, vmask in zip(feature_images, coords, valid_mask):
                coors = coors[vmask].long()
                coors = torch.unique(coors, dim=0)
                mask = torch.zeros(img.shape[:2])
                mask[coors[:, 1], coors[:, 0]] = 1.

                img_masks.append(mask)

            ni = len(images)
            
            fig, axs = plt.subplots(4, ni, figsize=(8 * ni, 5 * 3))

            axs[0, 0].set_ylabel('Raw')
            axs[1, 0].set_ylabel('PCA')
            axs[1, 0].set_ylabel('Raw + PCA')
            axs[2, 0].set_ylabel('Mask')

            for i in range(ni):
                ilabel = image_keys[i]
                raw_img = images[i].cpu().numpy()
                feat_img = normalize_dino(feature_images[i]).cpu().numpy()
                mask_img = img_masks[i].cpu().numpy()

                proj_extent = (0, feat_img.shape[1], 0, feat_img.shape[0])

                axs[0, i].imshow(raw_img, extent=proj_extent)
                axs[2, i].imshow(raw_img, extent=proj_extent)
                axs[3, i].imshow(raw_img, extent=proj_extent)

                axs[1, i].imshow(feat_img, extent=proj_extent)
                axs[2, i].imshow(feat_img, extent=proj_extent, alpha=0.3)
                
                axs[2, i].imshow(mask_img, cmap='gray', extent=proj_extent, alpha=0.3)

                axs[0, i].set_title(ilabel)

            plt.show()