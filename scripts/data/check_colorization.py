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
    parser.add_argument('--pc_in_local', action='store_true', help='set this flag if the pc is in the sensor frame, otherwise assume in odom frame')
    parser.add_argument('--pc_lim', type=float, nargs=2, required=False, default=[5., 100.], help='limit on range (m) of pts to consider')
    parser.add_argument('--pca_nfeats', type=int, required=False, default=64, help='number of pca feats to use')
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

        pcl_dir = os.path.join(ddir, config["pointcloud"]["folder"])
        pcl_ts = np.loadtxt(os.path.join(pcl_dir, "timestamps.txt"))

        for pcl_idx in np.arange(len(pcl_ts))[500::100]:
            ## setup pc ##
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

            ## setup images ##
            images = []
            image_Ps = []
            for ii, ik in enumerate(image_keys):
                img_dir = config['images'][ik]['folder']
                img_fp = os.path.join(args.data_dir, img_dir, '{:08d}.png'.format(pcl_idx))
                img_ts = np.loadtxt(os.path.join(args.data_dir, img_dir, 'timestamps.txt'))
                img_t = img_ts[pcl_idx]

                img = cv2.imread(img_fp)
                img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB) / 255.0
                img = torch.tensor(img).float().to(config['device'])
                images.append(img)

                # import pdb;pdb.set_trace()
                # pre-multiply by the tf from pc odom to img odom
                odom_pc_H = pose_to_htm(traj_interp(pcl_t)).to(config['device'])
                odom_img_H = pose_to_htm(traj_interp(img_t)).to(config['device'])
                pc_img_H = odom_img_H @ torch.linalg.inv(odom_pc_H)
                E = pc_img_H @ image_extrinsics[ii]
                I = image_intrinsics[ii]
                image_Ps.append(get_projection_matrix(I, E))

            images = torch.stack(images, dim=0)
            image_Ps = torch.stack(image_Ps, dim=0)
            coords, valid_mask = get_pixel_projection(pcl, image_Ps, images)

            ## viz ##
            ni = len(images)
            nax = ni+2
            fig, axs = plt.subplots(2, nax, figsize=(10*nax, 10*2))

            pcl = pcl.cpu().numpy()
            pcl_dists = pcl_dists.cpu().numpy()

            axs[0, 0].scatter(pcl[:, 0], pcl[:, 1], c=pcl[:, 2], cmap='jet', s=1.)
            axs[0, 0].set_title('pc orig')

            cs = 'rgbcmyk'
            for ik, ilabel in enumerate(image_keys):
                coors = coords[ik].cpu().numpy()
                vmask = valid_mask[ik].cpu().numpy()
                axs[0, 1+ik].scatter(pcl[vmask, 0], pcl[vmask, 1], c=pcl[vmask, 2], cmap='jet', s=1.)
                axs[0, 1+ik].set_title('pts in {}'.format(ilabel))

                axs[0, -1].scatter(pcl[vmask, 0], pcl[vmask, 1], c=cs[ik], label=ilabel, s=1.)

                axs[1, 1+ik].imshow(images[ik].cpu().numpy())
                axs[1, 1+ik].scatter(coors[vmask, 0], coors[vmask, 1], s=1., cmap='jet', c=pcl_dists[vmask])
                axs[1, 1+ik].set_title(ilabel)

            for ax in axs[0]:
                ax.set_aspect(1.)

            axs[0, -1].legend()

            plt.show()