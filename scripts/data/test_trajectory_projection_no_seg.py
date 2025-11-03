import os
import cv2
import tqdm
import yaml
import torch
import numpy as np
import argparse
import matplotlib.pyplot as plt


from tartandriver_utils.geometry_utils import TrajectoryInterpolator

from physics_atv_visual_mapping.pointcloud_colorization.torch_color_pcl_utils import *
from physics_atv_visual_mapping.utils import pose_to_htm, transform_points

from physics_atv_visual_mapping.localmapping.voxel.raytracing import setup_sensor_model, get_el_az_range_from_xyz, bin_el_az_range, get_el_az_range_bin_idxs

from torch_mpc.cost_functions.cost_terms.utils import apply_footprint, make_footprint

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, required=True, help='path to footprint config')
    parser.add_argument(
        "--data_dir",
        type=str,
        required=True,
        help="path to KITTI-formatted dataset to process",
    )
    parser.add_argument('--pc_in_local', action='store_true', help='set this flag if the pc is in the sensor frame, otherwise assume in odom frame')
    parser.add_argument('--pc_lim', type=float, nargs=2, required=False, default=[5., 100.], help='limit on range (m) of pts to consider')
    parser.add_argument('--use_voxel_grid', action='store_true', help="(TODO) set this flag to compute occlusions with voxel grid")
    args = parser.parse_args()

    config = yaml.safe_load(open(args.config, "r"))
    print(config)


    ## get extrinsics and intrinsics
    lidar_to_cam = np.concatenate(
        [
            np.array(config["extrinsics"]["p"]),
            np.array(config["extrinsics"]["q"]),
        ],
        axis=-1,
    )
    extrinsics = pose_to_htm(lidar_to_cam)

    intrinsics = get_intrinsics(torch.tensor(config["intrinsics"]["K"]).reshape(3, 3))

    ## setup vehicle footprint
    footprint = make_footprint(**config['footprint']['params'])
    #make footprint 3d
    footprint = torch.cat([
        footprint,
        torch.ones_like(footprint[..., 0]).unsqueeze(-1) * config['z_offset']
    ], dim=-1).view(-1, 3).to(config['device'])

    ## setup occlusion checker
    sensor_model = setup_sensor_model(config["occlusion_sensor_model"], device=config['device'])

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

            #pc burn in 
            pcl_start_idx = max(0, pcl_idx - 500)
            res = []
            for i in range(pcl_start_idx, pcl_idx+500, 5):
                pcl_fp = os.path.join(pcl_dir, "{:08d}.npy".format(i))
                pcl = torch.from_numpy(np.load(pcl_fp)).to(config["device"]).float()
                res.append(pcl)

            pcl = torch.cat(res, dim=0)

            img_idx = pcl_img_argmin[pcl_idx]
            img_fp = os.path.join(img_dir, "{:08d}.png".format(img_idx))
            img_t = img_ts[img_idx]
            img = cv2.imread(img_fp)
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB) / 255.0
            img = torch.from_numpy(img)

            img_pose = traj_interp(img_t)
            img_H = pose_to_htm(img_pose).to(config["device"]).float()
            img_pose = torch.tensor(img_pose).to(config['device']).float()

            if torch.linalg.norm(img_pose[7:10]) < 1.0:
                print('not enough motion. skipping...')
                continue

            #setup occlusion checking
            pcl_el_az_range = get_el_az_range_from_xyz(img_pose, pcl)
            depth_image = bin_el_az_range(pcl_el_az_range, sensor_model=sensor_model, reduce='min')
            depth_image_valid_mask = depth_image > 5.0

            # #debug viz
            # import matplotlib.pyplot as plt
            # fig, axs = plt.subplots(1, 3)
            # axs[0].imshow(img)

            # n_el = sensor_model['el_bins'].shape[0] - 1
            # n_az = sensor_model['az_bins'].shape[0] - 1
            # axs[1].imshow(depth_image.reshape(n_el, n_az).cpu().numpy(), vmin=0., cmap='jet', origin='lower')
            # plt.show()

            footprint_pts_in_origin = []

            n_invalid = 0

            for cnt in range(1000):
                curr_t = img_t + cnt * 0.1
                curr_pose = traj_interp(curr_t)
                curr_htm = pose_to_htm(curr_pose).to(config['device'])
                curr_pts = transform_points(footprint.clone(), curr_htm)

                #occlusion check
                curr_pts_in_origin = transform_points(curr_pts.clone(), torch.linalg.inv(img_H))
                curr_pts_sph = get_el_az_range_from_xyz(img_pose, curr_pts)

                #for debug
                curr_pts_depth_image = bin_el_az_range(curr_pts_sph, sensor_model=sensor_model, reduce='min')

                curr_pts_bin_idxs = get_el_az_range_bin_idxs(curr_pts_sph, sensor_model=sensor_model)

                curr_pts_ranges = curr_pts_sph[:, 2]
                occlusion_ranges = depth_image[curr_pts_bin_idxs]

                #only check footprint pts that project to a valid bin and have corresponding depth
                curr_pts_valid_bin = (curr_pts_bin_idxs >= 0)

                if not curr_pts_valid_bin.any():
                    print('footprint not in view. skipping...')
                    continue

                curr_pts_has_depth = (occlusion_ranges > 1e-6)

                curr_pts_occluded = (curr_pts_ranges > occlusion_ranges + config['occlusion_thresh']) & curr_pts_valid_bin & curr_pts_has_depth

                occlusion_frac = curr_pts_occluded.sum() / curr_pts_valid_bin.sum()

                print('occ frac = {}'.format(occlusion_frac))

                if occlusion_frac > config['occlusion_frac']:
                    n_invalid += 1
                else:
                    n_invalid = 0

                if n_invalid > 9:
                    break

                footprint_pts_in_origin.append(curr_pts_in_origin[~curr_pts_occluded])

                # # debug
                # import matplotlib.pyplot as plt
                # fig, axs = plt.subplots(1, 4)
                # axs[0].imshow(img)

                # n_el = sensor_model['el_bins'].shape[0] - 1
                # n_az = sensor_model['az_bins'].shape[0] - 1
                # axs[1].imshow(depth_image.reshape(n_el, n_az).cpu().numpy(), vmin=0., cmap='jet', origin='lower')

                # axs[2].imshow(curr_pts_depth_image.reshape(n_el, n_az).cpu().numpy(), vmin=0., vmax=depth_image.max(), cmap='jet', origin='lower')

                # axs[3].imshow((depth_image - curr_pts_depth_image).reshape(n_el, n_az).cpu().numpy(), cmap='jet', origin='lower')

                # plt.title('{} valid pts, {} occluded'.format(curr_pts_valid_bin.sum(), curr_pts_occluded.sum()))

                # # axs[3].imshow(curr_pts_occluded.reshape(n_el, n_az).cpu().numpy(), vmin=0., cmap='jet', origin='lower')

                # plt.show()

            # pose_ts = img_t + np.linspace(0., 10., 101)
            # poses = traj_interp(pose_ts)
            
            # footprint_pts = []
            # for pose in poses:
            #     htm = pose_to_htm(pose).to(config['device'])
            #     pts = transform_points(footprint.clone(), htm)
            #     footprint_pts.append(pts)

            if len(footprint_pts_in_origin) == 0:
                continue

            footprint_pts_in_origin = torch.cat(footprint_pts_in_origin, dim=0)

            footprint_pcl = footprint_pts_in_origin.clone()
            
            #convert to local
            # footprint_pcl = transform_points(footprint_pcl, torch.linalg.inv(img_H))
            footprint_pcl_dists = torch.linalg.norm(footprint_pcl, dim=-1)

            P = get_projection_matrix(intrinsics, extrinsics).to(
                config["device"]
            )
            
            print(footprint_pcl.median(dim=0))

            footprint_pcl_px_in_frame, ind_in_frame = get_pixel_projection(footprint_pcl, P.unsqueeze(0),img.unsqueeze(0))
            footprint_pcl_px_in_frame = footprint_pcl_px_in_frame[0]
            ind_in_frame = ind_in_frame[0]

            footprint_pcl_px_in_frame = footprint_pcl_px_in_frame[ind_in_frame]

            #convert to mask
            traj_mask_px = torch.unique(footprint_pcl_px_in_frame.long(), dim=0)
            
            #temp hack for veh mask
            traj_mask_px = traj_mask_px[traj_mask_px[:, 1] < 490]

            if traj_mask_px.shape[0] == 0:
                continue

            traj_mask = torch.zeros(img.shape[0], img.shape[1], dtype=torch.bool)
            traj_mask[traj_mask_px[..., 1], traj_mask_px[..., 0]] = True
            traj_mask_viz = torch.stack([
                torch.zeros_like(traj_mask).float(),
                torch.zeros_like(traj_mask).float(),
                torch.zeros_like(traj_mask).float(),
                1. - torch.ones_like(traj_mask).float(),
            ])

            

            fig, axs = plt.subplots(2, 3, figsize=(24, 24))
            axs = axs.flatten()

            for ax in axs:
                ax.imshow(img)

            
            #mask and query pts
            axs[2].imshow(img)
            axs[2].imshow(traj_mask, alpha=0.5, cmap='gray')


            axs[0].set_title('image')
            axs[1].set_title('traj pts')
            axs[2].set_title('mask + query')

            plt.show()
