import os
import yaml
import time
import argparse

import cv2
import torch
import numpy as np
import matplotlib.pyplot as plt

from ros_torch_converter.datatypes.pointcloud import FeaturePointCloudTorch

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
Run the dino mapping offline on the kitti-formatted dataset
"""

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--run_dir", type=str, required=True, help="path to dataset")
    parser.add_argument("--config", type=str, required=True, help="path to config")
    parser.add_argument("--pc_in_local", action='store_true', help="set this flag if pc in local frame")
    args = parser.parse_args()

    config = yaml.safe_load(open(args.config, "r"))

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

    image_pipeline = setup_image_pipeline(config)

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
    odom_dir = os.path.join(args.run_dir, config['odometry']['folder'])
    poses = np.loadtxt(os.path.join(odom_dir, "data.txt"))
    pose_ts = np.loadtxt(os.path.join(odom_dir, "timestamps.txt"))
    mask = np.ones_like(pose_ts).astype(bool)
    mask[1:] = np.abs(pose_ts[1:] - pose_ts[:-1]) > 1e-4
    poses = poses[mask]
    pose_ts = pose_ts[mask]

    traj_interp = TrajectoryInterpolator(pose_ts, poses, tol=0.5)

    ## next compute gridmap sample times
    pcl_dir = os.path.join(args.run_dir, config['pointcloud']['folder'])
    pcl_ts = np.loadtxt(os.path.join(pcl_dir, "timestamps.txt"))

    for pcl_idx in range(len(pcl_ts))[500:]:
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
            img_fp = os.path.join(args.run_dir, img_dir, '{:08d}.png'.format(pcl_idx))
            
            img = cv2.imread(img_fp)
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB) / 255.0
            img = torch.tensor(img).float().to(config['device'])
            images.append(img)

        images = torch.stack(images, dim=0)
        images = images.permute(0, 3, 1, 2)

        t1 = time.time()
        feature_images, feature_intrinsics = image_pipeline.run(
            images, image_intrinsics
        )
        torch.cuda.synchronize()
        t2 = time.time()

        #re-calc projection matrices
        image_Ps = []
        for ii, ik in enumerate(image_keys):
            img_dir = config['images'][ik]['folder']
            img_ts = np.loadtxt(os.path.join(args.run_dir, img_dir, 'timestamps.txt'))
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

        t3 = time.time()
        coords, valid_mask = get_pixel_projection(pcl_base[:, :3], image_Ps, feature_images)
        pc_features, cnt = colorize(coords, valid_mask, feature_images)
        torch.cuda.synchronize()
        t4 = time.time()

        pc_features = pc_features[cnt > 0]
        feature_pcl = FeaturePointCloudTorch.from_torch(pts=pcl_odom, features=pc_features, mask=(cnt > 0))

        t5 = time.time()
        localmapper.update_pose(pose[:3, -1])
        localmapper.add_feature_pc(pos=pose[:3, -1], feat_pc=feature_pcl, do_raytrace=False)

        if do_terrain_estimation:
            bev_features = terrain_estimator.run(localmapper.voxel_grid)
        
        torch.cuda.synchronize()
        t6 = time.time()

        print('_' * 30)
        print('VFM proc took {:.4f}s'.format(t2-t1))
        print('colorize took {:.4f}s'.format(t4-t3))
        print('mapping took  {:.4f}s'.format(t6-t5))

        ## viz code ##
        if (pcl_idx+1) % 50 == 0:
            ## viz proj ##
            ni = len(images)
            nax = ni+2
            fig, axs = plt.subplots(2, nax, figsize=(10*nax, 10*2))

            pcl = pcl_base.cpu().numpy()
            pcl_dists = np.linalg.norm(pcl, axis=-1)

            axs[0, 0].scatter(pcl[:, 0], pcl[:, 1], c=pcl[:, 2], cmap='jet', s=1.)
            axs[0, 0].set_title('pc orig')

            cs = 'rgbcmyk'
            for ik, ilabel in enumerate(image_keys):
                coors = coords[ik].cpu().numpy()
                vmask = valid_mask[ik].cpu().numpy()
                axs[0, 1+ik].scatter(pcl[vmask, 0], pcl[vmask, 1], c=pcl[vmask, 2], cmap='jet', s=1.)
                axs[0, 1+ik].set_title('pts in {}'.format(ilabel))

                axs[0, -1].scatter(pcl[vmask, 0], pcl[vmask, 1], c=cs[ik], label=ilabel, s=1.)

                axs[1, 1+ik].imshow(normalize_dino(feature_images[ik]).cpu().numpy())
                axs[1, 1+ik].scatter(coors[vmask, 0], coors[vmask, 1], s=1., cmap='jet', c=pcl_dists[vmask])
                axs[1, 1+ik].set_title(ilabel)

            for ax in axs[0]:
                ax.set_aspect(1.)

            axs[0, -1].legend()

            plt.show()

            #viz pc
            import open3d as o3d
            pc_viz = o3d.geometry.PointCloud()
            pc_viz.points = o3d.utility.Vector3dVector(pcl_base[cnt > 0].cpu().numpy())
            pc_viz.colors = o3d.utility.Vector3dVector(normalize_dino(pc_features).cpu().numpy())
            o3d.visualization.draw_geometries([pc_viz])

            localmapper.voxel_grid.visualize()
