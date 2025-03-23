import os
import yaml
import tqdm
import argparse

import cv2
import torch
import numpy as np
import matplotlib.pyplot as plt

from scipy.spatial.transform import Rotation as R

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

from physics_atv_visual_mapping.localmapping.voxel.raytracing import *

"""
Run the dino mapping offline on the kitti-formatted dataset
"""

def get_render_path(pose):
    """
    Generate a render path from pose

    For now I'm doing a linear az sweep + sinusoidal el
    """
    d_az = torch.linspace(0., 2*np.pi, 100)
    d_el = torch.linspace(0., 4*np.pi, 100).sin() * (np.pi/6)

    d_eul = torch.stack([torch.zeros_like(d_az), d_el, d_az], dim=-1)
    render_path = []

    for i, eul in enumerate(d_eul):
        dR = R.from_euler(angles=eul.cpu().numpy(), seq='xyz').as_matrix()
        dR = torch.tensor(dR, dtype=torch.float, device=pose.device)
        dH = torch.eye(4, device=pose.device)
        dH[:3, :3] = dR
        render_path.append(pose @ dH)

    """
    # debug
    import open3d as o3d
    axs = []
    for pose in render_path:
        ax = o3d.geometry.TriangleMesh.create_coordinate_frame()
        ax.transform(pose.cpu().numpy())
        axs.append(ax)

    o3d.visualization.draw_geometries(axs)
    """
    render_path_quat = []

    for H in render_path:
        p = H[:3, -1]
        rot = H[:3, :3]
        q = torch.tensor(R.from_matrix(rot.cpu().numpy()).as_quat(), dtype=torch.float, device=pose.device)
        pq = torch.cat([p, q])
        render_path_quat.append(pq)

    render_path = torch.stack(render_path_quat, dim=0)

    return render_path

def render_voxel_grid(pose, voxel_grid, sensor_model):
    """
    render a voxel grid from a given pose
    """
    n_el = sensor_model['el_bins'].shape[0] - 1
    n_az = sensor_model['az_bins'].shape[0] - 1

    #turn off for now

    #only work with feature voxels
    voxel_pts = voxel_grid.grid_indices_to_pts(voxel_grid.raster_indices_to_grid_indices(voxel_grid.raster_indices[voxel_grid.feature_mask]))

    voxel_el_az_range = get_el_az_range_from_xyz(pose, voxel_pts)
    voxel_mindist_el_az_bins = bin_el_az_range(voxel_el_az_range, sensor_model=sensor_model, reduce='min')

    depth_image_valid_mask = voxel_mindist_el_az_bins > 1e-6

    voxel_bin_idxs = get_el_az_range_bin_idxs(voxel_el_az_range, sensor_model=sensor_model)
    voxel_valid_bin = (voxel_bin_idxs >= 0)

    voxel_pt_mindist = voxel_mindist_el_az_bins[voxel_bin_idxs]
    voxel_is_mindist = (voxel_el_az_range[:, 2] - voxel_pt_mindist).abs() < 1e-6

    feat_proj_mask = (voxel_is_mindist & voxel_valid_bin)

    scatter_pts = voxel_pts[feat_proj_mask]
    scatter_idxs = voxel_bin_idxs[feat_proj_mask]
    feats_to_scatter = voxel_grid.features[feat_proj_mask]

    #for smaller fdims it's prob ok to just copy feats
    #note that I'm not correctly handling inter-voxel feat/depth scattering here
    vnx = 5
    vny = 5
    vnz = 5

    voxel_noise = torch.stack(torch.meshgrid([
        torch.linspace(0., 1., vnx),
        torch.linspace(0., 1., vny),
        torch.linspace(0., 1., vnz),
    ]), dim=-1).view(-1, 3).to(voxel_grid.device) * voxel_grid.metadata.resolution.view(1, 3)

    voxel_noise_pts = (scatter_pts.unsqueeze(1) + voxel_noise.unsqueeze(0)).view(-1, 3)
    voxel_noise_features = feats_to_scatter.unsqueeze(1).tile(1, vnx*vny*vnz, 1).view(-1, feats_to_scatter.shape[-1])

    voxel_noise_el_az_range = get_el_az_range_from_xyz(pose, voxel_noise_pts)
    voxel_noise_bin_idxs = get_el_az_range_bin_idxs(voxel_noise_el_az_range, sensor_model=sensor_model)
    voxel_noise_mindist_el_az_bins = bin_el_az_range(voxel_noise_el_az_range, sensor_model=sensor_model, reduce='min')
    
    img_el_idxs = (voxel_noise_bin_idxs / n_az).long()
    img_az_idxs = voxel_noise_bin_idxs % n_az

    #conveniently, scatter_idxs are guaranteed to be unique
    _, (fmin, fmax) = normalize_dino(voxel_grid.features, return_min_max = True)
    feat_img = torch.zeros(n_el, n_az, 3, device=voxel_grid.device)

    viz_feats = (voxel_noise_features[:, :3] - fmin.view(1, 3)) / (fmax-fmin).view(1, 3)
    viz_feats = viz_feats.clamp(0., 1.)

    feat_img[img_el_idxs, img_az_idxs] = viz_feats

    depth_img = voxel_noise_mindist_el_az_bins.reshape(n_el, n_az)

    return feat_img, depth_img

    # fig, axs = plt.subplots(1, 2)
    # axs[0].imshow(depth_img.cpu().numpy(), vmin=0., cmap='jet', origin='lower')
    # axs[1].imshow(feat_img.cpu().numpy(), origin='lower')
    # plt.show()

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

    render_sensor_model_config = {
        "type": "generic",
        "az_range": [-45., 45.],
        "n_az": 300,
        "az_thresh": "default",
        "el_range": [-30., 30.],
        "n_el": 200,
        "el_thresh": "default",
    }
    render_sensor_model = setup_sensor_model(render_sensor_model_config, device=config["device"])

    fig, axs = plt.subplots(1, 2)
    axs[0].set_title("Depth Image")
    axs[1].set_title("Feature Image")
    plt.show(block=False)

    for ii in tqdm.tqdm(range(len(image_ts))):
        pcl_fp = os.path.join(args.run_dir, args.pcl, "{:08d}.npy".format(pcl_idxs[ii]))
        image_fp = os.path.join(args.run_dir, args.image, "{:08d}.png".format(ii))
        img_t = image_ts[ii]

        pose = traj_interp(img_t)
        pose_pq = torch.tensor(pose, dtype=torch.float, device=config["device"])
        pose = pose_to_htm(pose).to(config["device"])

        pcl = torch.from_numpy(np.load(pcl_fp)).float().to(config["device"])
        img = cv2.imread(image_fp)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB) / 255.0
        img = torch.tensor(img).unsqueeze(0).permute(0, 3, 1, 2)

        feature_img, feature_intrinsics = image_pipeline.run(
            img, intrinsics.unsqueeze(0)
        )

        # move back to channels-last
        feature_img = feature_img[0].permute(1, 2, 0)
        feature_intrinsics = feature_intrinsics[0]

        I = get_intrinsics(feature_intrinsics).to(config["device"])
        E = get_extrinsics(extrinsics).to(config["device"])

        P = obtain_projection_matrix(I, E)

        pc_in_base = transform_points(pcl.clone(), torch.linalg.inv(pose))

        pixel_coordinates = get_pixel_from_3D_source(pc_in_base[:, :3], P)
        (
            lidar_points_in_frame,
            pixels_in_frame,
            ind_in_frame,
        ) = get_points_and_pixels_in_frame(
            pc_in_base[:, :3], pixel_coordinates, feature_img.shape[0], feature_img.shape[1]
        )

        feature_features = bilinear_interpolation(pixels_in_frame[..., [1,0]], feature_img)
        feature_pcl = torch.cat([pcl[ind_in_frame][:, :3], feature_features], dim=-1)

        mask = ~torch.ones(pcl.shape[0], dtype=torch.bool, device=config["device"])
        mask[ind_in_frame] = True
        feature_pcl = FeaturePointCloudTorch.from_torch(pts=pcl, features=feature_features, mask=mask)

        localmapper.update_pose(pose[:3, -1])
        localmapper.add_feature_pc(pos=pose[:3, -1], feat_pc=feature_pcl, do_raytrace=True)

        feat_img, depth_img = render_voxel_grid(pose_pq, localmapper.voxel_grid, render_sensor_model)

        for ax in axs:
            ax.cla()

        axs[0].set_title("Depth Image")
        axs[1].set_title("Feature Image")
        axs[0].imshow(depth_img.cpu().numpy(), vmin=0., cmap='jet', origin='lower')
        axs[1].imshow(feat_img.cpu().numpy(), origin='lower')
        plt.pause(0.1)

        #try spinning (it's a neat trick)
        if (ii+1) % 100 == 0 and ii >= 99:
            render_path = get_render_path(pose)

            for render_pose in render_path:
                feat_img, depth_img = render_voxel_grid(render_pose, localmapper.voxel_grid, render_sensor_model)

                for ax in axs:
                    ax.cla()

                axs[0].set_title("Depth Image")
                axs[1].set_title("Feature Image")
                axs[0].imshow(depth_img.cpu().numpy(), vmin=0., cmap='jet', origin='lower')
                axs[1].imshow(feat_img.cpu().numpy(), origin='lower')
                plt.pause(0.05)