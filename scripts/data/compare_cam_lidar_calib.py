import os
import time
import argparse
import json

import torch
import numpy as np

import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
import matplotlib.animation as animation

from physics_atv_visual_mapping.pointcloud_colorization.torch_color_pcl_utils import *

from tartandriver_utils.os_utils import load_yaml

from ros_torch_converter.tf_manager import TfManager
from ros_torch_converter.datatypes.intrinsics import IntrinsicsTorch
from ros_torch_converter.datatypes.image import ImageTorch
from ros_torch_converter.datatypes.pointcloud import PointCloudTorch

# Visualizes qualitative comparison of vehicle->RGB extrinsics by projecting input point clouds to
# rgb image frame
# offset argument is used to set arbitrary starting frames (i.e., non-zero) in KITTI dataset
def update_viz(frame_idx, offset, fig, args, sensors_dir, E_new, E_old, tf_manager, max_depth, im_objs, scat_objs, ax, save=False):
    if frame_idx % 50 == 0:
        print(f"Frame {frame_idx + offset}")

    # Get data to vizualize
    img = ImageTorch.from_kitti(os.path.join(sensors_dir, args.rgb_image_dir), frame_idx, device='cuda')
    intrinsics = IntrinsicsTorch.from_kitti(os.path.join(sensors_dir, args.rgb_intrinsics_dir), frame_idx, device='cuda')
    pc = PointCloudTorch.from_kitti(os.path.join(sensors_dir, args.pc_dir), frame_idx, device='cuda')

    if pc.frame_id not in {"velodyne_1", "vehicle"}:
        # need to transform pc to vehicle frame first
        tf_to_vehicle = tf_manager.get_transform("vehicle", pc.frame_id, img.stamp).cuda().transform
        pts_orig = pc.pts

        pts_orig_H = torch.cat([pts_orig, torch.ones_like(pts_orig[:, [0]])], axis=-1)
        pts_vehicle = tf_to_vehicle.view(1, 4, 4) @ pts_orig_H.view(-1, 4, 1)

        # update pc
        pc.pts = pts_vehicle[:, :3, 0]
        pc.frame_id = "vehicle"

    # make upper right corner 3x3
    I = torch.eye(4).cuda()
    I[:3, :3] = intrinsics.intrinsics
    P_old = get_projection_matrix(I, E_old)
    P_new = get_projection_matrix(I, E_new)

    # project pc to img
    image = img.image
    pts = pc.pts
    
    coords_old, valid_mask_old = get_pixel_projection(pts, P_old.unsqueeze(0), image.unsqueeze(0))
    valid_mask_old2 = cleanup_projection(pts, coords_old, valid_mask_old, image.unsqueeze(0))
    coords_old = coords_old.squeeze(0)
    valid_mask_old = valid_mask_old.squeeze(0) & valid_mask_old2.squeeze(0)

    coords_new, valid_mask_new = get_pixel_projection(pts, P_new.unsqueeze(0), image.unsqueeze(0))
    valid_mask_new2 = cleanup_projection(pts, coords_new, valid_mask_new, image.unsqueeze(0))
    coords_new = coords_new.squeeze(0)
    valid_mask_new = valid_mask_new.squeeze(0) & valid_mask_new2.squeeze(0)

    # generate scatter plot of depths overlaid onto img
    depths_scat_old = torch.linalg.norm(pts, dim=-1)[valid_mask_old].cpu().numpy()
    coords_scat_old = coords_old[valid_mask_old].cpu().numpy()

    depths_scat_new = torch.linalg.norm(pts, dim=-1)[valid_mask_new].cpu().numpy()
    coords_scat_new = coords_new[valid_mask_new].cpu().numpy()

    # remove points past max_depth, if defined
    if max_depth:
        max_depth_mask_old = depths_scat_old <= max_depth
        depths_scat_old = depths_scat_old[max_depth_mask_old]
        coords_scat_old = coords_scat_old[max_depth_mask_old]

        max_depth_mask_new = depths_scat_new <= max_depth
        depths_scat_new = depths_scat_new[max_depth_mask_new]
        coords_scat_new = coords_scat_new[max_depth_mask_new]

    # compute difference in projections
    valid_mask_common = valid_mask_old & valid_mask_new
    coords_old_common = coords_old[valid_mask_common].cpu().numpy()
    coords_new_common = coords_new[valid_mask_common].cpu().numpy()

    # remove points past max_depth, if defined
    depths_common = torch.linalg.norm(pts, dim=-1)[valid_mask_common].cpu().numpy()
    if max_depth:
        max_depth_mask_common = depths_common <= max_depth
        coords_old_common = coords_old_common[max_depth_mask_common]
        coords_new_common = coords_new_common[max_depth_mask_common]

    error = np.linalg.norm(coords_old_common - coords_new_common, axis=-1)

    # Update plot
    image_disp = cv2.cvtColor(image.cpu().numpy(), cv2.COLOR_BGR2RGB)

    # old
    im_objs[0].set_data(image_disp)
    scat_objs[0].set_offsets(coords_scat_old)
    scat_objs[0].set_array(depths_scat_old)
    scat_objs[0].set_clim(vmin=np.min(depths_scat_old), vmax=np.max(depths_scat_old))
    ax[0].set_title('Old Extrinsics')
    ax[0].axis('off')

    # new
    im_objs[1].set_data(image_disp)
    scat_objs[1].set_offsets(coords_scat_new)
    scat_objs[1].set_array(depths_scat_new)
    scat_objs[1].set_clim(vmin=np.min(depths_scat_new), vmax=np.max(depths_scat_new))
    ax[1].set_title('New Extrinsics')
    ax[1].axis('off')

    # error
    im_objs[2].set_data(np.ones(image_disp.shape))
    scat_objs[2].set_offsets(coords_old_common)
    scat_objs[2].set_array(error)
    scat_objs[2].set_clim(vmin=np.min(error), vmax=np.max(error))
    ax[2].set_title('Error (old - new)')
    ax[2].axis('off')

    if not save:
        fig.canvas.manager.set_window_title(f"Frame: {frame_idx}")
    else:
        fig.suptitle(f"Frame: {frame_idx}")
    
    return im_objs + scat_objs

def init_plot():
    fig, ax = plt.subplots(1, 3, figsize=(20,10))

    im_objs = [ax[i].imshow(np.ones((544, 1024, 3))) for i in range(3)]
    scat_objs = [ax[i].scatter(np.zeros(10), np.zeros(10), c=np.zeros(10), cmap='jet', s=1, vmin=0, vmax=20) for i in range(3)]
    ax[0].axis('off')
    ax[1].axis('off')
    ax[2].axis('off')

    # create colorbar for first/second scatterplot
    ext_cbar = plt.colorbar(scat_objs[0], ax=ax[0])
    # create colorbar for third scatterplot
    error_cbar = plt.colorbar(scat_objs[2], ax=ax[2])
    
    return fig, ax, im_objs, scat_objs

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--run_dir', type=str, required=True, help='path to KITTI dataset dir to parse')
    parser.add_argument('--save_path', required=False, help='path to save mp4/fig (include file name but not extension)')
    parser.add_argument('--rgb_image_dir', type=str, required=True, help='subdir name for rgb images in KITTI dataset')
    parser.add_argument('--rgb_intrinsics_dir', type=str, required=True, help='subdir name for rgb intrinsics in KITTI dataset')
    parser.add_argument('--pc_dir', type=str, required=True, help='subdir name for point clouds in KITTI dataset')
    parser.add_argument('--max_depth', type=int, required=False, help='maximum depth to show in projection')
    args = parser.parse_args()

    save_path = args.save_path
    save = False if not save_path else True

    sensors_dir = os.path.join(args.run_dir, 'sensors')
    N = len(np.loadtxt(os.path.join(sensors_dir, args.rgb_image_dir, 'timestamps.txt')))

    # init plot
    fig, ax, im_objs, scat_objs = init_plot()

    # tf manager
    calib_fp = "/home/tartandriver/tartandriver_ws/src/core/static_tf_publisher/config/offroad/yamaha.yaml"
    calib_config = load_yaml(calib_fp)
    tf_manager = TfManager.from_kitti(args.run_dir)
    tf_manager.update_from_calib_config(calib_config)

    # extrinsics to compare
    E_new = torch.tensor([
        [0.007624, -0.999948,  0.006789,  0.084273],
        [-0.220747, -0.008305, -0.975296, -0.167061],
        [0.975301,  0.005937, -0.220799,  0.036073],
        [0.0,        0.0,        0.0,        1.0]
    ]).float().cuda()

    E_old = torch.tensor([
        [0.00330039, -0.99971588,  0.02360654,  0.17265],
        [-0.22465677, -0.02374448, -0.97414862, -0.15227],
        [0.97443237, -0.0020883,  -0.22467131,  0.05708],
        [0.0,          0.0,          0.0,          1.0]
    ]).float().cuda()

    max_depth = args.max_depth

    # Animation mode
    # offset defines the starting frame for the animation
    # frames defines how many frames to show
    # frames + offset <= # frames in KITTI dataset
    # frames = N
    # offset = 0
    # ani = FuncAnimation(
    #     fig,
    #     update_viz,
    #     frames=frames,
    #     fargs=(offset, fig, args, sensors_dir, E_new, E_old, tf_manager, max_depth, im_objs, scat_objs, ax, save),
    #     interval=8,
    #     blit=True
    # )

    # if not save:
    #     plt.show()
    # else:
    #     ani.save(f'{save_path}.mp4', writer='ffmpeg', fps=5)

    # Step through frames (click in plot to advance frame)
    # for i in range(N):
    #     update_viz(i, 0, fig, args, sensors_dir, E_new, E_old, tf_manager, max_depth, im_objs, scat_objs, ax, save)
    #     plt.draw()
    #     if save:
    #         plt.savefig(f'{save_path}_f{i}.png')
    #     if plt.waitforbuttonpress(timeout=10) is None:
    #         print("Timed out")
    #         plt.close()
    #         break

    # Show arbitrary frame
    # ii = 100
    # update_viz(ii, 0, fig, args, sensors_dir, E_new, E_old, tf_manager, max_depth, im_objs, scat_objs, ax, save)

    # if not save:
    #     plt.show()
    # else:
    #     plt.savefig(f'{save_path}_f{ii}.png')
    