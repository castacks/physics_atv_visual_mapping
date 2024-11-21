import numpy as np
import rclpy
import torch

# import matplotlib.pyplot as plt
import cv2
from functools import reduce
import time
import struct
import std_msgs.msg
from sensor_msgs.msg import PointCloud2, PointField, Image, CameraInfo
from std_msgs.msg import Header

# TODO fix assumptions of 3/4D pointclouds (ideally just pass in the first 3 dimensions when calling these functions?)


def remove_invalid(lidar_points):
    """Removes invalid points from dense pointcloud to create sparse pointcloud.

    Args:
        - lidar_points:
            Nx3 array of XYZ points in lidar frame of reference

    Returns:
        - valid_points:
            Nx3 array of XYZ points that are not all 0s or NaNs
    """

    if lidar_points.shape[1] >= 4:
        lidar_points = lidar_points[
            :, :3
        ]  # Makes sure we only deal with XYZ information

    # Remove rows full of zeros or NaNs
    lidar_points_norm = torch.norm(lidar_points, dim=1)
    # empty_pts_idxs = torch.where(lidar_points_norm < 1e-5) # or np.isnan(lidar_points_norm))
    valid_points = lidar_points[lidar_points_norm > 1e-5]

    return valid_points


def get_intrinsics(intrinsics_matrix, tf_in_optical=True):
    """Returns intrinsics matrix depending on whether we have a transform to the camera available in optical_frame or not.

    Args:
        - intrinsics_matrix:
            Known 3x3 intrinsics matrix.
        - tf_in_optical:
            Boolean. True if the transform that we know for the camera extrinsics will be given in optical frame (meaning that z points forward, x points to the right, and y points down). False if the axes for the camera frame are aligned with the source frame (e.g. we know the location of the camera link wrt the source frame but it does not account for the rotation between camera coordinates and source coordinates).

    Returns:
        - intrinsics_matrix:
            4x4 intrinsics matrix that takes into account rotation between camera axes and source axes.
    """

    if tf_in_optical:
        I = torch.eye(4, device=intrinsics_matrix.device)
        I[:3, :3] = intrinsics_matrix
        return I
    else:
        T_p_i = torch.tensor([[-1, 0, 0], [0, -1, 0], [0, 0, -1]], dtype=torch.float32, device=intrinsics_matrix.device)
        intrinsics_matrix = torch.matmul(T_p_i, intrinsics_matrix)
        I = torch.ones(4)
        I[:3, :3] = intrinsics_matrix
        return I


def get_extrinsics(extrinsics_matrix, tf_in_optical=True):
    """Returns extrinsics matrix that takes into account rotation between camera axes and source axes.

    Args:
        - extrinsics_matrix:
            Known 4x4 extrinsics matrix that represents transformation from the camera frame (reference frame) to the source frame. Another way to say this, is that it is that the extrinsics matrix is the source frame represented in the camera frame.
        - tf_in_optical:
            Boolean. True if extrinsics matrix already takes into account rotation between the source coordinates and camera coordinates. False otherwise: if the axes for the camera frame are aligned with the source frame (and we need to perform two 90 degree rotations in two axes to align them, on top of whatever rotation exists between the two frames in a global frame of reference).

    Returns:
        - extrinsics_matrix:
            4x4 extrinsics matrix that takes into account rotation between camera axes and source axes.
    """

    if tf_in_optical:
        return extrinsics_matrix
    else:
        # TODO
        return extrinsics_matrix


def obtain_projection_matrix(intrinsics, extrinsics):
    """Returns projection matrix from 3D points in "target" coordinate frame to 2D points in pixel space.

    Args:
        - intrinsics:
            4x4 matrix of intrinsics that takes into account rotation between camera axes and source axes. Assumes intrinsics matrix has the form [I, 0; 0, 1] where I is 3x3.
        - extrinsics:
            4x4 matrix of extrinsics that takes into account rotation between camera axes and source axes.

    Returns:
        - P:
            3x4 camera projection matrix that transforms 3D points in "source" coordinate frame to 2D points in pixel space.
    """

    P = torch.matmul(intrinsics, extrinsics)
    P = P[:-1, :]

    return P


def get_pixel_from_3D_source(lidar_points, P):
    """Returns pixel coordinates from points in 3D given a valid projection matrix.

    Args:
        - lidar_points:
            Nx3 matrix of XYZ points in "source" frame of reference (global frame or local lidar frame).
        - P:
            3x4 camera projection matrix that transforms 3D points in "source" frame of reference into 2D coordinates in pixel frame of reference.

    Returns:
        - pixel_coordinates:
            Nx2 matrix of XY coordinates in pixel frame of reference (Origin at top left of image, (+)x points to the right, and (+)y points down).
    """

    ones = torch.ones((lidar_points.shape[0], 1)).to(lidar_points.device)
    homo_3D = torch.cat((lidar_points, ones), dim=1)
    homo_pixel = (torch.matmul(P, homo_3D.T)).T
    homo_pixel_norm = homo_pixel / homo_pixel[:, [2]]
    pixel_coordinates = homo_pixel_norm[:, :-1]

    return pixel_coordinates


def get_points_and_pixels_in_frame(
    lidar_points, pixel_coordinates, image_height, image_width
):
    """Returns a) array of pixels that lie inside image frame, b) indices of these pixels in the input pixel_coordinates array (to then match with pointcloud).

    Args:
        - lidar_points:
            Nx3 matrix of XYZ coordinates in "source" (lidar) frame of reference
        - pixel_coordinates:
            Nx2 matrix of XY coordinates in pixel frame of reference
        - image_height:
            Int, height of image
        - image_width:
            Int, width of image

    Returns:
        - lidar_points_in_frame:
            Nx3 matrix of XYZ coordinates in "source" (lidar) frame of reference that lie within image frame
        - pixels_in_frame:
            Nx2 matrix of XY pixel coordinates that lie within image frame
        -ind_in_frame:
            N2x1 mask of the original pointcloud points, where points in frame have true
    """
    pixel_coords_x = pixel_coordinates[:, 0]
    pixel_coords_y = pixel_coordinates[:, 1]
    lidar_points_x = lidar_points[:, 0]
    lidar_points_z = lidar_points[:, 2]

    ## Make sure pixels are within image frame
    cond1 = pixel_coords_x >= 0
    cond2 = pixel_coords_x < image_width
    cond3 = pixel_coords_y >= 0
    cond4 = pixel_coords_y < image_height
    ## Make sure lidar points are in front of camera
    cond5 = lidar_points_x > 0
    ## Don't count points above a certain height
    cond6 = lidar_points_z < 1

    ind_in_frame = cond1 & cond2 & cond3 & cond4 & cond5 #& cond6

    lidar_points_in_frame = lidar_points[ind_in_frame, :]
    pixels_in_frame = pixel_coordinates[ind_in_frame, :].long()

    return lidar_points_in_frame, pixels_in_frame, ind_in_frame


def get_rgb_from_pixel_coords(image, pixel_coords):
    """Returns Nx3 array of RGB values from array of pixel coordinates.
    TODO Fill the rest of this out
    """

    # pixel_coords_tuple = (pixel_coords[:,1].flatten(), pixel_coords[:,0].flatten())
    # print(pixel_coords.shape)

    rgb_vals = image[pixel_coords[:, 1], pixel_coords[:, 0]]

    return rgb_vals


def xyz_array_to_point_cloud_msg(points, frame, timestamp=None, rgb_values=None):
    """
    Modified from: https://github.com/castacks/physics_atv_deep_stereo_vo/blob/main/src/stereo_node_multisense.py
    Please refer to this ros answer about the usage of point cloud message:
        https://answers.ros.org/question/234455/pointcloud2-and-pointfield/
    :param points:
    :param header:
    :return:
    """

    # points = points.cpu().numpy()

    header = Header()
    header.frame_id = frame
    if timestamp is None:
        timestamp = rclpy.clock.Clock().now().to_msg()
    header.stamp = timestamp
    msg = PointCloud2()
    msg.header = header
    if len(points.shape) == 3:
        msg.width = points.shape[0]
        msg.height = points.shape[1]
    else:
        msg.width = points.shape[0]
        msg.height = 1
    msg.is_bigendian = False
    # organized clouds are non-dense, since we have to use std::numeric_limits<float>::quiet_NaN()
    # to fill all x/y/z value; un-organized clouds are dense, since we can filter out unfavored ones
    msg.is_dense = False

    if rgb_values is None:
        msg.fields = [
            PointField("x", 0, PointField.FLOAT32, 1),
            PointField("y", 4, PointField.FLOAT32, 1),
            PointField("z", 8, PointField.FLOAT32, 1),
        ]
        msg.point_step = 12
        msg.row_step = msg.point_step * msg.width
        xyz = points.astype(np.float32)
        msg.data = xyz.tostring()
    else:
        msg.fields = [
            PointField("x", 0, PointField.FLOAT32, 1),
            PointField("y", 4, PointField.FLOAT32, 1),
            PointField("z", 8, PointField.FLOAT32, 1),
            PointField("rgb", 12, PointField.UINT32, 1),
        ]
        msg.point_step = 16
        msg.row_step = msg.point_step * msg.width

        xyzcolor = np.zeros(
            (points.shape[0], 1),
            dtype={
                "names": ("x", "y", "z", "rgba"),
                "formats": ("f4", "f4", "f4", "u4"),
            },
        )
        xyzcolor["x"] = points[:, 0].reshape((-1, 1))
        xyzcolor["y"] = points[:, 1].reshape((-1, 1))
        xyzcolor["z"] = points[:, 2].reshape((-1, 1))
        color_rgba = np.zeros((points.shape[0], 4), dtype=np.uint8) + 255
        color_rgba[:, :3] = rgb_values[:, :3]
        xyzcolor["rgba"] = color_rgba.view("uint32")
        msg.data = xyzcolor.tostring()

    return msg


def create_point_cloud(points, parent_frame="lidar", colors=None):
    """Creates a point cloud message.

    From: https://gist.github.com/pgorczak/5c717baa44479fa064eb8d33ea4587e0

    Args:
        points:
            Nx3 array of xyz positions (m)
        colors:
            Nx3 or Nx4 array of rgba colors (0..1) [Optional]
        parent_frame:
            frame in which the point cloud is defined
    Returns:
        sensor_msgs/PointCloud2 message
    """

    msg = xyz_array_to_point_cloud_msg(
        points, frame=parent_frame, timestamp=None, rgb_values=colors
    )
    # if colors is None:
    #     msg = xyz_array_to_point_cloud_msg(points, frame=parent_frame, timestamp=None)
    # else:
    #     msg = xyz_array_to_point_cloud_msg(points, frame=parent_frame, timestamp=None, rgb_values=colors*255)

    return msg


def rotation(axis, angle):
    """Returns 3D rotation matrix w.r.t input axis for input angle

    Args:
        axis:
          A string representing axis of rotation. One of "x", "y", "z".
          If None or invalid, assume "z."
        angle:
          A float representing desired angle of rotation in degrees

    Returns:
        A NumPy 3x3 3D rotation matrix (SO(3)) w.r.t. input axis for
        input angle
    """
    angle = angle * np.pi / 180

    if axis == "x" or axis == "X":
        m = np.array(
            [
                [1, 0, 0],
                [0, np.cos(angle), -np.sin(angle)],
                [0, np.sin(angle), np.cos(angle)],
            ]
        )
    elif axis == "y" or axis == "Y":
        m = np.array(
            [
                [np.cos(angle), 0, np.sin(angle)],
                [0, 1, 0],
                [-np.sin(angle), 0, np.cos(angle)],
            ]
        )
    else:  # Around z axis
        m = np.array(
            [
                [np.cos(angle), -np.sin(angle), 0],
                [np.sin(angle), np.cos(angle), 0],
                [0, 0, 1],
            ]
        )

    return m