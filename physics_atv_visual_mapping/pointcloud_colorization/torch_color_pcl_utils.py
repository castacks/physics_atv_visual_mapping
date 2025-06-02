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
    if not isinstance(intrinsics_matrix, torch.Tensor):
        return get_intrinsics(torch.tensor(intrinsics_matrix).float(), tf_in_optical)

    if len(intrinsics_matrix.shape) == 1:
        return get_intrinsics(intrinsics_matrix.reshape(3, 3), tf_in_optical)

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


def get_projection_matrix(intrinsics, extrinsics):
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
    P = P[..., :-1, :]

    return P

def get_pixel_projection(points, P, images):
    """Returns projection information for a set of points
        onto a set of images

    Args:
        points: [Nx3] FloatTensor of points
        P: [B x 3 x 4] Projection matrix for each image
        images [B x W x H x C] FloatTensor of images

    Returns:
        coords: [B x N x 2] FloatTensor of pixel coords for each image
        valid_mask: [B x N] BoolTensor containing True if the N-th pt is visible in the B-th image
    """
    iw = images.shape[2]
    ih = images.shape[1]

    ones = torch.ones_like(points[:, [0]])
    points_hm = torch.cat([points, ones], dim=-1)

    #[B x N x 3]
    hm_px = (P.view(-1, 1, 3, 4) @ points_hm.view(1, -1, 4, 1)).squeeze(-1)
    hm_norm = hm_px / hm_px[..., [2]]

    coords = hm_norm[..., :-1]

    ## Make sure pixels are within image frame
    cond1 = coords[..., 0] >= 0.
    cond2 = coords[..., 0] < iw
    cond3 = coords[..., 1] >= 0.
    cond4 = coords[..., 1] < ih
    ## Make sure lidar points are in front of camera
    cond5 = hm_px[..., 2] > 0.

    valid_mask = cond1 & cond2 & cond3 & cond4 & cond5

    return coords, valid_mask

def colorize(pixel_coordinates, valid_mask, images, bilinear_interpolation=True, reduce=True):
    """
    get a set of features/colors for a set of pixel coordinats/images

    Args:
        coords: [B x N x 2] FloatTensor of pixel coords for each image
        valid_mask: [B x N] BoolTensor containing True if the N-th pt is visible in the B-th image
        images [B x W x H x C] FloatTensor of images
        bilinear_interpolation: If true, get feats w/ bilinear interpolation else truncate
        reduce: change the output

    Returns:
        if reduce=True:
            features: [N x C] FloatTensor of features for each pixel.
                If a pixel is in multiple images, we will average
                If a pixel is in no images, we will pad with zeros
            cnt: [N] LongTensor containing the amount of images the n-th pixel was in
        if reduce=False:
            features: [B x N x C] Float tensor containing the feature of the B-th image on the N-th coordinate
            cnt: same as valid_mask
    """
    ni, ih, iw, ic = images.shape

    cnt = valid_mask.sum(dim=0)

    coords = pixel_coordinates.clone()
    coords[~valid_mask] = 0

    if bilinear_interpolation:
        rem = torch.frac(coords)
        offset = torch.tensor([
            [0, 0],
            [1, 0],
            [0, 1],
            [1, 1]
        ], device=images.device).view(4, 1, 1, 2)

        #[4 x B x N x 2]
        idxs = torch.tile(coords.view(1, ni, -1, 2), (4, 1, 1, 1)) + offset
        #equivalent to same-padding
        ixs = idxs[..., 1].long().clip(0, ih-1)
        iys = idxs[..., 0].long().clip(0, iw-1)

        weights = torch.stack([
            (1.-rem[..., 0]) * (1.-rem[..., 1]),
            rem[..., 0] * (1.-rem[..., 1]),
            (1. - rem[..., 0]) * rem[..., 1],
            rem[..., 0] * rem[..., 1]
        ], dim=0)

        ibs = torch.arange(ni).view(1, ni, 1).tile(4, 1, ixs.shape[-1])

        features = images[ibs, ixs, iys]
        interp_features = (weights.view(4, ni, -1, 1) * features).sum(dim=0)

        interp_features[~valid_mask] = 0.

        if reduce:
            interp_features = interp_features.sum(dim=0) / (cnt + 1e-6).view(-1, 1)
            return interp_features, cnt
        else:
            return interp_features, valid_mask

    else:
        ixs = coords[..., 1].long()
        iys = coords[..., 0].long()
        ibs = torch.arange(ni).view(ni, 1).tile(1, ixs.shape[-1])

        features = images[ibs, ixs, iys]
        features[~valid_mask] = 0.

        if reduce:
            features = features.sum(dim=0) / (cnt + 1e-6).view(-1, 1)
            return features, cnt
        else:
            return features, valid_mask

def bilinear_interpolation(pixel_coordinates, image):
    """
    Perform bilinear interpolation at pixel coordinates in image
    Args:
        pixel_coordinates: [N x 2] FloatTensor of pixel coordinates
        image: [W x H x C] FloatTensor of image data
    """
    rem = torch.frac(pixel_coordinates) #ok to just do this bc no negative idxs
    offset = torch.tensor([
        [0, 0],
        [1, 0],
        [0, 1],
        [1, 1]
    ], device=image.device).view(4, 1, 2)

    idxs = torch.tile(pixel_coordinates.view(1, -1, 2), (4, 1, 1)).long() + offset

    #equivalent to same-padding
    idxs[..., 0] = idxs[..., 0].clip(0, image.shape[0]-1)
    idxs[..., 1] = idxs[..., 1].clip(0, image.shape[1]-1)

    weights = torch.stack([
        (1.-rem[:, 0]) * (1.-rem[:, 1]),
        rem[:, 0] * (1.-rem[:, 1]),
        (1. - rem[:, 0]) * rem[:, 1],
        rem[:, 0] * rem[:, 1]
    ], dim=0)

    feats = image[idxs[..., 0], idxs[..., 1]]

    interp_feats = (weights.view(4, -1, 1) * feats).sum(dim=0)
    return interp_feats