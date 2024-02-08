import torch
import numpy as np
np.float = np.float64 #hack for numpify
import ros_numpy

from scipy.spatial.transform import Rotation

"""
collection of common geometry operations on poses, points, etc
"""

def tf_msg_to_htm(tf_msg):
    p = np.array([
        tf_msg.transform.translation.x,
        tf_msg.transform.translation.y,
        tf_msg.transform.translation.z,
    ])

    q = np.array([
        tf_msg.transform.rotation.x,
        tf_msg.transform.rotation.y,
        tf_msg.transform.rotation.z,
        tf_msg.transform.rotation.w
    ])

    R = Rotation.from_quat(q).as_matrix()

    htm = np.eye(4)
    htm[:3, :3] = R
    htm[:3, -1] = p

    return torch.from_numpy(htm).float()

def pcl_msg_to_xyzrgb(pcl_msg):
    pcl_np = ros_numpy.numpify(pcl_msg)
    xyz = np.stack([
        pcl_np['x'],
        pcl_np['y'],
        pcl_np['z']
    ], axis=-1)

    colors_raw = pcl_np['rgb']
    red = ((colors_raw & 0x00FF0000)>>16)
    green = ((colors_raw & 0x0000FF00)>>8)
    blue = ((colors_raw & 0x000000FF)>>0)
    colors = np.stack([red, green, blue], axis=-1)/255.

    return torch.from_numpy(np.concatenate([
        xyz,
        colors
    ], axis=-1)).float()


def transform_points(points, htm):
    """
    """
    pt_pos = points[:, :3]
    pt_pos = torch.cat([
        pt_pos,
        torch.ones_like(pt_pos[:, [0]])
    ], dim=-1)
    pt_tf_pos = htm.view(1,4,4) @ pt_pos.view(-1,4,1)
    points[:, :3] = pt_tf_pos[:, :3, 0]
    return points
