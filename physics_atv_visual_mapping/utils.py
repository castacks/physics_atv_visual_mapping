import torch
import numpy as np
from numpy import pi as PI

np.float = np.float64  # hack for numpify
import ros2_numpy

from scipy.spatial.transform import Rotation

"""
collection of common utility fns
"""
def load_ontology(ontology):
    """
    convert ontology from human-readable yaml
    """
    sorted_keys = sorted([int(k) for k in ontology.keys()])

    res = {}
    res['ids'] = sorted_keys
    res['labels'] = [ontology[k]['id'] for k in sorted_keys]
    res['prompts'] = [ontology[k]['prompt'] for k in sorted_keys]
    res['palette'] = torch.tensor([ontology[k]['color'] for k in sorted_keys])

    return res

def random_palette(n):
    """
    Make a random palette with n colors

    Implement Fibonacci lattice to get clean coverage w/o obvious regularity
    https://extremelearning.com.au/how-to-evenly-distribute-points-on-a-sphere-more-effectively-than-the-canonical-fibonacci-lattice/
    """
    gr = (1 + 5**0.5)/2
    i = torch.arange(n)
    theta = 2 * PI * i / gr
    phi = torch.arccos(1 - 2*(i)/n)

    pts = torch.stack([
        theta.cos() * phi.sin(),
        theta.sin() * phi.sin(),
        phi.cos()
    ], dim=-1)

    return (pts + 1) / 2.

def apply_palette(x, palette, softmax=False):
    """
    Apply semantic segmentation palette to data x
    Args:
        x: [b1...bn x S] tensor
        palette: [Sx3] tensor
    Returns:
        [b1...bn x 3] viz tensor
    """
    bdims = [1] * (len(x.shape)-1)

    _x = x.unsqueeze(-1)

    if softmax:
        _x = _x.softmax(dim=-2)

    _cs = palette.float().to(_x.device).reshape(*bdims, -1, 3)

    return (_x * _cs).sum(dim=-2)

DEG_2_RAD = PI/180.
RAD_2_DEG = 180./PI

def normalize_dino(img, return_min_max=False):
    if img.numel() == 0:
        return img[..., :3]

    _img = img[..., :3]
    _ndims = len(img.shape) - 1
    _dims = [1] * _ndims + [3]
    vmin = _img.reshape(-1, 3).min(dim=0)[0].view(*_dims)
    vmax = _img.reshape(-1, 3).max(dim=0)[0].view(*_dims)
    if return_min_max:
        return (_img - vmin) / (vmax - vmin), (vmin, vmax)
    else:
        return (_img - vmin) / (vmax - vmin)


def tf_msg_to_htm(tf_msg):
    p = np.array(
        [
            tf_msg.transform.translation.x,
            tf_msg.transform.translation.y,
            tf_msg.transform.translation.z,
        ]
    )

    q = np.array(
        [
            tf_msg.transform.rotation.x,
            tf_msg.transform.rotation.y,
            tf_msg.transform.rotation.z,
            tf_msg.transform.rotation.w,
        ]
    )

    R = Rotation.from_quat(q).as_matrix()

    htm = np.eye(4)
    htm[:3, :3] = R
    htm[:3, -1] = p

    return torch.from_numpy(htm).float()


def pcl_msg_to_xyz(pcl_msg):
    pcl_np = ros2_numpy.numpify(pcl_msg)
    xyz = np.stack(
        [pcl_np["x"].flatten(), pcl_np["y"].flatten(), pcl_np["z"].flatten()], axis=-1
    )

    return torch.from_numpy(xyz).float()


def pcl_msg_to_xyzrgb(pcl_msg):
    pcl_np = ros2_numpy.numpify(pcl_msg)
    xyz = np.stack([pcl_np["x"], pcl_np["y"], pcl_np["z"]], axis=-1)

    colors_raw = pcl_np["rgb"]
    red = (colors_raw & 0x00FF0000) >> 16
    green = (colors_raw & 0x0000FF00) >> 8
    blue = (colors_raw & 0x000000FF) >> 0
    colors = np.stack([red, green, blue], axis=-1) / 255.0

    return torch.from_numpy(np.concatenate([xyz, colors], axis=-1)).float()


def pose_to_htm(pose):
    p = pose[:3]
    q = pose[3:7]

    R = Rotation.from_quat(q).as_matrix()

    htm = np.eye(4)
    htm[:3, :3] = R
    htm[:3, -1] = p

    return torch.from_numpy(htm).float()


def transform_points(points, htm):
    """ """
    pt_pos = points[:, :3]
    pt_pos = torch.cat([pt_pos, torch.ones_like(pt_pos[:, [0]])], dim=-1)
    pt_tf_pos = htm.view(1, 4, 4) @ pt_pos.view(-1, 4, 1)
    points[:, :3] = pt_tf_pos[:, :3, 0]
    return points
