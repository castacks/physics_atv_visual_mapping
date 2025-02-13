#raycasting functions
import torch
import torch_scatter

from numpy import pi as PI

def get_el_az_range_from_xyz(pos, pts):
    """
    Compute elevation, azimuth and range to a given position for all points
    """
    pts_to_pos_dx = pts - pos.view(-1, 3)

    ranges = torch.linalg.norm(pts_to_pos_dx, dim=-1)
    ranges_2d = torch.linalg.norm(pts_to_pos_dx[..., :2], dim=-1)
    az = torch.atan2(pts_to_pos_dx[..., 1], pts_to_pos_dx[..., 0]) % (2*PI)
    el = torch.atan2(pts_to_pos_dx[..., 2], ranges_2d) % (2*PI)

    return torch.stack([el, az, ranges], dim=-1)

def get_xyz_from_el_az_range(pos, el_az_range):
    x = el_az_range[:, 2] * el_az_range[:, 1].cos() * el_az_range[:, 0].cos()
    y = el_az_range[:, 2] * el_az_range[:, 1].sin() * el_az_range[:, 0].cos()
    z = el_az_range[:, 2] * el_az_range[:, 0].sin()

    return torch.stack([x, y, z], dim=-1) + pos.view(1, 3)

def bin_el_az_range(el_az_range, n_el, n_az, reduce='max'):
    """
    bin elevation and azimuth in to discrete bins and take the 'reduce' of data for each bin
    """
    raster_idxs = get_el_az_range_bin_idxs(el_az_range, n_el, n_az)

    #placeholder of zero should be ok?
    out = torch_scatter.scatter(src=el_az_range[..., 2], index=raster_idxs, dim_size=n_el*n_az, reduce=reduce)

    return out

def get_el_az_range_bin_idxs(el_az_range, n_el, n_az):
    el_disc = 2*PI / n_el
    az_disc = 2*PI / n_az

    el_idxs = (el_az_range[..., 0] / el_disc).long()
    az_idxs = (el_az_range[..., 1] / az_disc).long()

    raster_idxs = el_idxs * n_az + az_idxs

    return raster_idxs