#raycasting functions
import torch
import torch_scatter

from numpy import pi as PI

from physics_atv_visual_mapping.utils import transform_points, DEG_2_RAD, RAD_2_DEG

class FrustumRaytracer:
    """
    Implementation of raytracer that leverages frustum binning to clear out voxels
    High-level algo:
        1. Take in current pose, measurement voxels and aggregated voxels
        2. For every voxel in measurement voxels
            a. compute its spherical coordinates w.r.t. pose
            b. store maximum range for a set of spherical bins
        3. For every voxel in aggregated voxels
            a. compute its spherical coordinates w.r.t. pose
            b. if range less than the range in the corresponding bin, increment miss count
    """
    def __init__(self, config, device='cpu'):
        self.sensor_model = setup_sensor_model(config["sensor"], device=device)
        self.device = device

    def raytrace(self, pose, voxel_grid_meas, voxel_grid_agg):
        """
        Actual raytracing interface

        TODO: we're raycasting in global but the sensor is in local. Rotate the bins into local using pose
        """
        voxel_pts = voxel_grid_meas.grid_indices_to_pts(voxel_grid_meas.raster_indices_to_grid_indices(voxel_grid_meas.raster_indices))
        voxel_el_az_range = get_el_az_range_from_xyz(pose, voxel_pts)
        voxel_maxdist_el_az_bins = bin_el_az_range(voxel_el_az_range, sensor_model=self.sensor_model, reduce='max')

        voxel_from_el_az = get_xyz_from_el_az_range(pose, voxel_el_az_range)

        el_az = torch.stack(torch.meshgrid(self.sensor_model["el_bins"][:-1], self.sensor_model["az_bins"][:-1], indexing='ij'), dim=-1)
        voxel_maxdist_sph = torch.cat([el_az.view(-1, 2), voxel_maxdist_el_az_bins.view(-1, 1)], dim=-1)
        voxel_maxdist_sph = voxel_maxdist_sph[voxel_maxdist_sph[:, 2] > 1e-6]
        voxel_maxdist_xyz = get_xyz_from_el_az_range(pose, voxel_maxdist_sph)

        agg_voxel_pts = voxel_grid_agg.grid_indices_to_pts(voxel_grid_agg.raster_indices_to_grid_indices(voxel_grid_agg.raster_indices))
        agg_voxel_el_az_range = get_el_az_range_from_xyz(pose, agg_voxel_pts)
        agg_voxel_bin_idxs = get_el_az_range_bin_idxs(agg_voxel_el_az_range, sensor_model=self.sensor_model)

        #bin idx == -1 iff. outside sensor fov
        agg_voxel_valid_bin = (agg_voxel_bin_idxs >= 0)

        #set to large negative to not filter on misses
        voxel_maxdist_el_az_bins[voxel_maxdist_el_az_bins < 1e-6] = -1e10

        #set to lidar range to filter on misses
        # voxel_maxdist_el_az_bins[voxel_maxdist_el_az_bins < 1e-6] = 200.

        agg_ranges = agg_voxel_el_az_range[:, 2]
        query_ranges = voxel_maxdist_el_az_bins[agg_voxel_bin_idxs]
        passthrough_mask = (query_ranges > agg_ranges) & agg_voxel_valid_bin

        #dont increment hits, do that in the aggregate step
        voxel_grid_agg.misses += passthrough_mask.float()

        # import matplotlib.pyplot as plt
        # n_el = self.sensor_model['el_bins'].shape[0] - 1
        # n_az = self.sensor_model['az_bins'].shape[0] - 1
        # plt.imshow(voxel_maxdist_el_az_bins.reshape(n_el, n_az).cpu().numpy(), vmin=0., cmap='jet', origin='lower')
        # plt.show()

        # import open3d as o3d
        # pc_passthrough = o3d.geometry.PointCloud()
        # pc_passthrough.points = o3d.utility.Vector3dVector(agg_voxel_pts[passthrough_mask].cpu().numpy())
        
        # pc_hits = o3d.geometry.PointCloud()
        # pc_hits.points = o3d.utility.Vector3dVector(voxel_maxdist_xyz.cpu().numpy())
        # pc_hits.paint_uniform_color([1., 0., 0.])

        # origin = o3d.geometry.TriangleMesh.create_coordinate_frame(origin=pos.cpu().numpy())
        # o3d.visualization.draw_geometries([pc_passthrough, pc_hits, origin])

        return

    def to(self, device):
        self.device = self.device
        for k,v in self.sensor_model.items():
            self.sensor_model[k] = v.to(device)
        return self

def setup_sensor_model(sensor_config, device='cpu'):
    if sensor_config["type"] == "generic":
        assert sensor_config["el_range"][0] >= -180. and sensor_config["el_range"][1] <= 180., "expect el_range in [-180., 180.]"
        assert sensor_config["az_range"][0] >= -180. and sensor_config["az_range"][1] <= 180., "expect az_range in [-180., 180.]"

        el_bins = DEG_2_RAD * torch.linspace(*sensor_config["el_range"], sensor_config["n_el"]+1, device=device)
        az_bins = DEG_2_RAD * torch.linspace(*sensor_config["az_range"], sensor_config["n_az"]+1, device=device)

        if sensor_config["el_thresh"] == "default":
            el_thresh = (el_bins[1:] - el_bins[:-1]).min()
        else:
            el_thresh = torch.tensor(sensor_config["el_thresh"], dtype=torch.float, device=device)

        if sensor_config["az_thresh"] == "default":
            az_thresh = (az_bins[1:] - az_bins[:-1]).min()
        else:
            az_thresh = torch.tensor(sensor_config["az_thresh"], dtype=torch.float, device=device)

        return {
            "el_bins": el_bins,
            "el_thresh": el_thresh,
            "az_bins": az_bins,
            "az_thresh": az_thresh,
        }

    elif sensor_config["type"] == "VLP32C":
        #from the spec sheet @ 600RPM (https://icave2.cse.buffalo.edu/resources/sensor-modeling/VLP32CManual.pdf)
        az_bins = DEG_2_RAD * torch.linspace(-180., 180., 1801, dtype=torch.float, device=device)
        az_thresh = (az_bins[1:] - az_bins[:-1]).min()

        # EL_THRESH = 0.333 #min elev. diff bet. beams
        EL_THRESH = 0.5 #min elev. diff bet. beams (+ some slop)

        #implement elevation from spec sheet. subtract off half of thresh to get lower bin edges (and copy top bin edge)
        el_bins = DEG_2_RAD * (torch.tensor([
         -25.000, -15.6390, -11.3100,  -8.8430,  -7.2540,  -6.1480,  -5.3330,
         -4.6670,  -4.0000,  -3.6670,  -3.3330,  -3.0000,  -2.6670,  -2.3330,
         -2.0000,  -1.6670,  -1.0000,  -0.6670,  -0.3330,   0.0000,   0.3330,
          0.6670,   1.0000,   1.3330,   1.6670,   2.3330,   3.0000,   3.3330,
          4.6670,   7.0000,  10.3330,  15.0000, 15.+EL_THRESH], dtype=torch.float, device=device) - 0.5*EL_THRESH)

        el_thresh = DEG_2_RAD * torch.tensor(EL_THRESH, dtype=torch.float, device=device)

        return {
            "el_bins": el_bins,
            "el_thresh": el_thresh,
            "az_bins": az_bins,
            "az_thresh": az_thresh,
        }
    else:
        print("unsupported sensor model type {}".format(sensor_config["type"]))
        exit(1)

    #debug viz code
    """
    #debug spherical projection/binning
    import open3d as o3d
    pc = o3d.geometry.PointCloud()
    pc.points = o3d.utility.Vector3dVector(voxel_maxdist_xyz.cpu().numpy())
    pc.paint_uniform_color([1., 0., 0.])

    voxel_pc = o3d.geometry.PointCloud()
    voxel_pc.points = o3d.utility.Vector3dVector(voxel_pts.cpu().numpy())

    origin = o3d.geometry.TriangleMesh.create_coordinate_frame(origin=pos.cpu().numpy())
    o3d.visualization.draw_geometries([pc, voxel_pc, origin])

    #debug
    import matplotlib.pyplot as plt
    plt.imshow(voxel_maxdist_el_az_bins.reshape(n_el, n_az).cpu().numpy(), vmin=0., cmap='jet', origin='lower')
    plt.show()

    import open3d as o3d
    pc_passthrough = o3d.geometry.PointCloud()
    pc_passthrough.points = o3d.utility.Vector3dVector(agg_voxel_pts[passthrough_mask].cpu().numpy())
    
    pc_hits = o3d.geometry.PointCloud()
    pc_hits.points = o3d.utility.Vector3dVector(voxel_maxdist_xyz.cpu().numpy())
    pc_hits.paint_uniform_color([1., 0., 0.])

    origin = o3d.geometry.TriangleMesh.create_coordinate_frame(origin=pos.cpu().numpy())
    o3d.visualization.draw_geometries([pc_passthrough, pc_hits, origin])
    """

def htm_from_quat(q):
    """
    Args:
        quaternion as [x,y,z,w]

    the math (https://stackoverflow.com/questions/1556260/convert-quaternion-rotation-to-rotation-matrix):
    1.0f - 2.0f*qy*qy - 2.0f*qz*qz, 2.0f*qx*qy - 2.0f*qz*qw, 2.0f*qx*qz + 2.0f*qy*qw, 0.0f,
    2.0f*qx*qy + 2.0f*qz*qw, 1.0f - 2.0f*qx*qx - 2.0f*qz*qz, 2.0f*qy*qz - 2.0f*qx*qw, 0.0f,
    2.0f*qx*qz - 2.0f*qy*qw, 2.0f*qy*qz + 2.0f*qx*qw, 1.0f - 2.0f*qx*qx - 2.0f*qy*qy, 0.0f,
    0.0f, 0.0f, 0.0f, 1.0f);
    """
    qdims = q.shape[:-1]
    qx, qy, qz, qw = q.moveaxis(-1, 0)

    return torch.stack([
        1. - 2.*qy*qy - 2.*qz*qz, 2.*qx*qy - 2.*qz*qw, 2.*qx*qz + 2.*qy*qw, torch.zeros_like(qx),
        2.*qx*qy + 2.*qz*qw, 1. - 2.*qx*qx - 2.*qz*qz, 2.*qy*qz - 2.*qx*qw, torch.zeros_like(qx),
        2.*qx*qz - 2.*qy*qw, 2.*qy*qz + 2.*qx*qw, 1. - 2.*qx*qx - 2.*qy*qy, torch.zeros_like(qx),
        torch.zeros_like(qx), torch.zeros_like(qx), torch.zeros_like(qx), torch.ones_like(qx)
    ], dim=-1).view(*qdims, 4, 4)

def get_el_az_range_from_xyz(pose, pts, apply_rotation=True):
    """
    Compute elevation, azimuth and range to a given position for all points

    Args:
        pos: [p;q] Tensor of the local pose of the robot
        pts: [Nx3] Tensor of points to compute (expected to be in same frame as pos base frame)
        apply_rotation: If true, compute el/az relative to the orientation in pose, else global
    """
    pts_to_pos_dx = pts - pose[:3].view(-1, 3)

    if apply_rotation:
        R = htm_from_quat(pose[3:7])
        #pts are in global, so apply inverse of R to transform to local
        pts_to_pos_dx = transform_points(pts_to_pos_dx, torch.linalg.inv(R))

    ranges = torch.linalg.norm(pts_to_pos_dx, dim=-1)
    ranges_2d = torch.linalg.norm(pts_to_pos_dx[..., :2], dim=-1)
    az = torch.atan2(pts_to_pos_dx[..., 1], pts_to_pos_dx[..., 0])
    el = torch.atan2(pts_to_pos_dx[..., 2], ranges_2d)

    return torch.stack([el, az, ranges], dim=-1)

def get_xyz_from_el_az_range(pose, el_az_range, apply_rotation=True):
    x = el_az_range[:, 2] * el_az_range[:, 1].cos() * el_az_range[:, 0].cos()
    y = el_az_range[:, 2] * el_az_range[:, 1].sin() * el_az_range[:, 0].cos()
    z = el_az_range[:, 2] * el_az_range[:, 0].sin()

    pts = torch.stack([x, y, z], dim=-1)

    if apply_rotation:
        #assumed that pts in local, so rotate by pose to get global
        R = htm_from_quat(pose[3:7])
        pts = transform_points(pts, R)

    return pts + pose[:3].view(1, 3)

def bin_el_az_range(el_az_range, sensor_model, reduce='max'):
    """
    bin elevation and azimuth in to discrete bins and take the 'reduce' of data for each bin
    """
    #binedges are inclusive
    n_az = sensor_model["az_bins"].shape[0] - 1
    n_el = sensor_model["el_bins"].shape[0] - 1
    raster_idxs = get_el_az_range_bin_idxs(el_az_range, sensor_model)

    valid_mask = (raster_idxs > 0)

    #placeholder of zero should be ok?
    out = torch_scatter.scatter(src=el_az_range[..., 2][valid_mask], index=raster_idxs[valid_mask], dim_size=n_el*n_az, reduce=reduce)

    return out

def get_el_az_range_bin_idxs(el_az_range, sensor_model):
    """
    assume that angles in +-180
    """
    #binedges are inclusive
    n_az = sensor_model["az_bins"].shape[0] - 1
    n_el = sensor_model["el_bins"].shape[0] - 1

    #subtracting 1 to get the idx of the lower edge
    el_idxs = torch.bucketize(el_az_range[:, 0], sensor_model["el_bins"]) - 1
    az_idxs = torch.bucketize(el_az_range[:, 1], sensor_model["az_bins"]) - 1

    raster_idxs = el_idxs * n_az + az_idxs

    #compute and evaluate residual, as elems can fall outside bin edges
    el_res = (el_az_range[:, 0] - sensor_model["el_bins"][el_idxs])
    az_res = (el_az_range[:, 1] - sensor_model["az_bins"][az_idxs])

    el_invalid = (el_res < 0.) | (el_res > sensor_model["el_thresh"])
    az_invalid = (az_res < 0.) | (az_res > sensor_model["az_thresh"])

    #also filter out idxs at binedges
    el_oob = (el_idxs == -1) | (el_idxs == n_el)
    az_oob = (az_idxs == -1) | (az_idxs == n_az)

    invalid_mask = el_invalid | el_oob | az_invalid | az_oob

    #set invalid pts to -1 raster
    raster_idxs[invalid_mask] = -1

    return raster_idxs