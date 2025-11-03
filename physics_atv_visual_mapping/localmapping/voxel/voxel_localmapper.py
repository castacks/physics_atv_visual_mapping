import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import open3d as o3d
import torch
import torch_scatter

from ros_torch_converter.datatypes.pointcloud import FeaturePointCloudTorch
from physics_atv_visual_mapping.localmapping.base import LocalMapper
from physics_atv_visual_mapping.localmapping.metadata import LocalMapperMetadata
from physics_atv_visual_mapping.feature_key_list import FeatureKeyList
from physics_atv_visual_mapping.utils import *

def hist(var):
    import matplotlib.pyplot as plt
    import numpy as np

    var_np = var.cpu().numpy()
    ncols = var_np.shape[1] if var_np.ndim > 1 else 1

    if ncols == 1:
        plt.figure(figsize=(4, 4))
        plt.hist(var_np, bins=100)
    elif ncols == 2:
        plt.figure(figsize=(8, 4))
        for i in range(2):
            plt.subplot(1, 2, i + 1)
            plt.hist(var_np[:, i], bins=100)
    else:
        plt.figure(figsize=(12, 4))
        for i in range(min(3, ncols)):
            plt.subplot(1, 3, i + 1)
            plt.hist(var_np[:, i], bins=100)

    plt.tight_layout()
    plt.show()

def visualize_feature_pc_with_covariances(
    feature_pc: FeaturePointCloudTorch = None,
    vgt=None, voxel_covs=None,
    vgt2=None, voxel_covs2=None,
    vgt3=None, voxel_covs3=None,
    every_nth_point=1,
    cov_scale_factor=1.0
):
    """
    Visualize a FeaturePointCloudTorch with points and their covariance ellipsoids.

    Args:
        feature_pc: Optional FeaturePointCloudTorch containing feature points, colors Nx3, and Nx9 covariances
        vgt: Optional voxel grid transform for voxel ellipsoids
        voxel_covs: Optional voxel covariance matrices
        every_nth_point: show covariance ellipsoids only for every N points
        cov_scale_factor: scaling factor for ellipsoids
    """
    geometries = []

    # --- visualize point cloud and ellipsoids if given ---
    if feature_pc is not None:
        pts = feature_pc.feature_pts.cpu().numpy()
        # pts[:, 1] += 20  # shift y-coordinate (2nd column) by 20
        colors = feature_pc.features[:, :3]
        covs = feature_pc.features[:, 3:12].cpu().numpy().reshape(-1, 3, 3)

        # Create point cloud
        pc = o3d.geometry.PointCloud()
        pc.points = o3d.utility.Vector3dVector(pts)
        pc.colors = o3d.utility.Vector3dVector( normalize_dino(colors).cpu().numpy())
        # pc.colors = o3d.utility.Vector3dVector(colors / 255.0)
        geometries.append(pc)

        # Covariance ellipsoids for point cloud
        covs_torch = torch.from_numpy(covs).float()
        eigvals, eigvecs = torch.linalg.eigh(covs_torch)
        radii_all = torch.sqrt(torch.abs(eigvals))

        eigvals = eigvals.numpy()
        eigvecs = eigvecs.numpy()
        radii_all = radii_all.numpy()

        all_radii = radii_all.reshape(-1)
        mean_radius = all_radii.mean()
        std_radius = all_radii.std()
        max_radius = mean_radius + 1 * std_radius

        ellipsoids = []
        for i, (p, cov, radii, eigvec) in enumerate(zip(pts, covs, radii_all, eigvecs)):
            if i % every_nth_point != 0:
                continue
            if max_radius < radii.mean():
                continue
            ellipsoid = o3d.geometry.TriangleMesh.create_sphere(radius=1.0)
            ellipsoid.scale(cov_scale_factor, center=(0, 0, 0))
            verts = np.asarray(ellipsoid.vertices) * radii
            ellipsoid.vertices = o3d.utility.Vector3dVector(verts)
            ellipsoid.rotate(eigvec, center=(0, 0, 0))
            ellipsoid.translate(p)
            ellipsoid.paint_uniform_color([1, 0, 0])
            ellipsoids.append(ellipsoid)

        # geometries += ellipsoids

    # --- voxel ellipsoids if provided ---
    voxel_ellipsoids = []
    for idx, (voxel_covs_i, vgt_i, color) in enumerate([
        (voxel_covs,  vgt,  [0, 0, 1]),  # blue for num 1
        (voxel_covs2, vgt2, [0, 1, 0]),  # green for num 2
        (voxel_covs3, vgt3, [0, 1, 1])   # red for num 3
    ]):
        if voxel_covs_i is None or vgt_i is None:
            continue

        # every_nth_point = 100
        voxel_covariances = voxel_covs_i.cpu().numpy()
        voxel_centers = np.asarray(vgt_i.points)

        eigvals_v, eigvecs_v = torch.linalg.eigh(torch.from_numpy(voxel_covariances).float())
        radii_v = torch.sqrt(torch.abs(eigvals_v)).numpy()
        eigvecs_v = eigvecs_v.numpy()

        radii_all = torch.sqrt(torch.abs(eigvals_v))
        all_radii = radii_all.reshape(-1)
        mean_radius = all_radii.mean()
        std_radius = all_radii.std()
        max_radius = mean_radius + 2 * std_radius
        min_size = 0.1

        target_count = 500
        num_points = len(voxel_centers)
        every_nth_point = max(1, num_points // target_count)

        # for i, (c, radii, eigvec) in enumerate(zip(voxel_centers[::step], radii_v[::step], eigvecs_v[::step])):
        for i, (c, radii, eigvec) in enumerate(zip(voxel_centers, radii_v, eigvecs_v)):
            if i % every_nth_point != 0:
                continue

            if radii.max() < min_size:
                point_sphere = o3d.geometry.TriangleMesh.create_sphere(radius=min_size * 0.5)
                point_sphere.translate(c)
                point_sphere.paint_uniform_color(color)
                voxel_ellipsoids.append(point_sphere)
                continue

            ellipsoid = o3d.geometry.TriangleMesh.create_sphere(radius=1.0)
            ellipsoid.scale(cov_scale_factor, center=(0, 0, 0))
            verts = np.asarray(ellipsoid.vertices) * radii
            ellipsoid.vertices = o3d.utility.Vector3dVector(verts)
            ellipsoid.rotate(eigvec, center=(0, 0, 0))
            ellipsoid.translate(c)
            ellipsoid.paint_uniform_color(color)
            voxel_ellipsoids.append(ellipsoid)


    grid = create_grid(size=10, step=1.0)
    geometries.append(grid)

    if vgt is not None:
        geometries.append(vgt)

    if voxel_ellipsoids:
        geometries += voxel_ellipsoids

    o3d.visualization.draw_geometries(geometries)


def create_grid(size=10, step=1.0):
    # Build points
    points = []
    lines = []
    idx = 0
    for i in np.arange(-size, size+step, step):
        # vertical line along y
        points.append([i, -size, 0])
        points.append([i, size, 0])
        lines.append([idx, idx+1])
        idx += 2
        # horizontal line along x
        points.append([-size, i, 0])
        points.append([size, i, 0])
        lines.append([idx, idx+1])
        idx += 2

    points = o3d.utility.Vector3dVector(np.array(points))
    lines = o3d.utility.Vector2iVector(np.array(lines))

    line_set = o3d.geometry.LineSet(points=points, lines=lines)
    line_set.colors = o3d.utility.Vector3dVector(np.tile([0.5, 0.5, 0.5], (len(lines), 1)))
    return line_set


class VoxelLocalMapper(LocalMapper):
    """Class for local mapping voxels"""

    def __init__(self, metadata, feature_keys, ema, raytracer=None, n_features=-1, device='cpu'):
        super().__init__(metadata, device)
        assert metadata.ndims == 3, "VoxelLocalMapper requires 3d metadata"
        self.n_features = len(feature_keys) if n_features == -1 else n_features
        self.feature_keys = feature_keys[:self.n_features]

        assert ema >= 0. and ema <= 1.

        assert ema >= 0. and ema <= 1.

        self.voxel_grid = VoxelGrid(self.metadata, self.feature_keys, device)
        self.raytracer = raytracer
        self.do_raytrace = self.raytracer is not None
        self.ema = ema
        self.assert_feat_match = True
        self.assert_feat_match = True

    def update_pose(self, pose: torch.Tensor):
        """
        Args:
            pose: [N] Tensor (we will take the first two elements as the new pose)
        """
        #keep origin a multiple of resolution
        new_origin = torch.round((pose[:3] + self.base_metadata.origin)/self.base_metadata.resolution) * self.base_metadata.resolution
        self.voxel_grid.metadata = self.metadata

        px_shift = torch.round(
            (new_origin - self.metadata.origin) / self.metadata.resolution
        ).long()
        self.voxel_grid.shift(px_shift)
        self.metadata.origin = new_origin

    def add_feature_pc(self, pos: torch.Tensor, feat_pc: FeaturePointCloudTorch, do_raytrace=False, debug=False):
        voxel_grid_new = VoxelGrid.from_feature_pc(feat_pc, self.metadata, self.n_features, pos, strategy='mindist')
        voxel_grid_new = VoxelGrid.from_feature_pc(feat_pc, self.metadata, self.n_features, pos, strategy='mindist')

        if self.assert_feat_match:
            assert self.voxel_grid.feature_keys == voxel_grid_new.feature_keys, f"voxel feat key mismatch: mapper has {self.voxel_grid.feature_keys}, added pc has {voxel_grid_new.feature_keys}"
        if self.assert_feat_match:
            assert self.voxel_grid.feature_keys == voxel_grid_new.feature_keys, f"voxel feat key mismatch: mapper has {self.voxel_grid.feature_keys}, added pc has {voxel_grid_new.feature_keys}"

        if self.do_raytrace:
            # self.raytracer.raytrace(pos, voxel_grid_meas=voxel_grid_new, voxel_grid_agg=self.voxel_grid)
            self.raytracer.raytrace_but_better(pos, pc_meas=feat_pc, voxel_grid_agg=self.voxel_grid)

        #first map all indices with features
        all_raster_idxs = torch.cat([self.voxel_grid.raster_indices, voxel_grid_new.raster_indices])
        unique_raster_idxs, inv_idxs, counts = torch.unique(
            all_raster_idxs, return_inverse=True, return_counts=True, sorted=True
        )

        # we need an index into both the full set of idxs and also the feature buffer
        # note that since we're sorting by raster index, we can align the two buffers
        all_feature_raster_idxs = torch.cat([self.voxel_grid.feature_raster_indices, voxel_grid_new.feature_raster_indices])
        unique_feature_raster_idxs, feat_inv_idxs = torch.unique(all_feature_raster_idxs, return_inverse=True, sorted=True)

        # separate out idxs that are in 1 voxel grid vs both
        vg1_inv_idxs = inv_idxs[: self.voxel_grid.raster_indices.shape[0]] #index from vg1 idxs to aggregated buffer
        vg1_feat_inv_idxs = feat_inv_idxs[:self.voxel_grid.feature_raster_indices.shape[0]] #index from feature idxs into aggregated feature buffer
        vg1_has_feature = self.voxel_grid.feature_mask #this line requires that the voxel grid is sorted by raster idx

        vg2_inv_idxs = inv_idxs[self.voxel_grid.raster_indices.shape[0] :]
        vg2_feat_inv_idxs = feat_inv_idxs[self.voxel_grid.feature_raster_indices.shape[0]:]
        vg2_has_feature = voxel_grid_new.feature_mask #this line requires that the voxel grid is sorted by raster idx
        vg1_feat_buf = torch.zeros(
            unique_feature_raster_idxs.shape[0],
            self.voxel_grid.features.shape[-1],
            device=self.voxel_grid.device,
        )
        vg1_feat_buf_mask = torch.zeros(vg1_feat_buf.shape[0], dtype=torch.bool, device=self.voxel_grid.device)

        vg2_feat_buf = torch.zeros(
            unique_feature_raster_idxs.shape[0],
            voxel_grid_new.features.shape[-1],
            device=self.voxel_grid.device,
        )
        vg2_feat_buf = torch.zeros(
            unique_feature_raster_idxs.shape[0],
            voxel_grid_new.features.shape[-1],
            device=self.voxel_grid.device,
        )
        vg2_feat_buf_mask = vg1_feat_buf_mask.clone()

        #first copy over the original features
        vg1_feat_buf[vg1_feat_inv_idxs] += self.voxel_grid.features
        vg1_feat_buf_mask[vg1_feat_inv_idxs] = True

        vg2_feat_buf[vg2_feat_inv_idxs] += voxel_grid_new.features
        vg2_feat_buf_mask[vg2_feat_inv_idxs] = True

        #apply ema
        feat_buf = self._merge_voxel_features(vg1_feat_buf, vg1_feat_buf_mask, vg2_feat_buf, vg2_feat_buf_mask)

        #ok now i have the merged features and the final raster idxs. need to make the mask
        feature_mask = torch.zeros(unique_raster_idxs.shape[0], dtype=torch.bool, device=self.voxel_grid.device)
        feature_mask[vg1_inv_idxs] = (feature_mask[vg1_inv_idxs] | vg1_has_feature) 
        feature_mask[vg2_inv_idxs] = (feature_mask[vg2_inv_idxs] | vg2_has_feature)

        self.voxel_grid.raster_indices = unique_raster_idxs
        self.voxel_grid.features = feat_buf
        self.voxel_grid.feature_mask = feature_mask

        hit_buf = torch.zeros(
            unique_raster_idxs.shape[0],
            device=self.voxel_grid.device,
        )
        hit_buf[vg1_inv_idxs] += self.voxel_grid.hits
        hit_buf[vg2_inv_idxs] += voxel_grid_new.hits

        miss_buf = torch.zeros(
            unique_raster_idxs.shape[0],
            device=self.voxel_grid.device,
        )
        miss_buf[vg1_inv_idxs] += self.voxel_grid.misses
        miss_buf[vg2_inv_idxs] += voxel_grid_new.misses

        self.voxel_grid.hits = hit_buf
        self.voxel_grid.misses = miss_buf

        min_coords_buf = 1e10 * torch.ones(unique_raster_idxs.shape[0], 3, device=self.voxel_grid.device)
        min_coords_buf[vg1_inv_idxs] = torch.minimum(min_coords_buf[vg1_inv_idxs], self.voxel_grid.min_coords)
        min_coords_buf[vg2_inv_idxs] = torch.minimum(min_coords_buf[vg2_inv_idxs], voxel_grid_new.min_coords)
        self.voxel_grid.min_coords = min_coords_buf

        max_coords_buf = -1e10 * torch.ones(unique_raster_idxs.shape[0], 3, device=self.voxel_grid.device)
        max_coords_buf[vg1_inv_idxs] = torch.maximum(max_coords_buf[vg1_inv_idxs], self.voxel_grid.max_coords)
        max_coords_buf[vg2_inv_idxs] = torch.maximum(max_coords_buf[vg2_inv_idxs], voxel_grid_new.max_coords)
        self.voxel_grid.max_coords = max_coords_buf

        #compute passthrough rate
        passthrough_rate = self.voxel_grid.misses / (self.voxel_grid.hits + self.voxel_grid.misses)

        cull_mask = passthrough_rate > 0.75

        # print('culling {} voxels...'.format(cull_mask.sum()))

        new_grid_idxs = self.voxel_grid.raster_indices_to_grid_indices(self.voxel_grid.raster_indices)
        new_grid_idxs_bev = new_grid_idxs[:, :2]
        new_grid_idxs_z = new_grid_idxs[:, 2]
        new_grid_idxs_bev_raster = new_grid_idxs[:, 0] * self.voxel_grid.metadata.N[1] + new_grid_idxs[:, 1]
        new_minz = torch_scatter.scatter(new_grid_idxs_z, index=new_grid_idxs_bev_raster, reduce="min")
        bottom_voxel_mask = new_grid_idxs_z <= new_minz[new_grid_idxs_bev_raster]

        cull_mask = cull_mask & ~bottom_voxel_mask

        if False:
            import open3d as o3d
            pts = self.voxel_grid.grid_indices_to_pts(self.voxel_grid.raster_indices_to_grid_indices(self.voxel_grid.raster_indices))
            #solid=black, porous=green, cull=red
            colors = torch.stack([torch.zeros_like(passthrough_rate), passthrough_rate, torch.zeros_like(passthrough_rate)], dim=-1)
            colors[cull_mask] = torch.tensor([1., 0., 0.], device=self.device)
            porosity_pc = o3d.geometry.PointCloud()
            porosity_pc.points = o3d.utility.Vector3dVector(pts.cpu().numpy())
            porosity_pc.colors = o3d.utility.Vector3dVector(colors.cpu().numpy())
            o3d.visualization.draw_geometries([porosity_pc])

        self.voxel_grid.hits = self.voxel_grid.hits[~cull_mask]
        self.voxel_grid.misses = self.voxel_grid.misses[~cull_mask]
        self.voxel_grid.min_coords = self.voxel_grid.min_coords[~cull_mask]
        self.voxel_grid.max_coords = self.voxel_grid.max_coords[~cull_mask]
        self.voxel_grid.raster_indices = self.voxel_grid.raster_indices[~cull_mask]

        feat_cull_mask = cull_mask[self.voxel_grid.feature_mask]
        self.voxel_grid.features = self.voxel_grid.features[~feat_cull_mask]
        self.voxel_grid.feature_mask = self.voxel_grid.feature_mask[~cull_mask]

    def _merge_voxel_features(self, vg1_feats, vg1_mask, vg2_feats, vg2_mask):
        """
        Merge voxel grid features together with EMA
        Args:
            vg1_feats: [NxF] Tensor of raster feats from voxel grid 1. Note that this should be raster-padded w/ vg2
                (i.e. this is after the cat+uniq op)
            vg1_mask:  [N] Tensor which is True iff. that raster idx has a feature in vg1
            vg2_feats: Same as vg1_feats for vg2
            vg2_mask:  Same as vg1_mask for vg2
        Returns:
            feats_out: [NxF] Tensor of merged features for all unique raster indices
        """
        feats_out = vg1_feats.clone()
        merge_mask = vg1_mask & vg2_mask
        feats_out[vg1_mask & ~merge_mask] = vg1_feats[vg1_mask & ~merge_mask]
        feats_out[vg2_mask & ~merge_mask] = vg2_feats[vg2_mask & ~merge_mask]
        feats_out[merge_mask] = (1.-self.ema) * vg1_feats[merge_mask] + self.ema * vg2_feats[merge_mask]

        return feats_out

    def _merge_voxel_features(self, vg1_feats, vg1_mask, vg2_feats, vg2_mask):
        """
        Merge voxel grid features together with EMA
        Args:
            vg1_feats: [NxF] Tensor of raster feats from voxel grid 1. Note that this should be raster-padded w/ vg2
                (i.e. this is after the cat+uniq op)
            vg1_mask:  [N] Tensor which is True iff. that raster idx has a feature in vg1
            vg2_feats: Same as vg1_feats for vg2
            vg2_mask:  Same as vg1_mask for vg2
        Returns:
            feats_out: [NxF] Tensor of merged features for all unique raster indices
        """
        feats_out = vg1_feats.clone()
        merge_mask = vg1_mask & vg2_mask
        feats_out[vg1_mask & ~merge_mask] = vg1_feats[vg1_mask & ~merge_mask]
        feats_out[vg2_mask & ~merge_mask] = vg2_feats[vg2_mask & ~merge_mask]
        feats_out[merge_mask] = (1.-self.ema) * vg1_feats[merge_mask] + self.ema * vg2_feats[merge_mask]

        return feats_out

    def to(self, device):
        self.device = device
        self.voxel_grid = self.voxel_grid.to(device)
        self.metadata = self.metadata.to(device)
        if self.raytracer:
            self.raytracer = self.raytracer.to(device)
        return self

class VoxelCoordinateLocalMapper(VoxelLocalMapper):
    """
    Voxel localmapper class that maps image coordinates instead of features directly
    This requires us to implement a slightly different merge rule (list update instead of EMA)
    """
    def __init__(self, metadata, feature_keys, ema, raytracer=None, n_features=-1, max_n_coords=10, device='cpu'):
        """
        Args:
            max_n_coords: the maximum amount of image feats that a voxel can pull from
                note that with 0.5 ema, 7 coords is sufficient for >1% err
        """
        self.input_feature_keys = feature_keys
        self.output_feature_keys = []
        for k in ['seq', 'cam', 'u', 'v', 'w']:
            for i in range(max_n_coords):
                self.output_feature_keys.append(f"{k}_{i}")
        self.output_feature_keys = FeatureKeyList(
            label=self.output_feature_keys,
            metainfo=["img_coords"] * len(self.output_feature_keys)
        )

        super().__init__(metadata, self.output_feature_keys, ema, raytracer, n_features, device)
        self.assert_feat_match = False
        self.max_n_coords = max_n_coords

    def _merge_voxel_features(self, vg1_feats, vg1_mask, vg2_feats, vg2_mask):
        """
        Merge voxel grid features together with EMA
        Args:
            vg1_feats: [NxF] Tensor of raster feats from voxel grid 1. Note that this should be raster-padded w/ vg2
                (i.e. this is after the cat+uniq op)
            vg1_mask:  [N] Tensor which is True iff. that raster idx has a feature in vg1
            vg2_feats: Same as vg1_feats for vg2
            vg2_mask:  Same as vg1_mask for vg2
        Returns:
            feats_out: [NxF] Tensor of merged features for all unique raster indices
        """
        #[N x ncams x 5]
        vg1_coords = vg1_feats.view(vg1_feats.shape[0], 5, self.max_n_coords).permute(0,2,1)

        #[N x ncoords x 5]
        vg2_coords = vg2_feats.view(vg2_feats.shape[0], 5, -1).permute(0,2,1)
        
        #apply ema to weights (note that placeholder weight MUST be 0)
        vg1_coords[:, :, -1] = (vg1_coords[:, :, -1] * (1.-self.ema))
        vg2_coords[:, :, -1] = (vg2_coords[:, :, -1] * self.ema)

        #[N x (ncams+ncoords) x 5]
        cat_coords = torch.cat([vg1_coords, vg2_coords], dim=1)

        #[N x (ncams+ncoords)]
        idxs = cat_coords[:, :, 4].argsort(dim=-1, descending=True)

        idxs = idxs[:, :self.max_n_coords]

        _B = torch.arange(idxs.shape[0], device=self.device).unsqueeze(-1)
        cat_coords = cat_coords[_B, idxs] #[N x ncoords x 5]

        weight_sum = cat_coords[:, :, 4].sum(dim=-1, keepdim=True)
        cat_coords[:, :, 4] = cat_coords[:, :, 4] / weight_sum

        feats_out = cat_coords.permute(0,2,1).reshape(-1, 5*self.max_n_coords)

        return feats_out

class VoxelCovarianceLocalMapper(VoxelLocalMapper):
    """
    Voxel localmapper class uses covariance too!
    """
    def __init__(self, metadata, feature_keys, ema, raytracer=None, n_features=-1, max_n_coords=10, device='cpu'):
        """
        Args:
            
        """
        def create_new_feature_keys(feat_keys):
            new_labels = feat_keys.label + ["center_x", "center_y", "center_z"]
            new_metainfo = feat_keys.metainfo + 3 * ["voxelmapper"]
            return FeatureKeyList(label=new_labels, metainfo=new_metainfo)
                        
        new_feature_keys = create_new_feature_keys(feature_keys)
        n_features = len(new_feature_keys)

        super().__init__(metadata, new_feature_keys, ema, raytracer, n_features, device)
        
    def add_feature_pc(self, pos: torch.Tensor, feat_pc: FeaturePointCloudTorch, do_raytrace=False, debug=False):
        
        voxel_grid_new = VoxelCovarianceGrid.from_feature_pc(feat_pc, self.metadata, self.n_features)

        self.voxel_grid.feature_keys = voxel_grid_new.feature_keys

        assert self.voxel_grid.feature_keys == voxel_grid_new.feature_keys, f"voxel feat key mismatch: mapper has {self.voxel_grid.feature_keys}, added pc has {voxel_grid_new.feature_keys}"

        if self.do_raytrace:
            # self.raytracer.raytrace(pos, voxel_grid_meas=voxel_grid_new, voxel_grid_agg=self.voxel_grid)
            self.raytracer.raytrace_but_better(pos, pc_meas=feat_pc, voxel_grid_agg=self.voxel_grid)

        #first map all indices with features
        all_raster_idxs = torch.cat([self.voxel_grid.raster_indices, voxel_grid_new.raster_indices])
        unique_raster_idxs, inv_idxs, counts = torch.unique(
            all_raster_idxs, return_inverse=True, return_counts=True, sorted=True
        )

        # we need an index into both the full set of idxs and also the feature buffer
        # note that since we're sorting by raster index, we can align the two buffers
        all_feature_raster_idxs = torch.cat([self.voxel_grid.feature_raster_indices, voxel_grid_new.feature_raster_indices])
        unique_feature_raster_idxs, feat_inv_idxs = torch.unique(all_feature_raster_idxs, return_inverse=True, sorted=True)
       
        # separate out idxs that are in 1 voxel grid vs both
        vg1_inv_idxs = inv_idxs[: self.voxel_grid.raster_indices.shape[0]] #index from vg1 idxs to aggregated buffer
        vg1_feat_inv_idxs = feat_inv_idxs[:self.voxel_grid.feature_raster_indices.shape[0]] #index from feature idxs into aggregated feature buffer
        vg1_has_feature = self.voxel_grid.feature_mask #this line requires that the voxel grid is sorted by raster idx

        vg2_inv_idxs = inv_idxs[self.voxel_grid.raster_indices.shape[0] :]
        vg2_feat_inv_idxs = feat_inv_idxs[self.voxel_grid.feature_raster_indices.shape[0]:]
        vg2_has_feature = voxel_grid_new.feature_mask #this line requires that the voxel grid is sorted by raster idx

        vg1_feat_buf = torch.zeros(
            unique_feature_raster_idxs.shape[0],
            self.voxel_grid.features.shape[-1],
            device=self.voxel_grid.device,
        )
        vg1_feat_buf_mask = torch.zeros(vg1_feat_buf.shape[0], dtype=torch.bool, device=self.voxel_grid.device)

        vg2_feat_buf = vg1_feat_buf.clone()
        vg2_feat_buf_mask = vg1_feat_buf_mask.clone()

        feat_buf = vg1_feat_buf.clone()

        #first copy over the original features

        vg1_feat_buf[vg1_feat_inv_idxs] += self.voxel_grid.features
        vg1_feat_buf_mask[vg1_feat_inv_idxs] = True

        vg2_feat_buf[vg2_feat_inv_idxs] += voxel_grid_new.features
        vg2_feat_buf_mask[vg2_feat_inv_idxs] = True

        #Macvo stuff here.

        cov_idxs = self.voxel_grid.feature_keys.index_metainfo("macvo")
        centers_idxs = self.voxel_grid.feature_keys.index_metainfo("voxelmapper")
        feats_idxs = [i for i in range(len(self.voxel_grid.feature_keys)) if i not in set(cov_idxs) | set(centers_idxs)]
        
        cov_mask = torch.zeros(feat_buf.shape[1], dtype=torch.bool, device=feat_buf.device)
        centers_mask = torch.zeros(feat_buf.shape[1], dtype=torch.bool, device=feat_buf.device)
        feats_mask = torch.zeros(feat_buf.shape[1], dtype=torch.bool, device=feat_buf.device)
        cov_mask[cov_idxs] = True
        centers_mask[centers_idxs] = True
        feats_mask[feats_idxs] = True
        
        # extract covariance features for both voxel grids
        vg1_cov = vg1_feat_buf[:, cov_idxs].view(-1, 3, 3)
        vg2_cov = vg2_feat_buf[:, cov_idxs].view(-1, 3, 3)

        # identify overlaps: voxels present in both
        overlap_mask = vg1_feat_buf_mask & vg2_feat_buf_mask
        only_vg1 = vg1_feat_buf_mask & ~vg2_feat_buf_mask
        only_vg2 = ~vg1_feat_buf_mask & vg2_feat_buf_mask
        
        overlap_idxs = torch.nonzero(overlap_mask, as_tuple=True)[0] 

        #Non overlapping features copied over directly
        if only_vg1.any():
            feat_buf[only_vg1, :] = vg1_feat_buf[only_vg1, :]    # non cov features

        if only_vg2.any():
            feat_buf[only_vg2, :] = vg2_feat_buf[only_vg2, :]

        if overlap_mask.any():

            eps = 1e-6
            d = 3

            # Extract only the covariances for overlapping voxels
            vg1_cov_overlap = vg1_cov[overlap_mask]
            vg2_cov_overlap = vg2_cov[overlap_mask]

            # Compute log det Σ for each overlap voxel
            logdet_vg1 = torch.logdet(vg1_cov_overlap + eps * torch.eye(3, device=vg1_cov.device))
            logdet_vg2 = torch.logdet(vg2_cov_overlap + eps * torch.eye(3, device=vg2_cov.device))

            # Normalization constant (Mahalanobis term = 1)
            const_factor = -0.5 * (d * torch.log(torch.tensor(2.0 * torch.pi, device=vg1_cov.device)))
            norm_const_vg1 = torch.exp(const_factor - 0.5 * logdet_vg1)
            norm_const_vg2 = torch.exp(const_factor - 0.5 * logdet_vg2)

            # Normalize weights so they sum to 1
            total_weights = norm_const_vg1 + norm_const_vg2 + eps
            w1 = norm_const_vg1 / total_weights
            w2 = norm_const_vg2 / total_weights

            # Covariance fusion using weighted Covariance Intersection
            vg1_prec = torch.linalg.inv(vg1_cov_overlap + eps * torch.eye(3, device=vg1_cov.device))
            vg2_prec = torch.linalg.inv(vg2_cov_overlap + eps * torch.eye(3, device=vg2_cov.device))

            fused_cov = torch.linalg.inv(0.5 * vg1_prec + 0.5* vg2_prec) #w1.view(-1, 1, 1)
            # fused_cov = 0.5 * vg1_cov_overlap + 0.5* vg2_cov_overlap
            
            feat_buf[overlap_idxs.unsqueeze(1), cov_idxs] = fused_cov.reshape(overlap_mask.sum(), 9)
            feat_buf[overlap_idxs.unsqueeze(1), centers_idxs]= ( 0.5 * vg1_feat_buf[overlap_mask, :][:, centers_mask] + 0.5 * vg2_feat_buf[overlap_mask, :][:, centers_mask] )
            
            # Weighted feature fusion
            feat_buf[overlap_idxs.unsqueeze(1), feats_idxs] = (                
                w1.unsqueeze(-1) * vg1_feat_buf[overlap_mask, :][:, feats_mask] +
                w2.unsqueeze(-1) * vg2_feat_buf[overlap_mask, :][:, feats_mask]            
            )
            
        #ok now i have the merged features and the final raster idxs. need to make the mask
        feature_mask = torch.zeros(unique_raster_idxs.shape[0], dtype=torch.bool, device=self.voxel_grid.device)
        feature_mask[vg1_inv_idxs] = (feature_mask[vg1_inv_idxs] | vg1_has_feature) 
        feature_mask[vg2_inv_idxs] = (feature_mask[vg2_inv_idxs] | vg2_has_feature)


        # Initialize changed flag, points that have been updated will be marked True
        changed_flag = torch.zeros(unique_raster_idxs.shape[0], dtype=torch.bool, device=self.voxel_grid.device)

        # Overlapping voxels: mark those updated by vg2
        changed_flag[vg2_inv_idxs[overlap_mask[vg2_inv_idxs]]] = True

        # New voxels present only in vg2
        changed_flag[vg2_inv_idxs[only_vg2[vg2_inv_idxs]]] = True

        # Assign to voxel grid
        self.voxel_grid.cov_changed = changed_flag
        self.voxel_grid.raster_indices = unique_raster_idxs
        self.voxel_grid.features = feat_buf
        self.voxel_grid.feature_mask = feature_mask

        hit_buf = torch.zeros(
            unique_raster_idxs.shape[0],
            device=self.voxel_grid.device,
        )
        hit_buf[vg1_inv_idxs] += self.voxel_grid.hits
        hit_buf[vg2_inv_idxs] += voxel_grid_new.hits

        miss_buf = torch.zeros(
            unique_raster_idxs.shape[0],
            device=self.voxel_grid.device,
        )
        miss_buf[vg1_inv_idxs] += self.voxel_grid.misses
        miss_buf[vg2_inv_idxs] += voxel_grid_new.misses

        self.voxel_grid.hits = hit_buf
        self.voxel_grid.misses = miss_buf

        min_coords_buf = 1e10 * torch.ones(unique_raster_idxs.shape[0], 3, device=self.voxel_grid.device)
        min_coords_buf[vg1_inv_idxs] = torch.minimum(min_coords_buf[vg1_inv_idxs], self.voxel_grid.min_coords)
        min_coords_buf[vg2_inv_idxs] = torch.minimum(min_coords_buf[vg2_inv_idxs], voxel_grid_new.min_coords)
        self.voxel_grid.min_coords = min_coords_buf

        max_coords_buf = -1e10 * torch.ones(unique_raster_idxs.shape[0], 3, device=self.voxel_grid.device)
        max_coords_buf[vg1_inv_idxs] = torch.maximum(max_coords_buf[vg1_inv_idxs], self.voxel_grid.max_coords)
        max_coords_buf[vg2_inv_idxs] = torch.maximum(max_coords_buf[vg2_inv_idxs], voxel_grid_new.max_coords)
        self.voxel_grid.max_coords = max_coords_buf

        new_grid_idxs = self.voxel_grid.raster_indices_to_grid_indices(self.voxel_grid.raster_indices)
        new_grid_idxs_bev = new_grid_idxs[:, :2]
        new_grid_idxs_z = new_grid_idxs[:, 2]
        new_grid_idxs_bev_raster = new_grid_idxs[:, 0] * self.voxel_grid.metadata.N[1] + new_grid_idxs[:, 1]

        new_minz = torch_scatter.scatter(new_grid_idxs_z, index=new_grid_idxs_bev_raster, reduce="min")
        bottom_voxel_mask = new_grid_idxs_z <= new_minz[new_grid_idxs_bev_raster]

        if debug:
            import open3d as o3d
            #compute passthrough rate
            passthrough_rate = self.voxel_grid.misses / (self.voxel_grid.hits + self.voxel_grid.misses)
            pts = self.voxel_grid.grid_indices_to_pts(
                self.voxel_grid.raster_indices_to_grid_indices(self.voxel_grid.raster_indices)
            )
            colors = torch.stack(
                [torch.zeros_like(passthrough_rate), passthrough_rate, torch.zeros_like(passthrough_rate)], dim=-1
            )
            porosity_pc = o3d.geometry.PointCloud()
            porosity_pc.points = o3d.utility.Vector3dVector(pts.cpu().numpy())
            porosity_pc.colors = o3d.utility.Vector3dVector(colors.cpu().numpy())
            o3d.visualization.draw_geometries([porosity_pc])

        # No culling performed — retain all data
        self.voxel_grid.hits = self.voxel_grid.hits
        self.voxel_grid.misses = self.voxel_grid.misses
        self.voxel_grid.min_coords = self.voxel_grid.min_coords
        self.voxel_grid.max_coords = self.voxel_grid.max_coords
        self.voxel_grid.raster_indices = self.voxel_grid.raster_indices
        self.voxel_grid.features = self.voxel_grid.features
        self.voxel_grid.feature_mask = self.voxel_grid.feature_mask

        if False:
            visualize_feature_pc_with_covariances( vgt=self.voxel_grid.visualize(), voxel_covs=self.voxel_grid.features[:,cov_idxs].view(-1,3,3), every_nth_point=5)


class VoxelGrid:
    """
    Actual class that handles feature aggregation
    """
    def _get_voxel_features_mindist(voxelgrid, pts, features, pos):
        grid_idxs, valid_mask = voxelgrid.get_grid_idxs(pts)
        dists = torch.linalg.norm(pts - pos[:3].view(1,3), dim=-1)

        valid_grid_idxs = grid_idxs[valid_mask]
        valid_feats = features[valid_mask]
        valid_dists = dists[valid_mask]

        valid_raster_idxs = voxelgrid.grid_indices_to_raster_indices(valid_grid_idxs)
        
        sort_idxs = valid_raster_idxs.argsort()

        valid_raster_idxs = valid_raster_idxs[sort_idxs]
        valid_feats = valid_feats[sort_idxs]
        valid_dists = valid_dists[sort_idxs]
        feature_raster_idxs, inv_idxs = valid_raster_idxs.unique(return_inverse=True, sorted=True)

        raster_mindists = torch_scatter.scatter(
            src=valid_dists,
            index=inv_idxs,
            dim_size=feature_raster_idxs.shape[0],
            reduce='min'
        )

        voxel_is_mindist = (valid_dists - raster_mindists[inv_idxs]).abs() < 1e-16
        if voxel_is_mindist.sum() != feature_raster_idxs.shape[0]:
            if voxel_is_mindist.sum() < feature_raster_idxs.shape[0]:
                print("uh oh not all voxels have a mindist")
                import pdb;pdb.set_trace()

        voxel_is_mindist = voxel_is_mindist.float().unsqueeze(-1)

        feat_buf = torch_scatter.scatter(
            src=valid_feats * voxel_is_mindist,
            index=inv_idxs,
            dim_size=feature_raster_idxs.shape[0],
            reduce="sum",
            dim=0
        )

        ## need to do an extra scatter of the mask to handle cases where voxel_is_mindist True for multiple pts in voxel
        ## e.g. if the same point is in there twice
        ## hopefully same spatial pos -> same feature, otherwise tough luck :)
        cnt = torch_scatter.scatter(
            src=voxel_is_mindist,
            index=inv_idxs,
            dim_size=feature_raster_idxs.shape[0],
            reduce="sum",
            dim=0
        )

        voxel_feats = feat_buf / cnt

        return feature_raster_idxs, voxel_feats

    def _get_voxel_features_mindist(voxelgrid, pts, features, pos):
        grid_idxs, valid_mask = voxelgrid.get_grid_idxs(pts)
        dists = torch.linalg.norm(pts - pos[:3].view(1,3), dim=-1)

        valid_grid_idxs = grid_idxs[valid_mask]
        valid_feats = features[valid_mask]
        valid_dists = dists[valid_mask]

        valid_raster_idxs = voxelgrid.grid_indices_to_raster_indices(valid_grid_idxs)
        
        sort_idxs = valid_raster_idxs.argsort()

        valid_raster_idxs = valid_raster_idxs[sort_idxs]
        valid_feats = valid_feats[sort_idxs]
        valid_dists = valid_dists[sort_idxs]
        feature_raster_idxs, inv_idxs = valid_raster_idxs.unique(return_inverse=True, sorted=True)

        raster_mindists = torch_scatter.scatter(
            src=valid_dists,
            index=inv_idxs,
            dim_size=feature_raster_idxs.shape[0],
            reduce='min'
        )

        voxel_is_mindist = (valid_dists - raster_mindists[inv_idxs]).abs() < 1e-16
        if voxel_is_mindist.sum() != feature_raster_idxs.shape[0]:
            if voxel_is_mindist.sum() < feature_raster_idxs.shape[0]:
                print("uh oh not all voxels have a mindist")
                import pdb;pdb.set_trace()

        voxel_is_mindist = voxel_is_mindist.float().unsqueeze(-1)

        feat_buf = torch_scatter.scatter(
            src=valid_feats * voxel_is_mindist,
            index=inv_idxs,
            dim_size=feature_raster_idxs.shape[0],
            reduce="sum",
            dim=0
        )

        ## need to do an extra scatter of the mask to handle cases where voxel_is_mindist True for multiple pts in voxel
        ## e.g. if the same point is in there twice
        ## hopefully same spatial pos -> same feature, otherwise tough luck :)
        cnt = torch_scatter.scatter(
            src=voxel_is_mindist,
            index=inv_idxs,
            dim_size=feature_raster_idxs.shape[0],
            reduce="sum",
            dim=0
        )

        voxel_feats = feat_buf / cnt

        return feature_raster_idxs, voxel_feats

    def _get_voxel_features_scatter(voxelgrid, pts, features):
        grid_idxs, valid_mask = voxelgrid.get_grid_idxs(pts)
        valid_grid_idxs = grid_idxs[valid_mask]
        valid_feats = features[valid_mask]

        valid_raster_idxs = voxelgrid.grid_indices_to_raster_indices(valid_grid_idxs)
        #NOTE: we need the voxel raster indices to be in ascending order (at least, within feat/no-feat) for stuff to work
        feature_raster_idxs, inv_idxs = torch.unique(
            valid_raster_idxs, return_inverse=True, sorted=True
        )
        
        feat_buf = torch_scatter.scatter(
            src=valid_feats, index=inv_idxs, dim_size=feature_raster_idxs.shape[0], reduce="mean", dim=0
        )

        return feature_raster_idxs, feat_buf

    def from_feature_pc(feat_pc, metadata, n_features=-1, pos=None, strategy='avg'):
        """
        Instantiate a VoxelGrid from a feauture pc

        Steps:
            1. separate out feature points and non-feature points
        """
        n_features = len(feat_pc.feature_keys) if n_features == -1 else n_features
        feat_keys = feat_pc.feature_keys[:n_features]

        voxelgrid = VoxelGrid(metadata, feat_keys, feat_pc.device)

        feature_pts = feat_pc.feature_pts.clone()
        feature_pts_features = feat_pc.features[:, :n_features].clone()

        #first scatter and average the feature points
        if strategy == 'avg':
            feature_raster_idxs, voxel_features = VoxelGrid._get_voxel_features_scatter(
                voxelgrid, feature_pts, feature_pts_features
            )
        elif strategy == 'mindist':
            feature_raster_idxs, voxel_features = VoxelGrid._get_voxel_features_mindist(
                voxelgrid, feature_pts, feature_pts_features, pos
                # voxelgrid.to('cpu'), feature_pts.to('cpu'), feature_pts_features.to('cpu'), pos.to('cpu')
            )

        # voxelgrid = voxelgrid.to('cuda')
        # feature_raster_idxs = feature_raster_idxs.to('cuda')
        # voxel_features = voxel_features.to('cuda')

        return feature_raster_idxs, feat_buf
    
    def from_feature_pc(feat_pc, metadata, n_features=-1, pos=None, strategy='avg'):
        """
        Instantiate a VoxelGrid from a feauture pc

        Steps:
            1. separate out feature points and non-feature points
        """
        n_features = len(feat_pc.feature_keys) if n_features == -1 else n_features
        feat_keys = feat_pc.feature_keys[:n_features]

        voxelgrid = VoxelGrid(metadata, feat_keys, feat_pc.device)

        feature_pts = feat_pc.feature_pts.clone()
        feature_pts_features = feat_pc.features[:, :n_features].clone()

        #first scatter and average the feature points
        if strategy == 'avg':
            feature_raster_idxs, voxel_features = VoxelGrid._get_voxel_features_scatter(
                voxelgrid, feature_pts, feature_pts_features
            )
        elif strategy == 'mindist':
            feature_raster_idxs, voxel_features = VoxelGrid._get_voxel_features_mindist(
                voxelgrid, feature_pts, feature_pts_features, pos
                # voxelgrid.to('cpu'), feature_pts.to('cpu'), feature_pts_features.to('cpu'), pos.to('cpu')
            )

        # voxelgrid = voxelgrid.to('cuda')
        # feature_raster_idxs = feature_raster_idxs.to('cuda')
        # voxel_features = voxel_features.to('cuda')

        #then add in non-feature points
        non_feature_pts = feat_pc.non_feature_pts.clone()
        non_feature_pts = feat_pc.non_feature_pts.clone()
        grid_idxs, valid_mask = voxelgrid.get_grid_idxs(non_feature_pts)
        valid_grid_idxs = grid_idxs[valid_mask]
        valid_raster_idxs = voxelgrid.grid_indices_to_raster_indices(valid_grid_idxs)
        non_feature_raster_idxs = torch.unique(valid_raster_idxs)

        _raster_idxs_cnt_in = torch.cat([feature_raster_idxs, feature_raster_idxs, non_feature_raster_idxs])
        _raster_idxs, _raster_idx_cnts = torch.unique(_raster_idxs_cnt_in, return_counts=True)
        non_feature_raster_idxs = _raster_idxs[_raster_idx_cnts == 1]

        #store in voxel grid
        n_feat_voxels = feature_raster_idxs.shape[0]
        all_raster_idxs = torch.cat([feature_raster_idxs, non_feature_raster_idxs])
        feat_mask = torch.zeros(all_raster_idxs.shape[0], dtype=torch.bool, device=feat_pc.device)
        feat_mask[:n_feat_voxels] = True

        #I need raster idxs to be sorted to do other downstream ops
        all_raster_idxs, idxs = torch.sort(all_raster_idxs)
        feat_mask = feat_mask[idxs]

        voxelgrid.raster_indices = all_raster_idxs
        voxelgrid.features = voxel_features
        voxelgrid.feature_mask = feat_mask

        voxelgrid.hits = torch.ones(all_raster_idxs.shape[0], device=voxelgrid.device)
        voxelgrid.misses = torch.zeros(all_raster_idxs.shape[0], device=voxelgrid.device)

        #scatter min/max coords
        all_pts = feat_pc.pts
        all_grid_idxs, valid_mask = voxelgrid.get_grid_idxs(all_pts)
        all_valid_pts = all_pts[valid_mask]
        all_pts_raster_idxs = voxelgrid.grid_indices_to_raster_indices(all_grid_idxs[valid_mask])

        pt_raster_idxs, inv_idxs = torch.unique(all_pts_raster_idxs, return_inverse=True, sorted=True)

        voxelgrid.min_coords = torch_scatter.scatter(
            src=all_valid_pts, index=inv_idxs, dim_size=all_raster_idxs.shape[0], reduce="min", dim=0
        )

        voxelgrid.max_coords = torch_scatter.scatter(
            src=all_valid_pts, index=inv_idxs, dim_size=all_raster_idxs.shape[0], reduce="max", dim=0
        )

        return voxelgrid

    def __init__(self, metadata, feature_keys, device):
        self.metadata = metadata
        self.feature_keys = feature_keys
        self.n_features = len(self.feature_keys)

        #raster indices of all points in voxel grid
        self.raster_indices = torch.zeros(0, dtype=torch.long, device=device)

        #list of features for all points in grid with features
        self.features = torch.zeros(0, self.n_features, dtype=torch.float, device=device)

        #mapping from indices to features (i.e. raster_indices[mask] = features)
        self.feature_mask = torch.zeros(0, dtype=torch.bool, device=device)

        self.hits = torch.zeros(0, dtype=torch.float, device=device) + 1e-8
        self.misses = torch.zeros(0, dtype=torch.float, device=device)

        #store min/max coords per voxel for more accurate reconstruction
        self.min_coords = torch.zeros(0, 3, dtype=torch.float, device=device)
        self.max_coords = torch.zeros(0, 3, dtype=torch.float, device=device)

        self.device = device

    @property
    def non_feature_raster_indices(self):
        return self.raster_indices[~self.feature_mask]

    @property
    def feature_raster_indices(self):
        return self.raster_indices[self.feature_mask]

    @property
    def midpoints(self):
        """
        Return the midpoints of all occupied voxels
            (i.e. (max coords + min_coords) / 2.)
        """
        return 0.5 * (self.min_coords + self.max_coords)

    @property
    def feature_midpoints(self):
        return 0.5 * (self.min_coords[self.feature_mask] + self.max_coords[self.feature_mask])

    @property
    def non_feature_midpoints(self):
        return 0.5 * (self.min_coords[~self.feature_mask] + self.max_coords[~self.feature_mask])

    def get_grid_idxs(self, pts):
        """
        Get indexes for positions given map metadata
        """
        gidxs = torch.div((pts[:, :3] - self.metadata.origin.view(1,3)), self.metadata.resolution.view(1,3), rounding_mode='floor').long()
        mask = (gidxs >=  0).all(dim=-1) & (gidxs < self.metadata.N.view(1,3)).all(dim=-1) 
        return gidxs, mask

    def shift(self, px_shift):
        """
        Apply a pixel shift to the map

        Args:
            px_shift: Tensor of [dx, dy, dz], where the ORIGIN of the map is moved by this many cells
                e.g. if px_shift is [-3, 5], the new origin is 3*res units left and 5*res units up
                        note that this means the data is shifted 3 cells right and 5 cells down
        """
        #shift feature indices
        grid_indices = self.raster_indices_to_grid_indices(self.raster_indices)
        grid_indices = grid_indices - px_shift.view(1, 3)
        mask = self.grid_idxs_in_bounds(grid_indices)
        self.raster_indices = self.grid_indices_to_raster_indices(grid_indices[mask])

        self.features = self.features[mask[self.feature_mask]]
        self.feature_mask = self.feature_mask[mask]
        self.hits = self.hits[mask]
        self.misses = self.misses[mask]
        self.min_coords = self.min_coords[mask]
        self.max_coords = self.max_coords[mask]

        self.metadata.origin += px_shift * self.metadata.resolution

    def pts_in_bounds(self, pts):
        """Check if points are in bounds

        Args:
            pts: [Nx3] Tensor of coordinates

        Returns:
            valid: [N] mask of whether point is within voxel grid
        """
        _min = self.metadata.origin.view(1, 3)
        _max = (self.metadata.origin + self.metadata.length).view(1, 3)

        low_check = (pts >= _min).all(axis=-1)
        high_check = (pts < _max).all(axis=-1)
        return low_check & high_check

    def grid_idxs_in_bounds(self, grid_idxs):
        """Check if grid idxs are in bounds

        Args:
            grid_idxs: [Nx3] Long Tensor of grid idxs

        Returns:
            valid: [N] mask of whether idx is within voxel grid
        """
        _min = torch.zeros_like(self.metadata.N).view(1, 3)
        _max = self.metadata.N.view(1, 3)

        low_check = (grid_idxs >= _min).all(axis=-1)
        high_check = (grid_idxs < _max).all(axis=-1)

        return low_check & high_check

    def grid_indices_to_pts(self, grid_indices, centers=True):
        """Convert a set of grid coordinates to cartesian coordinates

        Args:
            grid_indices: [Nx3] Tensor of grid coordinates
            centers: Set this flag to false to return voxel lower-bottom-left, else return voxel centers
        """
        coords = grid_indices * self.metadata.resolution.view(
            1, 3
        ) + self.metadata.origin.view(1, 3)

        if centers:
            coords += (self.metadata.resolution / 2.0).view(1, 3)

        return coords

    def grid_indices_to_raster_indices(self, grid_idxs):
        """Convert a set of grid indices to raster indices

        Args:
            grid_idxs: [Nx3] Long Tensor of grid indices

        Returns:
            raster_idxs: [N] Long Tensor of raster indices
        """
        _N1 = self.metadata.N[1] * self.metadata.N[2]
        _N2 = self.metadata.N[2]

        return _N1 * grid_idxs[:, 0] + _N2 * grid_idxs[:, 1] + grid_idxs[:, 2]

    def raster_indices_to_grid_indices(self, raster_idxs):
        """Convert a set of raster indices to grid indices

        Args:
            raster_idxs: [N] Long Tensor of raster indices

        Returns:
            grid_idxs: [Nx3] Long Tensor of grid indices
        """
        _N1 = self.metadata.N[1] * self.metadata.N[2]
        _N2 = self.metadata.N[2]

        xs = torch.div(raster_idxs, _N1, rounding_mode="floor").long()
        ys = torch.div(raster_idxs % _N1, _N2, rounding_mode="floor").long()
        zs = raster_idxs % _N2

        return torch.stack([xs, ys, zs], axis=-1)

    def random_init(device='cpu'):
        metadata = LocalMapperMetadata.random_init(ndim=3, device=device)

        fks = FeatureKeyList(
            label=[f'rand_{i}' for i in range(5)],
            metainfo=['feat'] * 5
        )

        maxn = torch.prod(metadata.N)
        nvox = int(maxn/2)
        nfeats = int(nvox/2)

        raster_idxs = torch.randperm(maxn, device=device)[:nvox]
        feats = torch.randn(nfeats, 5, device=device)

        mask_idxs = torch.randperm(nvox)[:nfeats]
        mask = torch.zeros(len(raster_idxs), dtype=torch.bool, device=device)
        mask[mask_idxs] = True

        hits = (10. * torch.rand(nvox, device=device)).long()
        misses = (10. * torch.rand(nvox, device=device)).long()

        voxel_grid = VoxelGrid(metadata, fks, device=device)
        voxel_grid.raster_indices = raster_idxs
        voxel_grid.feature_mask = mask
        voxel_grid.features = feats
        voxel_grid.hits = hits
        voxel_grid.misses = misses

        voxel_centers = voxel_grid.grid_indices_to_pts(voxel_grid.raster_indices_to_grid_indices(raster_idxs))
        voxel_grid.min_coords = voxel_centers - metadata.resolution/2.
        voxel_grid.max_coords = voxel_centers + metadata.resolution/2.

        return voxel_grid

    def __eq__(self, other):
        if self.feature_keys != other.feature_keys:
            return False

        if self.metadata != other.metadata:
            return False

        if not (self.raster_indices == other.raster_indices).all():
            return False

        if not (self.feature_mask == other.feature_mask).all():
            return False

        if not torch.allclose(self.features, other.features):
            return False

        if not (self.hits == other.hits).all():
            return False

        if not (self.misses == other.misses).all():
            return False

        if not torch.allclose(self.min_coords, other.min_coords):
            return False

        if not torch.allclose(self.max_coords, other.max_coords):
            return False

        return True

    def visualize(self, viz_all=True, midpoints=True, sample_frac=1.0):
        pc = o3d.geometry.PointCloud()
        if midpoints:
            pts = self.feature_midpoints
        else:
            pts = self.grid_indices_to_pts(
                self.raster_indices_to_grid_indices(self.feature_raster_indices)
            )
        colors = normalize_dino(self.features[:, :3])

        #all_indices is a superset of indices
        if viz_all:
            if midpoints:
                non_colorized_pts = self.non_feature_midpoints
            else:
                non_colorized_idxs = self.non_feature_raster_indices

                non_colorized_pts = self.grid_indices_to_pts(
                    self.raster_indices_to_grid_indices(non_colorized_idxs)
                )

            color_placeholder = 0.3 * torch.ones(non_colorized_pts.shape[0], 3, device=non_colorized_pts.device)

            pts = torch.cat([pts, non_colorized_pts], dim=0)
            colors = torch.cat([colors, color_placeholder], dim=0)

        pc.points = o3d.utility.Vector3dVector(pts.cpu().numpy())
        pc.colors = o3d.utility.Vector3dVector(colors.cpu().numpy())

        if sample_frac < 1.0:
            pc = pc.uniform_down_sample(every_k_points=int(1/sample_frac))

        return pc
        # o3d.visualization.draw_geometries([pc])

    def to(self, device):
        self.device = device
        self.metadata = self.metadata.to(device)
        self.raster_indices = self.raster_indices.to(device)
        self.feature_mask = self.feature_mask.to(device)
        self.features = self.features.to(device)
        self.hits = self.hits.to(device)
        self.misses = self.misses.to(device)
        self.min_coords = self.min_coords.to(device)
        self.max_coords = self.max_coords.to(device)
        return self

class VoxelCovarianceGrid(VoxelGrid):
    """
    Voxel grid that builds using covariance information
    """

    def __init__(self, metadata, feature_keys, device):
        super().__init__(self, metadata, feature_keys, device)

    def from_feature_pc(feat_pc, metadata, n_features=-1):
            """
            Instantiate a VoxelGrid from a feature pc

            Steps:
                1. separate out feature points and non-feature points
            """

            # --- Feature key construction ---
            def create_new_feature_keys(feat_keys):
                new_labels = feat_keys.label + [f"center_{axis}" for axis in ("x", "y", "z")]
                new_metainfo = feat_keys.metainfo + ["voxelmapper"] * 3
                return FeatureKeyList(label=new_labels, metainfo=new_metainfo)

            feat_keys = create_new_feature_keys(feat_pc.feature_keys)
            n_features = len(feat_keys) if n_features == -1 else n_features
            voxelgrid = VoxelGrid(metadata, feat_keys, feat_pc.device)

            feature_pts = feat_pc.feature_pts
            feature_pts_features = feat_pc.features[:, :n_features]
            non_feature_pts = feat_pc.non_feature_pts

            #first scatter and average the feature points
            grid_idxs, valid_mask = voxelgrid.get_grid_idxs(feature_pts)
            valid_grid_idxs = grid_idxs[valid_mask]
            
            valid_raster_idxs = voxelgrid.grid_indices_to_raster_indices(valid_grid_idxs) #for each PC point, which voxel it belongs to

            # --- Filter points with too large covariances ---
            device = feat_pc.device
            eps = 1e-6
            eps_eye = eps *  torch.eye(3, device=device)
            pts = feat_pc.feature_pts                          # [N, 3]

            feature_keys = feat_pc.feature_keys 
            cov_idxs = feature_keys.index_metainfo("macvo")
            covs = feat_pc.features[:,cov_idxs].reshape(-1, 3, 3)   # [N, 3, 3]
            idxs_set = set(cov_idxs)
            remaining_feats_idxs = [i for i in range(len(feature_keys)) if i not in idxs_set]

            colors = feat_pc.features[:, remaining_feats_idxs]

            if len(pts) == 0 or covs.numel() == 0:  
                return voxelgrid
                
            # Filter by 97.5 percentile of total variance (size of covariance ellipses)
            var_sum_per_point = torch.diagonal(covs, dim1=1, dim2=2).sum(dim=1)  # [N]
            var_thresh = torch.quantile(var_sum_per_point, 0.99)
            valid_mask = var_sum_per_point <= var_thresh

            pts = pts[valid_mask]
            colors = colors[valid_mask]
            covs = covs[valid_mask]
            
            # Filter out non-positive definite covariances based on determinant
            # A matrix is positive definite if all eigenvalues > 0, which means det > 0
            # (though det > 0 is necessary but not sufficient, we also check smallest eigenvalue)
            covs_det = torch.linalg.det(covs)
            eigvals_min = torch.linalg.eigvalsh(covs).min(dim=1).values
            pd_mask = (covs_det > 0) & (eigvals_min > 0)
            
            n_removed = (~pd_mask).sum().item()
            if n_removed > 0:
                print(f"Removing {n_removed}/{covs.shape[0]} non-positive-definite covariances")
            
            pts = pts[pd_mask]
            colors = colors[pd_mask]
            covs = covs[pd_mask]


            # --- Only use pts that are in the new voxelgrid!---
            grid_idxs, valid_mask = voxelgrid.get_grid_idxs(pts)
            valid_grid_idxs = grid_idxs[valid_mask]
            valid_pts = pts[valid_mask]
            valid_covs = covs[valid_mask]
            valid_colors = colors[valid_mask]
            valid_raster_idxs = voxelgrid.grid_indices_to_raster_indices(valid_grid_idxs)

            # Compact voxel indices
            feature_raster_idxs, inv_idxs = torch.unique(valid_raster_idxs, return_inverse=True, sorted=True)
            num_voxels = feature_raster_idxs.shape[0]

            # --- Weight normalization ---
            voxel_counts = torch_scatter.scatter(
                torch.ones_like(inv_idxs, dtype=torch.float32),
                inv_idxs,
                dim=0,
                dim_size=num_voxels,
                reduce="sum"
            )

            weights = 1.0 / voxel_counts[inv_idxs].clamp_min(1.0) 

            # Compute precisions via Cholesky solve (without explicit inverse)
            
            # # Debug: Print valid_covs information
            # print(f"valid_covs shape: {valid_covs.shape}")
            # print(f"valid_covs dtype: {valid_covs.dtype}, device: {valid_covs.device}")
            # print(f"valid_covs stats - min: {valid_covs.min().item():.6f}, max: {valid_covs.max().item():.6f}, mean: {valid_covs.mean().item():.6f}")
            
            # # Check for NaN or Inf values
            # if torch.isnan(valid_covs).any():
            #     print(f"WARNING: Found {torch.isnan(valid_covs).sum().item()} NaN values in valid_covs!")
            # if torch.isinf(valid_covs).any():
            #     print(f"WARNING: Found {torch.isinf(valid_covs).sum().item()} Inf values in valid_covs!")
            
            # # Print first few matrices for inspection
            # print(f"First 3 covariance matrices:\n{valid_covs[:3]}")
            
            # # Check positive definiteness by eigenvalues
            # eigvals = torch.linalg.eigvalsh(valid_covs)
            # is_pd = (eigvals > 0).all(dim=1)
            # n_pd = is_pd.sum().item()
            # n_not_pd = (~is_pd).sum().item()
            # print(f"Positive definite check: {n_pd}/{valid_covs.shape[0]} are PD, {n_not_pd} are NOT PD (min eigenvalue: {eigvals.min().item():.2e})")
            
            L = torch.linalg.cholesky(valid_covs + eps_eye)               # [N,3,3]
            
            # Solve L L^T X = I to get precision matrices
            I3 = torch.eye(3, device=device).expand(valid_covs.shape[0], 3, 3)
            precisions = torch.cholesky_solve(I3, L)                     # [N,3,3]

            weighted_precisions = weights[:, None, None] * precisions
            fused_precision_sum = torch_scatter.scatter(
                weighted_precisions, inv_idxs, dim=0, dim_size=num_voxels, reduce="sum"
            )

            weighted_precision_centers = weights[:, None] * (precisions @ valid_pts[:, :, None]).squeeze(-1)
            fused_precision_center_sum = torch_scatter.scatter(
                weighted_precision_centers, inv_idxs, dim=0, dim_size=num_voxels, reduce="sum"
            )

            # Compute fused covariance and centers without explicit inverse
            L_fused = torch.linalg.cholesky(fused_precision_sum + eps_eye)
            fused_covariances = torch.cholesky_solve(torch.eye(3, device=device).expand(num_voxels, 3, 3), L_fused)
            fused_centers = torch.cholesky_solve(fused_precision_center_sum.unsqueeze(-1), L_fused).squeeze(-1)

            # Compute mean color per voxel instead of weighted fusion
            fused_colors = torch_scatter.scatter_mean(
                valid_colors,
                inv_idxs,
                dim=0,
                dim_size=num_voxels
            )

            # --- Combine fused features ---
            feat_buf = torch.cat([fused_colors, fused_covariances.reshape(-1, 9), fused_centers], dim=1)

            # --- Non-feature voxels ---
            valid_mask = voxelgrid.get_grid_idxs(feat_pc.non_feature_pts)[1]
            non_feature_raster_idxs = voxelgrid.grid_indices_to_raster_indices(voxelgrid.get_grid_idxs(feat_pc.non_feature_pts)[0][valid_mask])
            non_feature_raster_idxs = torch.unique(non_feature_raster_idxs)

            # Keep only non-feature voxels that are not already in feature_raster_idxs
            all_raster_idxs_combined = torch.cat([feature_raster_idxs, non_feature_raster_idxs])
            unique_raster_idxs, counts = torch.unique(all_raster_idxs_combined, return_counts=True)
            non_feature_raster_idxs = unique_raster_idxs[counts == 1]

            # Combine feature and non-feature voxels
            all_raster_idxs = torch.cat([feature_raster_idxs, non_feature_raster_idxs])
            feat_mask = torch.zeros(all_raster_idxs.shape[0], dtype=torch.bool, device=feat_pc.device)
            feat_mask[:feature_raster_idxs.shape[0]] = True

            # Sort raster indices and align mask
            all_raster_idxs, sort_idx = torch.sort(all_raster_idxs)
            feat_mask = feat_mask[sort_idx]

            # --- Assign to voxel grid ---
            voxelgrid.raster_indices = all_raster_idxs
            voxelgrid.features = feat_buf
            voxelgrid.feature_mask = feat_mask
            voxelgrid.hits = torch.ones(all_raster_idxs.shape[0], device=voxelgrid.device)
            voxelgrid.misses = torch.zeros(all_raster_idxs.shape[0], device=voxelgrid.device)

            # --- Min/max per voxel (only valid points) ---
            all_pts = pts
            all_grid_idxs, valid_mask = voxelgrid.get_grid_idxs(all_pts)
            all_valid_pts = all_pts[valid_mask]
            all_pts_raster_idxs = voxelgrid.grid_indices_to_raster_indices(all_grid_idxs[valid_mask])
            pt_raster_idxs, inv_idxs = torch.unique(all_pts_raster_idxs, return_inverse=True, sorted=True)

            voxelgrid.min_coords = torch_scatter.scatter(
                src=all_valid_pts, index=inv_idxs, dim_size=all_raster_idxs.shape[0], reduce="min", dim=0
            )
            voxelgrid.max_coords = torch_scatter.scatter(
                src=all_valid_pts, index=inv_idxs, dim_size=all_raster_idxs.shape[0], reduce="max", dim=0
            )

            if False:
                visualize_feature_pc_with_covariances( feature_pc=feat_pc, vgt=voxelgrid.visualize(),
                    # voxel_covs=voxelgrid.features[:, cov_idxs].view(-1, 3, 3),
                    # every_nth_point=100
                )
                import pdb; pdb.set_trace()
            return voxelgrid