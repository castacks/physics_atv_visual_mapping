import open3d as o3d
import torch
import torch_scatter
from physics_atv_visual_mapping.localmapping.voxel.voxel_localmapper import VoxelLocalMapper, VoxelGrid, create_grid
from physics_atv_visual_mapping.feature_key_list import FeatureKeyList
from ros_torch_converter.datatypes.pointcloud import FeaturePointCloudTorch
from physics_atv_visual_mapping.utils import *

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