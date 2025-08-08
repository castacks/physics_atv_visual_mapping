# Import slope and normal estimation processing blocks
from physics_atv_visual_mapping.terrain_estimation.processing_blocks.slope import Slope
from physics_atv_visual_mapping.terrain_estimation.processing_blocks.terrain_normals_gradient import TerrainNormalsGradient
from physics_atv_visual_mapping.localmapping.bev.bev_localmapper import BEVGrid
from physics_atv_visual_mapping.localmapping.metadata import LocalMapperMetadata

import torch
import numpy as np
import open3d as o3d
import matplotlib.pyplot as plt

class NormalsTester():

    def viz(self, slope_x, gt_slope_x, slope_y, gt_slope_y, terrain, normal_vectors, gt_normal_vectors):
        # Flatten terrain and normals for 3D point cloud visualization
        terrain_pts = np.reshape(terrain, (-1, 3))
        normals_list = np.reshape(normal_vectors, (-1, 3))
        gt_normals_list = np.reshape(gt_normal_vectors, (-1, 3))

        # Compute absolute error in slope estimates
        absolute_error_x = np.abs(slope_x - gt_slope_x)
        absolute_error_y = np.abs(slope_y - gt_slope_y)

        # Print average absolute slope errors (ignoring borders)
        average_absolute_error_x = np.mean(absolute_error_x[1:-1, 1:-1])
        print(average_absolute_error_x)
        average_absolute_error_y = np.mean(absolute_error_y[1:-1, 1:-1])
        print(average_absolute_error_y)

        # Compute angular error between estimated and ground truth normals (in degrees)
        angle_error = np.arccos(np.sum(normal_vectors * gt_normal_vectors, axis=-1)) * (180 / torch.pi)
        average_angel_error = np.mean(angle_error[2:-2, 2:-2])  # ignore edges
        print(average_angel_error)

        # If grid is small, generate detailed 2D plots for slopes and normal error
        if bev_grid.metadata.N[0] <= 40 and bev_grid.metadata.N[1] <= 40:

            # Plot slope gradients along X and Y
            fig_x, ax1 = plt.subplots(1, 2)
            fig_y, ax2 = plt.subplots(1, 2)
            fig_n, ax3 = plt.subplots()

            fig_x.suptitle('X Gradient', fontsize=16)
            fig_x.text(0.5, 0.1, f'Average Error: {average_absolute_error_x:.2f}', va='bottom', ha='center', fontsize=14)
            ax1[0].matshow(slope_x.round(4), cmap='viridis')
            ax1[0].set_title("Sobel Gradient")
            ax1[1].matshow(gt_slope_x.round(4), cmap='viridis')
            ax1[1].set_title("Ground Truth Gradient")

            fig_y.suptitle('Y Gradient', fontsize=16)
            fig_y.text(0.5, 0.1, f'Average Error: {average_absolute_error_y:.2f}', va='bottom', ha='center', fontsize=14)
            ax2[0].matshow(slope_y.round(4), cmap='viridis')
            ax2[0].set_title("Sobel Gradient")
            ax2[1].matshow(gt_slope_y.round(4), cmap='viridis')
            ax2[1].set_title("Ground Truth Gradient")

            fig_n.suptitle('Estimated Normal Error', fontsize=16)
            fig_n.text(0.5, 0.05, f'Average Error: {average_angel_error:.2f} Degrees', va='bottom', ha='center', fontsize=14)
            ax3.matshow(angle_error.round(4), cmap='viridis')

            # Calculate text size based on map size
            text_size = 10 + 2 * ((10 - bev_grid.metadata.length[0].item()) / 5)

            # Annotate each cell with values
            for i in np.arange(bev_grid.metadata.N[0]):
                for j in np.arange(bev_grid.metadata.N[1]):
                    dx = slope_x.round(1)[j, i]
                    gt_dx = gt_slope_x.round(1)[j, i]
                    dy = slope_y.round(1)[j, i]
                    gt_dy = gt_slope_y.round(1)[j, i]
                    n_error = angle_error.round(1)[j, i]
                    ax1[0].text(i, j, str(dx), va='center', ha='center', size=text_size)
                    ax1[1].text(i, j, str(gt_dx), va='center', ha='center', size=text_size)
                    ax2[0].text(i, j, str(dy), va='center', ha='center', size=text_size)
                    ax2[1].text(i, j, str(gt_dy), va='center', ha='center', size=text_size)
                    ax3.text(i, j, str(n_error), va='center', ha='center', size=text_size)

            plt.show()

        # 3D visualization of normals using Open3D
        terrain_viz = o3d.geometry.PointCloud()
        terrain_viz.points = o3d.utility.Vector3dVector(terrain_pts)

        end_pts_sobel = terrain_pts + normals_list
        end_pts_gt = terrain_pts + gt_normals_list

        # Create line sets for normals
        pts_list_sobel = np.concatenate((terrain_pts, end_pts_sobel), axis=0)
        normal_lines_idx = np.stack((np.arange(0, end_pts_sobel.shape[0]), np.arange(end_pts_sobel.shape[0], 2 * end_pts_sobel.shape[0])), axis=-1)

        pts_list_gt = np.concatenate((terrain_pts, end_pts_gt), axis=0)
        gt_lines_idx = np.stack((np.arange(0, end_pts_gt.shape[0]), np.arange(end_pts_gt.shape[0], 2 * end_pts_gt.shape[0])), axis=-1)

        # Line sets for predicted and GT normals
        normals_lineset = o3d.geometry.LineSet()
        normals_lineset.points = o3d.utility.Vector3dVector(pts_list_sobel)
        normals_lineset.lines = o3d.utility.Vector2iVector(normal_lines_idx)

        gt_lineset = o3d.geometry.LineSet()
        gt_lineset.points = o3d.utility.Vector3dVector(pts_list_gt)
        gt_lineset.lines = o3d.utility.Vector2iVector(gt_lines_idx)

        # Color GT normals green
        colors = [[0, 1, 0] for _ in range(len(gt_lines_idx))]
        gt_lineset.colors = o3d.utility.Vector3dVector(colors)

        # Show 3D terrain + normals
        o3d.visualization.draw_geometries([terrain_viz, gt_lineset, normals_lineset])

    def create_terrain_from_normal(self, metadata, gt_normal_vector):
        bev_extent = (
            metadata.origin[0].item(),
            metadata.origin[0].item() + bev_grid.metadata.length[0].item(),
            metadata.origin[1].item(),
            metadata.origin[1].item() + bev_grid.metadata.length[1].item(),
        )

        # Create BEV X/Y grid
        bev_x = np.linspace(bev_extent[0], bev_extent[1], metadata.N[0])
        bev_y = np.linspace(bev_extent[2], bev_extent[3], metadata.N[1])
        bev_x, bev_y = np.meshgrid(bev_x, bev_y, indexing='ij')

        # Compute terrain height using plane equation
        terrain = -((gt_normal_vector[0]*bev_x) + (gt_normal_vector[1]*bev_y)) / gt_normal_vector[2]
        terrain = np.stack((bev_x, bev_y, terrain), axis=-1)
        terrain = torch.from_numpy(terrain)

        # Ground truth slopes derived from the normal
        slope_x_gt = np.full_like(bev_x, -gt_normal_vector[0]/gt_normal_vector[2])
        slope_y_gt = np.full_like(bev_y, -gt_normal_vector[1]/gt_normal_vector[2])

        # Fill GT normal vector field
        gt_normal_vectors = np.zeros_like(terrain)
        gt_normal_vectors[..., 0] = gt_normal_vector[0]
        gt_normal_vectors[..., 1] = gt_normal_vector[1]
        gt_normal_vectors[..., 2] = gt_normal_vector[2]

        return terrain, slope_x_gt, slope_y_gt, gt_normal_vectors

    def create_wave_terrain(self, metadata, frequency_x, frequency_y, amplitude_x, amplitude_y, h_shift_x, h_shift_y, terrain_noise):
        bev_extent = (
            metadata.origin[0].item(),
            metadata.origin[0].item() + metadata.length[0].item(),
            metadata.origin[1].item(),
            metadata.origin[1].item() + metadata.length[1].item(),
        )

        # Create BEV grid
        bev_x = torch.linspace(bev_extent[0], bev_extent[1], metadata.N[0])
        bev_y = torch.linspace(bev_extent[2], bev_extent[3], metadata.N[1])
        bev_x, bev_y = torch.meshgrid(bev_x, bev_y, indexing='ij')

        # Compute terrain elevation as a sum of sine/cosine waves
        terrain = amplitude_x * torch.cos(frequency_x * (bev_x - h_shift_x)) + amplitude_y * torch.cos(frequency_y * (bev_y - h_shift_y))
        terrain = terrain + terrain_noise
        terrain = torch.stack((bev_x, bev_y, terrain), axis=-1)

        # Compute analytical ground-truth slopes
        gt_slope_x = (-amplitude_x * frequency_x) * torch.sin(frequency_x * (bev_x - h_shift_x))
        gt_slope_y = (-amplitude_y * frequency_y) * torch.sin(frequency_y * (bev_y - h_shift_y))

        # Compute ground-truth normals using cross product
        zero_dim = torch.zeros_like(gt_slope_x)
        ones_dim = torch.ones_like(gt_slope_y)

        slope_x_vec = torch.stack((ones_dim, zero_dim, gt_slope_x), axis=-1)
        slope_y_vec = torch.stack((zero_dim, ones_dim, gt_slope_y), axis=-1)

        gt_normal_vectors = torch.cross(slope_x_vec, slope_y_vec)
        gt_normal_vectors = gt_normal_vectors / torch.linalg.norm(gt_normal_vectors, axis=-1, keepdims=True)

        return terrain, gt_slope_x, gt_slope_y, gt_normal_vectors

    def get_wave_bevgrid(self, metadata, feature_keys, wave_config, kernel_config, terrain_noise):
        n_features = len(feature_keys)
        bev_grid = BEVGrid(metadata, n_features, feature_keys)

        # Create synthetic wave terrain and ground-truth data
        terrain, gt_slope_x, gt_slope_y, gt_normal_vectors = self.create_wave_terrain(
            metadata,
            wave_config['frequency_x'],
            wave_config['frequency_y'],
            wave_config['amplitude_x'],
            wave_config['amplitude_y'],
            wave_config['h_shift_x'],
            wave_config['h_shift_y'],
            terrain_noise
        )

        # Extract and populate BEV features
        bev_grid.data[..., bev_grid.feature_keys.index('bev_x')] = terrain[..., 0]
        bev_grid.data[..., bev_grid.feature_keys.index('bev_y')] = terrain[..., 1]
        bev_grid.data[..., bev_grid.feature_keys.index('terrain')] = terrain[..., 2]
        bev_grid.data[..., bev_grid.feature_keys.index('mask')] = torch.ones_like(terrain[..., 2])

        # Run slope estimation
        slope_solver = Slope(metadata, n_features, 'terrain', 'mask', radius=kernel_config['radius'], max_slope=100, kernel_type=kernel_config['kernel_type'])
        bev_grid = slope_solver.run(0, bev_grid)

        # Save GT slope
        bev_grid.data[..., bev_grid.feature_keys.index('gt_slope_x')] = gt_slope_x
        bev_grid.data[..., bev_grid.feature_keys.index('gt_slope_y')] = gt_slope_y

        # Run normal estimation from slopes
        normal_solver = TerrainNormalsGradient(metadata, n_features, 'slope_x', 'slope_y', 'mask')
        bev_grid = normal_solver.run(0, bev_grid)

        # Save GT normals
        for i, key in enumerate(['gt_normals_x', 'gt_normals_y', 'gt_normals_z']):
            bev_grid.data[..., bev_grid.feature_keys.index(key)] = gt_normal_vectors[..., i]

        return bev_grid

if __name__ == '__main__':
    # Define metadata and features
    metadata = LocalMapperMetadata([-10, -10], [20, 20], [0.5, 0.5])
    feature_keys = ['mask', 'bev_x', 'bev_y', 'terrain', 'slope_x', 'slope_y', 'slope', 'gt_slope_x', 'gt_slope_y', 'normals_x', 'normals_y', 'normals_z', 'gt_normals_x', 'gt_normals_y', 'gt_normals_z']

    # Define test wave and kernel configs
    wave_config = {
        'frequency_x': 1,
        'frequency_y': 1,
        'amplitude_x': 1,
        'amplitude_y': 1,
        'h_shift_x': 0,
        'h_shift_y': 0,
        'noise_range': 20
    }
    kernel_config = {
        'radius': 0.5,
        'kernel_type': 'sobel'
    }

    # Generate terrain noise at 3 different intensities
    terrain_noise_5 = (np.random.rand(metadata.N[0], metadata.N[1]) - 0.5) * (5 / 100)
    terrain_noise_10 = (np.random.rand(metadata.N[0], metadata.N[1]) - 0.5) * (10 / 100)
    terrain_noise_20 = (np.random.rand(metadata.N[0], metadata.N[1]) - 0.5) * (20 / 100)

    # Create tester and run terrain generation + estimation
    normals_tester = NormalsTester()
    bev_grid = normals_tester.get_wave_bevgrid(metadata, feature_keys, wave_config, kernel_config, terrain_noise_20)

    # Extract slope and normal data
    slope_x = bev_grid.data[..., 4].cpu().numpy()
    slope_y = bev_grid.data[..., 5].cpu().numpy()
    gt_slope_x = bev_grid.data[..., 7].cpu().numpy()
    gt_slope_y = bev_grid.data[..., 8].cpu().numpy()

    terrain = np.stack((bev_grid.data[..., 1], bev_grid.data[..., 2], bev_grid.data[..., 3]), axis=-1)
    normal_vectors = np.stack((bev_grid.data[..., 9], bev_grid.data[..., 10], bev_grid.data[..., 11]), axis=-1)
    gt_normal_vectors = np.stack((bev_grid.data[..., 12], bev_grid.data[..., 13], bev_grid.data[..., 14]), axis=-1)

    # Visualize results
    normals_tester.viz(slope_x, gt_slope_x, slope_y, gt_slope_y, terrain, normal_vectors, gt_normal_vectors)
