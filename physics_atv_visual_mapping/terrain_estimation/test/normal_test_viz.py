import os
import yaml
import copy
import torch
import numpy as np
import itertools
import matplotlib.pyplot as plt
import pandas as pd

class NormalTesterVisualizer():

    def save_raw_data_viz(slope_x, gt_slope_x, slope_y, gt_slope_y, normal_vectors, gt_normal_vectors, base_dir, idx):

        # Compute absolute error between predicted and ground truth slope (x and y)
        absolute_error_x = np.abs(slope_x - gt_slope_x)
        absolute_error_y = np.abs(slope_y - gt_slope_y)

        # Compute average absolute error for slope, excluding borders
        average_absolute_error_x = np.mean(absolute_error_x[1:-1, 1:-1])
        print(average_absolute_error_x)
        average_absolute_error_y = np.mean(absolute_error_y[1:-1, 1:-1])
        print(average_absolute_error_y)

        # Compute angular error between predicted and ground truth normal vectors
        angle_error = np.arccos(np.sum(normal_vectors * gt_normal_vectors, axis=-1)) * (180 / torch.pi)
        average_angel_error = np.mean(angle_error[2:-2, 2:-2])
        print(average_angel_error)

        # Round average errors for display
        average_absolute_error_x = average_absolute_error_x.round(2)
        average_absolute_error_y = average_absolute_error_y.round(2)
        average_angel_error = average_angel_error.round(2)

        # Create visualizations for x-slope
        fig_x, ax1 = plt.subplots(1, 2)
        fig_y, ax2 = plt.subplots(1, 2)
        fig_n, ax3 = plt.subplots()

        fig_x.suptitle('X Gradient', fontsize=16)
        fig_x.text(0.5, 0.1, s='Average Error: ' + str(average_absolute_error_x), va='bottom', ha='center', fontsize=14)
        ax1[0].matshow(slope_x.round(4), cmap='viridis')
        ax1[0].set_title("Sobel Gradient")
        ax1[1].matshow(gt_slope_x.round(4), cmap='viridis')
        ax1[1].set_title("Ground Truth Gradient")

        # Create visualizations for y-slope
        fig_y.suptitle('Y Gradient', fontsize=16)
        fig_y.text(0.5, 0.1, s='Average Error: ' + str(average_absolute_error_x), va='bottom', ha='center', fontsize=14)
        ax2[0].matshow(slope_y.round(4), cmap='viridis')
        ax2[0].set_title("Sobel Gradient")
        ax2[1].matshow(gt_slope_y.round(4), cmap='viridis')
        ax2[1].set_title("Ground Truth Gradient")

        # Visualization of angular error in normal estimation
        fig_n.suptitle('Estimated Normal Error', fontsize=16)
        fig_n.text(0.5, 0.05, s='Average Error: ' + str(average_angel_error) + ' Degrees', va='bottom', ha='center', fontsize=14)
        ax3.matshow(angle_error.round(4), cmap='viridis')

        # Set text size based on input dimensions
        text_size = 10 + 2 * ((10 - slope_x.shape[0].item()) / 5)

        # Overlay numeric values for visual comparison
        for i in np.arange(slope_x.shape[0]):
            for j in np.arange(slope_x.shape[1]):
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

        # Save all visualizations to disk
        gradient_x_image_fp = os.path.join(base_dir, "{:08d}_gradient_x.png".format(idx))
        gradient_y_image_fp = os.path.join(base_dir, "{:08d}_gradient_y.png".format(idx))
        angular_error_image_fp = os.path.join(base_dir, "{:08d}_angular_error.png".format(idx))

        fig_x.savefig(gradient_x_image_fp)
        fig_y.savefig(gradient_y_image_fp)
        fig_n.savefig(angular_error_image_fp)

    def get_error_data(self, slope_x, gt_slope_x, slope_y, gt_slope_y, normal_vectors, gt_normal_vectors, buffer):

        absolute_error_x = np.abs(slope_x - gt_slope_x)
        absolute_error_y = np.abs(slope_y - gt_slope_y)

        average_absolute_error_x = np.mean(absolute_error_x[buffer:-buffer, buffer:-buffer])
        average_absolute_error_y = np.mean(absolute_error_y[buffer:-buffer, buffer:-buffer])

        angle_error = np.arccos(np.sum(normal_vectors * gt_normal_vectors, axis=-1)) * (180 / np.pi)
        average_angel_error = np.mean(angle_error[buffer:-buffer, buffer:-buffer])

        return average_absolute_error_x, average_absolute_error_y, average_angel_error

    def get_data_frame(self, data_dir):

        # Estimate number of samples in directory
        data_points = int(len(os.listdir(data_dir)) / 2)

        # Initialize data storage dictionary
        data = {
            'amplitude_x': [],
            'amplitude_y': [],
            'frequency_x': [],
            'frequency_y': [],
            'noise_range': [],
            'kernel_size': [],
            'average_absolute_error_x': [],
            'average_absolute_error_y': [],
            'average_angel_error': [],
        }

        for idx in range(data_points):
            # Filepaths for .npy and .yaml config
            bev_grid_fp = os.path.join(data_dir, "{:08d}_data.npy".format(idx))
            data_config_fp = os.path.join(data_dir, "{:08d}_data_config.yaml".format(idx))

            # Load BEV grid and YAML config
            bev_grid = np.load(bev_grid_fp)
            data_config = yaml.safe_load(open(data_config_fp))

            # Extract slope and normal vectors (predicted and ground truth)
            slope_x = bev_grid[..., 4]
            slope_y = bev_grid[..., 5]

            gt_slope_x = bev_grid[..., 7]
            gt_slope_y = bev_grid[..., 8]

            normal_vectors = np.stack((bev_grid[..., 9], bev_grid[..., 10], bev_grid[..., 11]), axis=-1)
            gt_normal_vectors = np.stack((bev_grid[..., 12], bev_grid[..., 13], bev_grid[..., 14]), axis=-1)

            # Compute mask buffer in pixels based on radius
            buffer = int(data_config['radius'] / 0.5)

            # Extract relevant metadata
            amplitude_x = data_config['amplitude_x']
            amplitude_y = data_config['amplitude_y']
            frequency_x = data_config['frequency_x']
            frequency_y = data_config['frequency_y']
            noise_range = data_config['noise_range']
            kernel_size = data_config['radius']

            # Compute errors
            average_absolute_error_x, average_absolute_error_y, average_angel_error = self.get_error_data(
                slope_x, gt_slope_x, slope_y, gt_slope_y, normal_vectors, gt_normal_vectors, buffer
            )

            # Append data to the lists
            data['amplitude_x'].append(amplitude_x)
            data['amplitude_y'].append(amplitude_y)
            data['frequency_x'].append(frequency_x)
            data['frequency_y'].append(frequency_y)
            data['noise_range'].append(noise_range)
            data['kernel_size'].append(kernel_size)
            data['average_absolute_error_x'].append(average_absolute_error_x)
            data['average_absolute_error_y'].append(average_absolute_error_y)
            data['average_angel_error'].append(average_angel_error)

        # Convert dictionary to DataFrame
        data_frame = pd.DataFrame(data)
        data_frame.fillna(0, inplace=True)  # Replace any NaNs with 0
        print(data_frame.info())

        return data_frame

    def viz_data(self, data_frame):

        # Unique values for each analysis dimension
        noise_ranges = data_frame['noise_range'].unique()
        kernel_sizes = data_frame['kernel_size'].unique()

        segmented_data = {}
        data_analysis = {}

        for range in noise_ranges:
            # Filter data for current noise level
            data_by_noise = data_frame[data_frame['noise_range'] == range]

            kernel_segmented_data = {}
            kernel_segmented_data_analysis = {}

            for size in kernel_sizes:
                # Further segment data by kernel size
                data_by_kernel = data_by_noise[data_by_noise['kernel_size'] == size]
                kernel_segmented_data[size] = data_by_kernel

                # Compute mean and standard deviation of errors
                mean = data_by_kernel[['average_absolute_error_x', 'average_absolute_error_y', 'average_angel_error']].mean().round(4)
                std = data_by_kernel[['average_absolute_error_x', 'average_absolute_error_y', 'average_angel_error']].std().round(4)

                kernel_segmented_data_analysis[size] = {'mean': mean, 'std': std}

            segmented_data[range] = kernel_segmented_data
            data_analysis[range] = kernel_segmented_data_analysis

        print(data_analysis)

if __name__ == '__main__':

    viz = NormalTesterVisualizer()
    data_dir = '/home/tartandriver/rosbags/normals_test_kitti/data'
    data_frame_fp = '/home/tartandriver/rosbags/normals_test_kitti/data_fame.yaml'

    # Generate data frame from raw inputs
    data_frame = viz.get_data_frame(data_dir)

    # Optional: save to YAML for later use
    # data_frame_dict = data_frame.to_dict()
    # yaml.dump(data_frame_dict, open(data_frame_fp, 'w'), default_flow_style=False)

    # Optional: reload from YAML
    # data_frame_yaml = yaml.safe_load(open(data_frame_fp,'r'))
    # data_frame = pd.DataFrame(data_frame_yaml)

    # Analyze and print results
    viz.viz_data(data_frame)
