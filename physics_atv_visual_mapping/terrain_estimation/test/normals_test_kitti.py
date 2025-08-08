import os
import yaml
import copy
import torch
import numpy as np
import itertools

from physics_atv_visual_mapping.terrain_estimation.test.normals_test import NormalsTester
from physics_atv_visual_mapping.localmapping.metadata import LocalMapperMetadata

class NormalTesterKitti():
    
    def create_noise_layers(metadata, noise_ranges, base_dir):
        noise_layers = []

        # Output filepath for all generated noise layers
        noise_layers_fp = os.path.join(base_dir, "noise_layers.npy")
        
        for range in noise_ranges:
            # Create a noise layer scaled by the given noise range
            noise_layer = (np.random.rand(metadata.N[0], metadata.N[1]) - 0.5) * (range / 100)
            noise_layers.append(noise_layer)

        # Stack all noise layers along the last dimension: shape (H, W, num_ranges)
        noise_layers = np.stack(noise_layers, axis=-1)

        # Save noise layers to disk
        np.save(noise_layers_fp, noise_layers)

    def create_wave_configs(frequencys_x, frequencys_y, amplitudes_x, amplitudes_y, noise_layer_keys, base_dir):
        # Cartesian product of all combinations
        combinations = itertools.product(frequencys_x, frequencys_y, amplitudes_x, amplitudes_y, noise_layer_keys)

        for idx, combo in enumerate(combinations):
            wave_config = {
                'frequency_x': float(combo[0]),
                'frequency_y': float(combo[1]),
                'amplitude_x': float(combo[2]),
                'amplitude_y': float(combo[3]),
                'h_shift_x': 0,   # horizontal shift (not varied)
                'h_shift_y': 0,
                'noise_range': float(combo[4])
            }

            # Save wave config YAML
            wave_config_fp = os.path.join(base_dir, "{:08d}_wave_config.yaml".format(idx))
            yaml.dump(wave_config, open(wave_config_fp, 'w'), default_flow_style=False)

    def create_kernel_configs(radius, kernel_type, base_dir):
        combinations = itertools.product(radius, kernel_type)

        for idx, combo in enumerate(combinations):
            kernel_config = {
                'radius': combo[0],
                'kernel_type': combo[1],
            }

            # Save kernel config YAML
            kernel_config_fp = os.path.join(base_dir, "{:08d}_kernel_config.yaml".format(idx))
            yaml.dump(kernel_config, open(kernel_config_fp, 'w'), default_flow_style=False)

    def create_gridmap_data(metadata, feature_keys, wave_configs_dir, kernel_configs_dir, noise_layers_fp, noise_ranges, base_dir):
        # Get list of wave config files
        wave_config_files = os.listdir(wave_configs_dir)
        wave_configs_fp = [os.path.join(wave_configs_dir, config) for config in wave_config_files]

        # Get list of kernel config files
        kernel_configs_files = os.listdir(kernel_configs_dir)
        kernel_configs_fp = [os.path.join(kernel_configs_dir, config) for config in kernel_configs_files]

        # Load pre-generated noise layers
        noise_layers = np.load(noise_layers_fp)

        # Cartesian product of wave configs and kernel configs
        combinations = itertools.product(wave_configs_fp, kernel_configs_fp)

        for idx, combo in enumerate(combinations):
            wave_config_fp = combo[0]
            kernel_config_fp = combo[1]

            # Load wave and kernel config from YAML
            wave_config = yaml.safe_load(open(wave_config_fp, 'r'))
            kernel_config = yaml.safe_load(open(kernel_config_fp, 'r'))

            # Get the noise layer corresponding to the wave config's noise_range
            terrain_noise = noise_layers[..., noise_ranges.index(wave_config['noise_range'])]
            terrain_noise = np.array(terrain_noise)

            # Generate BEV grid with synthetic terrain and run normal estimation
            normals_tester = NormalsTester()
            bev_grid = normals_tester.get_wave_bevgrid(metadata, feature_keys, wave_config, kernel_config, terrain_noise)

            # Save generated data as .npy file
            data_fp = os.path.join(base_dir, "{:08d}_data.npy".format(idx))
            data = bev_grid.data.cpu().numpy()
            np.save(data_fp, data)

            # Combine wave + kernel config into one metadata file for this dataset
            data_config = {}
            data_config.update(wave_config)
            data_config.update(kernel_config)

            # Save metadata config as YAML
            data_config_fp = os.path.join(base_dir, "{:08d}_data_config.yaml".format(idx))
            yaml.dump(data_config, open(data_config_fp, 'w'), default_flow_style=False)

if __name__ == '__main__':
    
    # Define grid metadata: origin, length, and resolution
    metadata = LocalMapperMetadata([-10, -10], [20, 20], [0.5, 0.5])

    # Define all the feature keys used in the BEV grid (mask, slopes, normals, etc.)
    feature_keys = [
        'mask', 'bev_x', 'bev_y', 'terrain',
        'slope_x', 'slope_y', 'slope',
        'gt_slope_x', 'gt_slope_y',
        'normals_x', 'normals_y', 'normals_z',
        'gt_normals_x', 'gt_normals_y', 'gt_normals_z'
    ]

    # Define file paths for outputs
    noise_layers_dir = '/home/tartandriver/rosbags/normals_test_kitti'
    wave_configs_dir = '/home/tartandriver/rosbags/normals_test_kitti/wave_configs'
    kernel_configs_dir = '/home/tartandriver/rosbags/normals_test_kitti/kernel_configs'
    data_dir = '/home/tartandriver/rosbags/normals_test_kitti/data'

    # List of noise magnitudes to generate
    noise_ranges = [5, 10, 20]

    # Create noise layers 
    NormalTesterKitti.create_noise_layers(metadata, noise_ranges, noise_layers_dir)

    # Define parameter sweeps for synthetic terrain generation
    frequencys_x = np.linspace(0, 1, 11)
    frequencys_y = np.linspace(0, 1, 11)
    amplitudes_x = np.linspace(0, 1, 11)
    amplitudes_y = np.linspace(0, 1, 11)

    # Path to saved noise layers
    noise_layers_fp = '/home/tartandriver/rosbags/normals_test_kitti/noise_layers.npy'

    # Generate wave config YAMLs
    NormalTesterKitti.create_wave_configs(frequencys_x, frequencys_y, amplitudes_x, amplitudes_y, noise_ranges, wave_configs_dir)

    # Generate kernel config YAMLs (e.g., Scharr filter with radius 0.5)
    NormalTesterKitti.create_kernel_configs([0.5,1,1.5], ['sobel'], kernel_configs_dir)

    # Generate the actual data samples for all combinations
    NormalTesterKitti.create_gridmap_data(
        metadata, feature_keys,
        wave_configs_dir, kernel_configs_dir,
        noise_layers_fp, noise_ranges,
        data_dir
    )
