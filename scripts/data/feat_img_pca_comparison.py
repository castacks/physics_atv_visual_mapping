import numpy as np
import matplotlib.pyplot as plt
import argparse
import yaml
import os
import cv2
import torch
from scipy.spatial import procrustes

from physics_atv_visual_mapping.image_processing.image_pipeline import setup_image_pipeline 

"""
Based on compute_pca script but for more specific usage of DINO feature image comaprison
"""

def save_feat_img_buffer(modality, img_dir, intrinsics, num_frames, img_proc_cfg_fp, device='cpu'):
    image_pipeline_config = yaml.safe_load(open(img_proc_cfg_fp, 'r'))
    config = {
        'image_processing': image_pipeline_config,
        'models_dir': os.environ['TARTANDRIVER_MODELS_DIR'],
        'device': device
    }
    pipeline = setup_image_pipeline(config)

    img_hw = 224
    # dinov2s
    dim = 384

    print(f"Modality: {modality}")
    for ii in range(num_frames):
        if ii % 100 == 0:
            print(f"Frame {ii}/{num_frames}")
        img_fp = os.path.join(img_dir, '{:08d}.png'.format(ii))
        img = cv2.imread(os.path.join(img_dir, img_fp))
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB) / 255.0
        # B x C x H X W
        img = torch.tensor(img).unsqueeze(0).permute(0, 3, 1, 2)

        if modality == "rgb":
            # load intrinsics from directory
            intrinsics_in = np.loadtxt(os.path.join(intrinsics, "{:08d}.txt".format(ii)), dtype=np.float64)
            intrinsics_in = torch.from_numpy(intrinsics_in).reshape(1,3,3)
        else:
            # thermal intrinsics are hardcoded
            intrinsics_in = torch.from_numpy(intrinsics).reshape(1,3,3)

        # feat_img is 1xdimximg_hwximg_hw
        feat_img, _ = pipeline.run(img, intrinsics_in)

def compute_pca(img_feats, k, modality, out_fp):
    feats = torch.from_numpy(img_feats).cuda()

    # Center data
    feat_mean = feats.mean(dim=0)
    feats_norm = feats - feat_mean.unsqueeze(0)

    # Compute PCA
    U, S, V = torch.pca_lowrank(feats_norm, q=k)

    # Save
    base_label = "pca"
    base_metainfo = modality
    pca_res = {"mean": feat_mean.cpu(), "V": V.cpu(), "base_label": base_label, "base_metainfo": base_metainfo}
    print(f"Saving pca for {modality} to {out_fp}")
    torch.save(pca_res, out_fp)

def compute_feature_norms(features):
    norms = torch.norm(features, p=2, dim=-1)
    return norms.cpu().numpy()

def compare_feature_norms(rgb_features, thermal_features):
    rgb_norms = compute_feature_norms(rgb_features)
    thermal_norms = compute_feature_norms(thermal_features)

    print(rgb_norms.shape)
    print(thermal_norms.shape)
    
    h = 10
    w = 10

    # Statistical comparison
    print(f"RGB norm - mean: {rgb_norms.mean():.3f}, std: {rgb_norms.std():.3f}")
    print(f"Thermal norm - mean: {thermal_norms.mean():.3f}, std: {thermal_norms.std():.3f}")
    
    # Correlation between spatial norm patterns
    correlation = np.corrcoef(rgb_norms.flatten(), thermal_norms.flatten())[0,1]
    print(f"Spatial norm correlation: {correlation:.3f}")
    
    # Visualize norm maps side-by-side
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    axes[0].imshow(rgb_norms[0, :].reshape(h, w), cmap='viridis')
    axes[0].set_title('RGB Feature Norms')
    axes[1].imshow(thermal_norms[0, :].reshape(h, w), cmap='viridis')
    axes[1].set_title('Thermal Feature Norms')
    axes[2].imshow(np.abs(rgb_norms[0, :] - thermal_norms[0, :]).reshape(h, w), cmap='hot')
    axes[2].set_title('Absolute Norm Difference')
    for ax in axes:
        ax.axis('off')
    plt.show()

def procrustes_analysis(rgb_pca, thermal_pca, rgb_feats, thermal_feats):
    rgb_feats = rgb_feats.reshape(-1, rgb_feats.shape[-1])
    rgb_feats = torch.from_numpy(rgb_feats)

    thermal_feats = thermal_feats.reshape(-1, thermal_feats.shape[-1])
    thermal_feats = torch.from_numpy(thermal_feats)

    # Center data
    rgb_feat_mean = rgb_feats.mean(dim=0)
    rgb_feats_norm = rgb_feats - rgb_feat_mean.unsqueeze(0)

    thermal_feat_mean = thermal_feats.mean(dim=0)
    thermal_feats_norm = thermal_feats - thermal_feat_mean.unsqueeze(0)

    # Apply PCA
    rgb_feats_proj = rgb_feats_norm @ rgb_pca['V']
    thermal_feats_proj = thermal_feats_norm @ thermal_pca['V']

    # Compute rotation
    mtx1, mt2, disparity = procrustes(rgb_feats_proj, thermal_feats_proj)

    print(f"Disparity: {disparity}")


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--run_dir', type=str, required=False, help='path to KITTI dataset')
    parser.add_argument('--k', type=int, required=False, help='number of PCA components')
    args = parser.parse_args()

    # num_frames = 1846

    # rgb_img_proc_fp = '/home/tartandriver/tartandriver_ws/src/perception/physics_atv_visual_mapping/config/image_processing/loftup_grayscale_rgb.yaml'
    # rgb_img_dir = os.path.join(args.run_dir, 'rgb_image')
    # rgb_intrinsics_dir = os.path.join(args.run_dir, 'rgb_raw_image_intrinsics')
    
    # thermal_img_proc_fp = '/home/tartandriver/tartandriver_ws/src/perception/physics_atv_visual_mapping/config/image_processing/loftup_thermal.yaml'
    # thermal_img_dir = os.path.join(args.run_dir, 'thermal_left_processed')
    # thermal_intrinsics = np.array([[412.42744452, 0.0, 313.38643993,0.0, 412.60673097, 249.37501763, 0.0, 0.0, 1.0]])
    
    # save_feat_img_buffer('rgb', rgb_img_dir, rgb_intrinsics_dir, num_frames, rgb_img_proc_fp)
    # save_feat_img_buffer('thermal', thermal_img_dir, thermal_intrinsics, num_frames, thermal_img_proc_fp)

    # Load buffer
    rgb_buffer_fp = '/home/tartandriver/tartandriver_ws/loftup_dinov2s_gray_rgb_224x224_feats.npy'
    rgb_buffer = np.load(rgb_buffer_fp)

    thermal_buffer_fp = '/home/tartandriver/tartandriver_ws/loftup_dinov2s_thermal_224x224_feats.npy'
    thermal_buffer = np.load(thermal_buffer_fp)

    # print(f"rgb buffer shape: {rgb_buffer.shape}")
    # print(f"thermal buffer shape: {thermal_buffer.shape}")

    # Compute PCAs
    rgb_pca_out_fp = '/home/tartandriver/tartandriver_ws/models/physics_atv_visual_mapping/pca/pca_loftup_dinov2s_gray_rgb_comp.pt'
    # rgb_feats = rgb_buffer.reshape(-1, rgb_buffer.shape[-1])
    # print(f"rgb feats shape: {rgb_feats.shape}")
    # compute_pca(rgb_feats, args.k, 'rgb', rgb_pca_out_fp)

    thermal_pca_out_fp = '/home/tartandriver/tartandriver_ws/models/physics_atv_visual_mapping/pca/pca_loftup_dinov2s_thermal_comp.pt'
    # thermal_feats = thermal_buffer.reshape(-1, thermal_buffer.shape[-1])
    # print(f"thermal feats shape: {thermal_feats.shape}")
    # compute_pca(thermal_feats, args.k, 'thermal', thermal_pca_out_fp)
    
    combined_pca_out_fp = '/home/tartandriver/tartandriver_ws/models/physics_atv_visual_mapping/pca/pca_loftup_dinov2s_combined_comp.pt'
    # combined_buffer = np.concatenate((rgb_feats, thermal_feats), axis=0)
    # combined_feats = combined_buffer.reshape(-1, combined_buffer.shape[-1])
    # print(f"combined feats shape: {combined_feats.shape}")
    # compute_pca(combined_feats, args.k, 'combined', combined_pca_out_fp)

    # Compare Feature Norms
    # rgb_buffer = torch.from_numpy(rgb_buffer)
    # thermal_buffer = torch.from_numpy(thermal_buffer)
    # compare_feature_norms(rgb_buffer, thermal_buffer)

    # Compute rotation matrices
    rgb_pca = torch.load(rgb_pca_out_fp, weights_only=False)
    thermal_pca = torch.load(thermal_pca_out_fp, weights_only=False)
    combined_pca = torch.load(combined_pca_out_fp, weights_only=False)

    procrustes_analysis(rgb_pca, thermal_pca, rgb_buffer, thermal_buffer)