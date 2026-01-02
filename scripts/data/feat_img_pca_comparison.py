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
def get_image_proc_pipeline(img_proc_cfg_fp, device='cpu'):
    image_pipeline_config = yaml.safe_load(open(img_proc_cfg_fp, 'r'))
    config = {
        'image_processing': image_pipeline_config,
        'models_dir': os.environ['TARTANDRIVER_MODELS_DIR'],
        'device': device
    }
    pipeline = setup_image_pipeline(config)
    return pipeline

def run_pipeline(pipeline, modality, img_dir, frame_idx, intrinsics):
    img_fp = os.path.join(img_dir, '{:08d}.png'.format(frame_idx))
    img = cv2.imread(os.path.join(img_dir, img_fp))
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB) / 255.0
    # B x C x H X W
    img = torch.tensor(img).unsqueeze(0).permute(0, 3, 1, 2)

    if modality == "rgb":
        # load intrinsics from directory
        intrinsics_in = np.loadtxt(os.path.join(intrinsics, "{:08d}.txt".format(frame_idx)), dtype=np.float64)
        intrinsics_in = torch.from_numpy(intrinsics_in).reshape(1,3,3)
    else:
        # thermal intrinsics are hardcoded
        intrinsics_in = torch.from_numpy(intrinsics).reshape(1,3,3)

    # feat_img is 1xdimximg_hwximg_hw
    feat_img, out_intrinsics = pipeline.run(img, intrinsics_in)
    return feat_img, out_intrinsics

def compute_feat_img_buffer(modality, img_dir, intrinsics, num_frames, img_proc_cfg_fp, device='cpu'):
    pipeline = get_image_proc_pipeline(img_proc_cfg_fp, device)

    img_hw = 224
    # dinov2s
    dim = 384

    print(f"Modality: {modality}")
    for ii in range(num_frames):
        if ii % 100 == 0:
            print(f"Frame {ii}/{num_frames}")
        run_pipeline(pipeline, modality, img_dir, ii, intrinsics)

def compute_feature_buffers(args, rgb_img_proc_fp, thermal_img_proc_fp):
    num_frames = 2831

    rgb_img_dir = os.path.join(args.run_dir, 'rgb_image')
    rgb_intrinsics_dir = os.path.join(args.run_dir, 'rgb_raw_image_intrinsics')
    
    thermal_img_dir = os.path.join(args.run_dir, 'thermal_left_processed')
    thermal_intrinsics = np.array([[412.42744452, 0.0, 313.38643993,0.0, 412.60673097, 249.37501763, 0.0, 0.0, 1.0]])
    
    compute_feat_img_buffer('rgb', rgb_img_dir, rgb_intrinsics_dir, num_frames, rgb_img_proc_fp)
    compute_feat_img_buffer('thermal', thermal_img_dir, thermal_intrinsics, num_frames, thermal_img_proc_fp)

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
    pca_res = {"mean": feat_mean.cpu(), "V": V.cpu(), "base_label": base_label, "base_metainfo": base_metainfo, "k": k}
    print(f"Saving pca for {modality} to {out_fp}")
    torch.save(pca_res, out_fp)

def compute_pcas(args, rgb_buffer_fp, rgb_pca_out_fp, thermal_buffer_fp, thermal_pca_out_fp, combined_pca_out_fp):
    # Load buffers
    rgb_buffer = np.load(rgb_buffer_fp)
    thermal_buffer = np.load(thermal_buffer_fp)

    # print(f"rgb buffer shape: {rgb_buffer.shape}")
    # print(f"thermal buffer shape: {thermal_buffer.shape}")

    # Compute PCAs
    rgb_feats = rgb_buffer.reshape(-1, rgb_buffer.shape[-1])
    # print(f"rgb feats shape: {rgb_feats.shape}")
    compute_pca(rgb_feats, args.k, 'rgb', rgb_pca_out_fp)

    thermal_feats = thermal_buffer.reshape(-1, thermal_buffer.shape[-1])
    # print(f"thermal feats shape: {thermal_feats.shape}")
    compute_pca(thermal_feats, args.k, 'thermal', thermal_pca_out_fp)
    
    combined_buffer = np.concatenate((rgb_buffer, thermal_buffer), axis=1)
    combined_feats = combined_buffer.reshape(-1, combined_buffer.shape[-1])
    # print(f"combined buffer shape: {combined_buffer.shape}")
    # print(f"combined feats shape: {combined_feats.shape}")
    compute_pca(combined_feats, args.k, 'combined', combined_pca_out_fp)

def get_rgb_img_for_frame(rgb_image_dir, ii):
    rgb_image_fp = os.path.join(rgb_image_dir, "{:08d}.png".format(ii))
    rgb_img = cv2.imread(rgb_image_fp, cv2.IMREAD_UNCHANGED)
    rgb_img = cv2.cvtColor(rgb_img, cv2.COLOR_BGR2RGB)
    return rgb_img

def get_thermal_img_for_frame(thermal_img_dir, ii):
    thermal_image_fp = os.path.join(thermal_img_dir, "{:08d}.png".format(ii))
    thermal_img = cv2.imread(thermal_image_fp, cv2.IMREAD_UNCHANGED)
    thermal_img = cv2.cvtColor(thermal_img, cv2.COLOR_BGR2RGB)
    return thermal_img

def get_normalized_image_feats(feat_img):
    # reshape from 1 x dim x h x w to (h*w) x dim
    dim = feat_img.shape[1]
    features = feat_img.squeeze(0).permute(1, 2, 0).reshape(-1, dim)

    # Center data
    feat_mean = features.mean(dim=0)
    feat_norm = features - feat_mean.unsqueeze(0)

    return feat_norm

def apply_pca(feat_img, pca):
    feat_norm = get_normalized_image_feats(feat_img)

    # (h*w) x dim -> (h*w) x k, where dim is DINO embedding dim and k is # of PCA components
    return feat_norm @ pca['V'].cuda()

# Normalize PCA to RGB range for visualization
def pca_to_rgb(pca_features):
    pca_rgb = (pca_features - pca_features.min(axis=0)) / \
              (pca_features.max(axis=0) - pca_features.min(axis=0))
    return pca_rgb

def compute_feature_norms(features):
    norms = torch.norm(features, p=2, dim=-1)
    return norms.cpu().numpy()

def compare_feature_norms(rgb_img, thermal_img, rgb_features, thermal_features, h, w):
    rgb_norms = compute_feature_norms(rgb_features)
    thermal_norms = compute_feature_norms(thermal_features)

    # Visualize first 3 components of PCA as image
    rgb_pca_img = pca_to_rgb(rgb_features.cpu().numpy()).reshape(h, w, rgb_features.shape[-1])[:, :, :3]
    thermal_pca_img = pca_to_rgb(thermal_features.cpu().numpy()).reshape(h, w, thermal_features.shape[-1])[:, :, :3]

    # Statistical comparison
    print(f"RGB norm - mean: {rgb_norms.mean():.3f}, std: {rgb_norms.std():.3f}")
    print(f"Thermal norm - mean: {thermal_norms.mean():.3f}, std: {thermal_norms.std():.3f}")
    
    # Correlation between spatial norm patterns
    correlation = np.corrcoef(rgb_norms.flatten(), thermal_norms.flatten())[0,1]
    print(f"Spatial norm correlation: {correlation:.3f}")
    
    fig, axes = plt.subplots(3, 3, figsize=(15, 5))
    # Original images
    axes[0][0].imshow(rgb_img)
    axes[0][0].set_title("RGB Img")
    axes[0][1].imshow(thermal_img)
    axes[0][1].set_title("Thermal Img")
    # PCA images
    axes[1][0].imshow(rgb_pca_img)
    axes[1][0].set_title("RGB PCA Img")
    axes[1][1].imshow(thermal_pca_img)
    axes[1][1].set_title("Thermal PCA Img")
    # Visualize norm maps side-by-side
    im0 = axes[2][0].imshow(rgb_norms.reshape(h, w), cmap='viridis')
    axes[2][0].set_title('RGB Feature Norms')
    fig.colorbar(im0, ax=axes[2][0], label='RGB Norms')
    im1 = axes[2][1].imshow(thermal_norms.reshape(h, w), cmap='viridis')
    axes[2][1].set_title('Thermal Feature Norms')
    fig.colorbar(im1, ax=axes[2][1], label='Thermal Norms')
    im2 = axes[2][2].imshow(np.abs(rgb_norms - thermal_norms).reshape(h, w), cmap='hot')
    axes[2][2].set_title('RGB-Thermal Norm Difference')
    fig.colorbar(im2, ax=axes[2][2], label='RGB-Thermal Norm Difference')

    for ax in axes.flat:
        ax.axis('off')
    plt.show()

def compute_reconstruction_error(pca, rgb_feat_img, thermal_feat_img):
    # Apply PCA
    rgb_feats_proj = apply_pca(rgb_feat_img, pca)
    thermal_feats_proj = apply_pca(thermal_feat_img, pca)

    # Reconstruct original data
    rgb_feats_reconstruct = rgb_feats_proj @ pca['V'].cuda().T
    thermal_feats_reconstruct = thermal_feats_proj @ pca['V'].cuda().T

    # Compute reconstruction error as norm of original vs reconstructed data
    norm_rgb_feats_orig = get_normalized_image_feats(rgb_feat_img)
    norm_thermal_feats_orig = get_normalized_image_feats(thermal_feat_img)

    reconstruction_error_rgb = torch.linalg.norm(rgb_feats_reconstruct - norm_rgb_feats_orig)
    reconstruction_error_thermal = torch.linalg.norm(thermal_feats_reconstruct - norm_thermal_feats_orig)

    print(f"RGB error: {reconstruction_error_rgb} thermal error: {reconstruction_error_thermal}")

def verify_img_pipeline(args, rgb_img_proc_fp, rgb_intrinsics_dir, thermal_img_proc_fp, thermal_intrinsics):
    rgb_image_dir = os.path.join(args.run_dir, 'rgb_image')
    rgb_pipeline = get_image_proc_pipeline(rgb_img_proc_fp)
    rgb_feat_img, _ = run_pipeline(rgb_pipeline, 'rgb', rgb_image_dir, 10, rgb_intrinsics_dir)
    rgb_feat_img = rgb_feat_img.squeeze(0).permute(1, 2, 0)
    plt.imshow(rgb_feat_img[:, :, :3].cpu().numpy())
    plt.show()

    thermal_image_dir = os.path.join(args.run_dir, 'thermal_left_processed')
    thermal_pipeline = get_image_proc_pipeline(thermal_img_proc_fp)
    thermal_feat_img, _ = run_pipeline(thermal_pipeline, 'thermal', thermal_image_dir, 10, thermal_intrinsics)
    thermal_feat_img = thermal_feat_img.squeeze(0).permute(1, 2, 0)
    plt.imshow(thermal_feat_img[:, :, :3].cpu().numpy())
    plt.show()

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--run_dir', type=str, required=False, help='path to KITTI dataset')
    parser.add_argument('--k', type=int, required=False, help='number of PCA components')
    args = parser.parse_args()

    rgb_img_proc_fp = '/home/tartandriver/tartandriver_ws/src/perception/physics_atv_visual_mapping/config/image_processing/loftup_grayscale_rgb.yaml'
    rgb_intrinsics_dir = os.path.join(args.run_dir, 'rgb_raw_image_intrinsics')
    thermal_img_proc_fp = '/home/tartandriver/tartandriver_ws/src/perception/physics_atv_visual_mapping/config/image_processing/loftup_thermal.yaml'
    thermal_intrinsics = np.array([[412.42744452, 0.0, 313.38643993,0.0, 412.60673097, 249.37501763, 0.0, 0.0, 1.0]])

    # Compute feature buffers for rgb/thermal feature images
    # compute_feature_buffers(args, rgb_img_proc_fp, thermal_img_proc_fp)

    # 09/17 bag - 3692 frames, feature buffer from first 1840 frames
    # rgb_buffer_fp = '/home/tartandriver/workspace/feat_img_ablations/dinov2s/0917_bag/loftup_dinov2s_gray_rgb_224x224_feats.npy'
    # thermal_buffer_fp = '/home/tartandriver/workspace/feat_img_ablations/dinov2s/0917_bag/loftup_dinov2s_thermal_224x224_feats.npy'
    # rgb_pca_fp = '/home/tartandriver/workspace/feat_img_ablations/dinov2s/0917_bag/pca_loftup_dinov2s_gray_rgb_comp.pt'
    # thermal_pca_fp = '/home/tartandriver/workspace/feat_img_ablations/dinov2s/0917_bag/pca_loftup_dinov2s_thermal_comp.pt'
    # combined_pca_fp = '/home/tartandriver/workspace/feat_img_ablations/dinov2s/0917_bag/pca_loftup_dinov2s_combined_comp.pt'

    # 11/06 bag - 5657 frames, feature buffer from first 2830 frames
    rgb_buffer_fp = '/home/tartandriver/workspace/feat_img_ablations/dinov2s/1106_0_mac_salon_day_rgb/loftup_dinov2s_gray_rgb_224x224_feats.npy'
    thermal_buffer_fp = '/home/tartandriver/workspace/feat_img_ablations/dinov2s/1106_0_mac_salon_day_rgb/loftup_dinov2s_thermal_224x224_feats.npy'
    rgb_pca_fp = '/home/tartandriver/workspace/feat_img_ablations/dinov2s/1106_0_mac_salon_day_rgb/pca_loftup_dinov2s_gray_rgb_comp.pt'
    thermal_pca_fp = '/home/tartandriver/workspace/feat_img_ablations/dinov2s/1106_0_mac_salon_day_rgb/pca_loftup_dinov2s_thermal_comp.pt'
    combined_pca_fp = '/home/tartandriver/workspace/feat_img_ablations/dinov2s/1106_0_mac_salon_day_rgb/pca_loftup_dinov2s_combined_comp.pt'
    

    # Compute rgb, thermal and combined feature pcas based on rgb/thermal feature buffers
    # compute_pcas(args, rgb_buffer_fp, rgb_pca_fp, thermal_buffer_fp, thermal_pca_fp, combined_pca_fp)

    # Load PCAs
    rgb_pca = torch.load(rgb_pca_fp, weights_only=False)
    thermal_pca = torch.load(thermal_pca_fp, weights_only=False)
    combined_pca = torch.load(combined_pca_fp, weights_only=False)

    # Apply PCA to RGB/thermal frames to get features
    rgb_pipeline = get_image_proc_pipeline(rgb_img_proc_fp)
    thermal_pipeline = get_image_proc_pipeline(thermal_img_proc_fp)

    rgb_image_dir = os.path.join(args.run_dir, 'rgb_image')
    thermal_image_dir = os.path.join(args.run_dir, 'thermal_left_processed')

    frame_idx = 1496
    rgb_feat_img, _ = run_pipeline(rgb_pipeline, 'rgb', rgb_image_dir, frame_idx, rgb_intrinsics_dir)
    thermal_feat_img, _ = run_pipeline(thermal_pipeline, 'thermal', thermal_image_dir, frame_idx, thermal_intrinsics)

    rgb_pca_feats = apply_pca(rgb_feat_img, combined_pca)
    thermal_pca_feats = apply_pca(thermal_feat_img, combined_pca)

    # Compare Feature Norms
    h = 224
    w = 224
    rgb_img = get_rgb_img_for_frame(rgb_image_dir, frame_idx)
    thermal_img = get_thermal_img_for_frame(thermal_image_dir, frame_idx)
    compare_feature_norms(rgb_img, thermal_img, rgb_pca_feats, thermal_pca_feats, h, w)

    # Compute reconstruction errors
    compute_reconstruction_error(rgb_pca, rgb_feat_img, thermal_feat_img)
    compute_reconstruction_error(thermal_pca, rgb_feat_img, thermal_feat_img)
    compute_reconstruction_error(combined_pca, rgb_feat_img, thermal_feat_img)

    # Verifying img pipeline
    # verify_img_pipeline(args, rgb_img_proc_fp, rgb_intrinsics_dir, thermal_img_proc_fp, thermal_intrinsics)