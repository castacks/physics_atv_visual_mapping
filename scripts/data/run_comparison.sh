#!/bin/bash

base_dir="/home/tartandriver/tartandriver_ws/models/physics_atv_visual_mapping/dino_clusters"
ws_dir="/home/tartandriver/workspace/img_feat_descs"

# For plotting as heatmap

# comparison set 1
# centers1="8_clusters_dinov2b_16x16/c_centers.pt"
# centers2="8_clusters_dinov2b_thermal_16x16/c_centers.pt"
# centers2="8_clusters_anythermal_16x16/c_centers.pt"

# comparison set 2
# centers1="8_clusters_bilinear_dinov2b_224x224/c_centers.pt"
# centers2="8_clusters_bilinear_dinov2b_thermal_224x224/c_centers.pt"
# centers2="8_clusters_bilinear_anythermal_224x224/c_centers.pt"

# python3 plot_vlad_clusters.py --cluster1_fp $base_dir/$centers1 --cluster2_fp $base_dir/$centers2

# For plotting PCAs
# comparison set 1
# centers1="8_clusters_dinov2b_16x16/c_centers.pt"
# centers2="8_clusters_dinov2b_thermal_16x16/c_centers.pt"
# centers3="8_clusters_anythermal_16x16/c_centers.pt"
# centers4="8_clusters_dinov2b_grayscale_16x16/c_centers.pt"

# comparison set 2
# centers1="8_clusters_bilinear_dinov2b_224x224/c_centers.pt"
# centers2="8_clusters_bilinear_dinov2b_thermal_224x224/c_centers.pt"
# centers3="8_clusters_bilinear_anythermal_224x224/c_centers.pt"
# centers4="8_clusters_bilinear_dinov2b_grayscale_224x224/c_centers.pt"

# comparison set 3
# centers1="dinov2b_rgb_16x16_feats.npy"
# centers2="dinov2b_grayscale_rgb_16x16_feats.npy"
# centers3="dinov2b_thermal_16x16_feats.npy"
# centers4="anythermal_thermal_16x16_feats.npy"

# comparison set 4
centers1="bilinear_dinov2b_rgb_224x224_feats.npy"
centers2="bilinear_dinov2b_grayscale_rgb_224x224_feats.npy"
centers3="bilinear_dinov2b_thermal_224x224_feats.npy"
centers4="bilinear_anythermal_thermal_224x224_feats.npy"

# python3 plot_vlad_clusters.py --cluster1_fp $base_dir/$centers1 \
#     --cluster2_fp $base_dir/$centers2 \
#     --cluster3_fp $base_dir/$centers3 \
#     --cluster4_fp  $base_dir/$centers4

python3 plot_vlad_clusters.py --cluster1_fp $ws_dir/$centers1 \
    --cluster2_fp $ws_dir/$centers2 \
    --cluster3_fp $ws_dir/$centers3 \
    --cluster4_fp  $ws_dir/$centers4