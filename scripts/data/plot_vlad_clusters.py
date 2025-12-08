import numpy as np
import torch
import argparse
from scipy.spatial.distance import cdist
import matplotlib.pyplot as plt

def plot_pairwise_distances(centers1, centers1_name, centers2, centers2_name):
    # pairwise distances between cluster centers
    dists = cdist(centers1, centers2, metric='euclidean')

    # plot as heatmap
    plt.figure(figsize=(6,5))
    plt.imshow(dists, cmap='viridis')
    plt.colorbar(label='Distance')
    plt.xlabel(f"{centers2_name} centroids")
    plt.ylabel(f"{centers1_name} centroids")
    plt.title("Centroid Euclidean Distances")
    plt.xticks(np.arange(centers2.shape[0]))
    plt.yticks(np.arange(centers1.shape[0]))
    plt.show()

def center_data(centers):
    centers = centers.reshape(-1, centers.shape[-1])
    centers_mean = centers.mean(dim=0)
    centers_norm = centers - centers_mean.unsqueeze(0)

    return centers_norm

def plot_pca_spaces(centers1, centers2, centers3, centers4):
    # Center data
    centers1_norm = center_data(centers1)
    centers2_norm = center_data(centers2)
    centers3_norm = center_data(centers3)
    centers4_norm = center_data(centers4)

    # 2 component PCAs for visualization
    U1, S1, V1 = torch.pca_lowrank(centers1_norm, q=2)
    U2, S2, V2 = torch.pca_lowrank(centers2_norm, q=2)
    U3, S3, V3 = torch.pca_lowrank(centers3_norm, q=2)
    U4, S4, V4 = torch.pca_lowrank(centers4_norm, q=2)

    # Project centers onto PCA directions
    centers1_proj = (centers1_norm @ V1).cpu()
    centers2_proj = (centers2_norm @ V2).cpu()
    centers3_proj = (centers3_norm @ V3).cpu()
    centers4_proj = (centers4_norm @ V3).cpu()

    # Plot
    plt.scatter(centers1_proj[:, 0], centers1_proj[:, 1], label=centers1_name)
    plt.scatter(centers2_proj[:, 0], centers2_proj[:, 1], label=centers2_name)
    plt.scatter(centers3_proj[:, 0], centers3_proj[:, 1], label=centers3_name)
    plt.scatter(centers4_proj[:, 0], centers4_proj[:, 1], label=centers4_name)
    plt.legend()
    plt.show()

"""
Plots two or three sets of VLAD clusters for alignment comparison
"""

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--cluster1_fp', type=str, required=True, help='path to cluster centers file 1')
    parser.add_argument('--cluster2_fp', type=str, required=True, help='path to cluster centers file 2')
    parser.add_argument('--cluster3_fp', type=str, required=True, help='path to cluster centers file 3')
    parser.add_argument('--cluster4_fp', type=str, required=True, help='path to cluster centers file 4')
    parser.add_argument('--out_fp', type=str, required=False, help='path to save plot to')
    args = parser.parse_args()

    file1 = args.cluster1_fp
    file2 = args.cluster2_fp
    file3 = args.cluster3_fp
    file4 = args.cluster4_fp
    out_fp = None
    if args.out_fp:
        out_fp = args.out_fp

    # for loading cluster centers
    # centers1 = torch.load(file1).cuda()
    # centers2 = torch.load(file2).cuda()
    # centers3 = torch.load(file3).cuda()
    # centers4 = torch.load(file4).cuda()

    # for loading descriptors
    centers1 = np.load(file1)
    centers2 = np.load(file2)
    centers3 = np.load(file3)
    centers4 = np.load(file4)

    centers1 = torch.from_numpy(centers1).cuda()
    centers2 = torch.from_numpy(centers2).cuda()
    centers3 = torch.from_numpy(centers3).cuda()
    centers4 = torch.from_numpy(centers4).cuda()

    # for vlad cluster centers
    # centers1_name = file1.split("/")[7][11:]
    # centers2_name = file2.split("/")[7][11:]
    # centers3_name = file3.split("/")[7][11:]
    # centers4_name = file4.split("/")[7][11:]

    # for descriptors
    centers1_name = file1.split("/")[-1][:-4]
    centers2_name = file2.split("/")[-1][:-4]
    centers3_name = file3.split("/")[-1][:-4]
    centers4_name = file4.split("/")[-1][:-4]


    plot_pca_spaces(centers1, centers2, centers3, centers4)