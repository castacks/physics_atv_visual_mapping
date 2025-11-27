import torch
import numpy as np
import argparse

"""
Computes PCA from feature image samples
"""

def compute_pca_from_descs(args):
    # Load and reshape descs
    descs = np.load(args.desc_fp)
    descs = descs.reshape(-1, descs.shape[-1])
    descs = torch.from_numpy(descs).cuda()

    # Center data
    feat_mean = descs.mean(dim=0)
    descs_norm = descs - feat_mean.unsqueeze(0)

    # Compute PCA
    U, S, V = torch.pca_lowrank(descs_norm, q=args.k)

    # Save
    base_label = "pca"
    base_metainfo = "bilinear_anythermal"
    pca_res = {"mean": feat_mean.cpu(), "V": V.cpu(), "base_label": base_label, "base_metainfo": base_metainfo}
    torch.save(pca_res, args.out_fp)

    return pca_res

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--desc_fp', type=str, required=True, help='path to descriptors file')
    parser.add_argument('--out_fp', type=str, required=True, help='path to save computed PCA to')
    parser.add_argument('--k', type=int, required=True, help='number of PCA components')
    args = parser.parse_args()

    pca_model = compute_pca_from_descs(args)