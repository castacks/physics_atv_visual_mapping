import os
import time
import argparse

import cv2
import torch
import numpy as np
import matplotlib.pyplot as plt

from physics_atv_visual_mapping.image_processing.anyloc_utils import DinoV2ExtractFeatures
from physics_atv_visual_mapping.utils import normalize_dino

def viz_dino_features(img, dino_img, dino):
    n_layers = len(dino.layers)

    fig, axs = plt.subplots(n_layers + 2, 4, figsize=(20, 20))

    axs[0, 0].imshow(img.cpu().numpy())
    axs[0, 0].set_title('PCA 1-3')
    axs[0, 1].set_title('PCA 4-6')
    axs[0, 2].set_title('PCA 7-9')
    axs[0, 3].set_title('PCA res')

    feats_per_layer = int(dino_img.shape[-1] / len(dino.layers))
    for li, layer in enumerate(dino.layers):
        layer_img = dino_img[..., li*feats_per_layer:(li+1)*feats_per_layer]
        layer_feats = layer_img.view(-1, feats_per_layer)
        feat_mean = layer_feats.mean(dim=0, keepdim=True)
        layer_feats = layer_feats - feat_mean

        U, S, V = torch.pca_lowrank(layer_feats, q=32)
        layer_feats_proj = layer_feats @ V[:, :9]

        layer_feats_res = torch.sqrt(torch.linalg.norm(layer_feats, dim=-1)**2 - torch.linalg.norm(layer_feats_proj, dim=-1)**2)

        pca_img = layer_feats_proj.view(layer_img.shape[0], layer_img.shape[1], 9)
        pca_res_img = layer_feats_res.view(layer_img.shape[0], layer_img.shape[1])

        axs[li + 1, 0].imshow(normalize_dino(pca_img[..., :3]).cpu().numpy())
        axs[li + 1, 1].imshow(normalize_dino(pca_img[..., 3:6]).cpu().numpy())
        axs[li + 1, 2].imshow(normalize_dino(pca_img[..., 6:]).cpu().numpy())
        axs[li + 1, 3].imshow(pca_res_img.cpu().numpy())
        # axs[li + 1, 0].set_ylabel('Layer {}'.format(layer))

    layer_img = dino_img
    layer_feats = layer_img.view(-1, layer_img.shape[-1])
    feat_mean = layer_feats.mean(dim=0, keepdim=True)
    layer_feats = layer_feats - feat_mean

    U, S, V = torch.pca_lowrank(layer_feats, q=32)
    layer_feats_proj = layer_feats @ V[:, :9]
    pca_img = layer_feats_proj.view(layer_img.shape[0], layer_img.shape[1], 9)


    layer_feats_res = torch.sqrt(torch.linalg.norm(layer_feats, dim=-1)**2 - torch.linalg.norm(layer_feats_proj, dim=-1)**2)
    pca_res_img = layer_feats_res.view(layer_img.shape[0], layer_img.shape[1])

    axs[-1, 0].imshow(normalize_dino(pca_img[..., :3]).cpu().numpy())
    axs[-1, 1].imshow(normalize_dino(pca_img[..., 3:6]).cpu().numpy())
    axs[-1, 2].imshow(normalize_dino(pca_img[..., 6:]).cpu().numpy())
    axs[-1, 3].imshow(pca_res_img.cpu().numpy())

    # axs[-1, 0].set_ylabel('all layer PCA')

    plt.show()

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--run_dir', type=str, required=True, help='path to data dir')
    parser.add_argument('--image_dir', type=str, required=False, default='image_left_color')
    parser.add_argument('--models_dir', type=str, required=False, default='/home/tartandriver/tartandriver_ws/models')
    args = parser.parse_args()

    img_dir = os.path.join(args.run_dir, args.image_dir)
    img_fps = os.listdir(img_dir)

    np.random.shuffle(img_fps)

    torch.hub.set_dir(os.path.join(args.models_dir, "torch_hub"))
    dino_dir = os.path.join(args.models_dir, "torch_hub", "facebookresearch_dinov2_main")
    radio_dir = os.path.join(args.models_dir, "torch_hub", "NVlabs_RADIO_main")

    #vitg
    # dino_args = {
    #     'dino_dir': dino_dir,
    #     'dino_model': 'dinov2_vitg14_reg',
    #     'layers': [18, 21, 24, 27, 30],
    #     # 'input_size': [686, 364],
    #     'input_size': [854, 448],
    #     'facet': 'value',
    #     'device': 'cuda'
    # }

    #vitb
    # dino_args = {
    #     'dino_dir': dino_dir,
    #     'dino_model': 'dinov2_vitb14_reg',
    #     'layers': [2, 4, 6, 8, 10],
    #     # 'input_size': [686 * 2, 364 * 2],
    #     'input_size': [854, 448],
    #     'facet': 'value',
    #     'device': 'cuda'
    # }

    # vitl
    dino_args = {
        'dino_dir': dino_dir,
        'dino_model': 'dinov2_vitl14_reg',
        'layers': [8, 12, 16, 20],
        # 'input_size': [686 * 2, 364 * 2],
        # 'input_size': [686, 364],
        'input_size': [854, 448],
        'facet': 'value',
        'device': 'cuda'
    }

    # radio
    # dino_args = {
    #     'dino_dir': radio_dir,
    #     'dino_model': 'radio_v2.5-l',
    #     'layers': [9, 13, 17, 21],
    #     # 'input_size': [672, 352],
    #     'input_size': [848, 448],
    #     'facet': 'value',
    #     'device': 'cuda'
    # }

    dino = DinoV2ExtractFeatures(**dino_args)

    #double-check correctness
    dino_args['layers'] = [dino_args['layers'][-1]]
    dino_debug = DinoV2ExtractFeatures(**dino_args)

    for ifp in img_fps:
        img = cv2.imread(os.path.join(img_dir, ifp))
        img = torch.tensor(img, device='cuda') / 255.
        img = img.permute(2,0,1).unsqueeze(0)[:, [2,1,0]]

        t1 = time.time()
        feats = dino(img)
        torch.cuda.synchronize()
        t2 = time.time()

        feats2 = dino_debug(img)
        torch.cuda.synchronize()
        t3 = time.time()

        assert torch.allclose(feats2, feats[:, -feats2.shape[1]:])

        print('{}-layer inference took {:.4f}s, 1-layer took {:.4f}s'.format(len(dino.layers), t2-t1, t3-t2))

        img = img[0].permute(1, 2, 0)
        dino_img = feats[0].permute(1, 2, 0)

        viz_dino_features(img, dino_img, dino)