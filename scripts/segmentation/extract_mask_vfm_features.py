import os
import yaml
import argparse

import cv2
import torch
import numpy as np
import matplotlib.pyplot as plt

from termcolor import colored

from physics_atv_visual_mapping.image_processing.image_pipeline import setup_image_pipeline
from physics_atv_visual_mapping.utils import *

def get_feat_img_prototype_cosine_sim(feat_img, prototypes):
    ptype_csim = (feat_img * prototypes.view(prototypes.shape[0], prototypes.shape[1], 1, 1)).sum(dim=1)
    ptype_csim /= (torch.linalg.norm(prototypes, dim=1).view(-1, 1, 1) * torch.linalg.norm(feat_img, dim=1))
    return ptype_csim

def get_prototype_scores(feat_img, prototypes):
    pos_ptypes = torch.stack([v['ptype'] for v in prototypes['obstacle']], dim=0)
    neg_ptypes = torch.stack([v['ptype'] for v in prototypes['nonobstacle']], dim=0)

    pos_csim = get_feat_img_prototype_cosine_sim(feat_img, pos_ptypes)
    neg_csim = get_feat_img_prototype_cosine_sim(feat_img, neg_ptypes)

    return pos_csim, neg_csim

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, required=True, help="path to config")
    parser.add_argument("--data_dir", type=str, required=True, help="path to data_dir")
    parser.add_argument("--viz_dir", type=str, required=False, help="img dir to viz on")
    parser.add_argument("--remove_pca", action='store_true', help='set this flag to remove pca block from config')
    args = parser.parse_args()
    
    config = yaml.safe_load(open(args.config, "r"))

    #if config already has a pca, remove it.
    if args.remove_pca:
        image_processing_config = []
        for ip_block in config['image_processing']:
            if ip_block['type'] == 'pca':
                print(colored('removing pca block...', 'yellow'))
                break
            image_processing_config.append(ip_block)

        config['image_processing'] = image_processing_config

    image_pipeline = setup_image_pipeline(config)

    img_dir = os.path.join(args.data_dir, 'raw')

    neg_mask_dir = os.path.join(args.data_dir, 'masks', 'no')
    neg_mask_fps = [x for x in os.listdir(neg_mask_dir) if '.mask.png' in x]

    print('get obstacle prototypes for these images:')
    for mfp in neg_mask_fps:
        print('\t' + mfp)

    # store as list to guarantee ordering
    prototypes = {
        'obstacle': [],
        'nonobstacle': [],
    }

    for mfp in neg_mask_fps:
        ptype_name = mfp.rsplit('_', 1)[0]

        mask_fp = os.path.join(neg_mask_dir, mfp)
        img_fp = os.path.join(img_dir, ptype_name + '.png')

        mask = cv2.imread(mask_fp)
        mask = torch.tensor(mask[..., 0]).to(config['device']) > 254

        img = cv2.imread(img_fp)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB) / 255.0
        img = torch.tensor(img).float().to(config['device'])
        img = img.unsqueeze(0).permute(0, 3, 1, 2)

        intrinsics = torch.zeros(1, 3, 3) #intrinsics not relevant for this script

        feat_img, _ = image_pipeline.run(
            img, intrinsics
        )

        ## get VFM pixels that intersect the mask
        rx = mask.shape[0] / feat_img.shape[2]
        ry = mask.shape[1] / feat_img.shape[3]

        mask_px_list = torch.argwhere(mask)
        mask_feat_px_list = mask_px_list.clone().float()
        mask_feat_px_list[:, 0] /= rx
        mask_feat_px_list[:, 1] /= ry

        nonmask_px_list = torch.argwhere(~mask)
        nonmask_feat_px_list = nonmask_px_list.clone().float()
        nonmask_feat_px_list[:, 0] /= rx
        nonmask_feat_px_list[:, 1] /= ry

        #note that these lists can overlap for px on the mask boundary
        px_in_mask = torch.unique(mask_feat_px_list.long(), dim=0)
        px_not_in_mask = torch.unique(nonmask_feat_px_list.long(), dim=0)

        all_px = torch.cat([px_in_mask, px_not_in_mask], dim=0)
        uniq, inv, cnt = torch.unique(all_px, dim=0, return_inverse=True, return_counts=True)
        px_only_in_mask = px_in_mask[cnt[inv[:px_in_mask.shape[0]]] == 1]

        # [K x C]
        mask_feats = feat_img[0][:, px_only_in_mask[:, 0], px_only_in_mask[:, 1]].T
        all_feats = feat_img[0].reshape(feat_img.shape[0], -1).T

        #this is probably bad but will work for now
        ptype = mask_feats.mean(dim=0)
        prototypes['obstacle'].append({
            'label': ptype_name,
            'ptype': ptype
        })

    pos_mask_dir = os.path.join(args.data_dir, 'masks', 'yes')
    pos_mask_fps = [x for x in os.listdir(pos_mask_dir) if '.mask.png' in x]

    print('get nonobstacle prototypes for these images:')
    for mfp in pos_mask_fps:
        print('\t' + mfp)

    for mfp in pos_mask_fps:
        ptype_name = mfp.rsplit('_', 1)[0]
        mask_fp = os.path.join(pos_mask_dir, mfp)
        img_fp = os.path.join(img_dir, ptype_name + '.png')

        mask = cv2.imread(mask_fp)
        mask = torch.tensor(mask[..., 0]).to(config['device']) > 254

        img = cv2.imread(img_fp)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB) / 255.0
        img = torch.tensor(img).float().to(config['device'])
        img = img.unsqueeze(0).permute(0, 3, 1, 2)

        intrinsics = torch.zeros(1, 3, 3) #intrinsics not relevant for this script

        feat_img, _ = image_pipeline.run(
            img, intrinsics
        )

        ## get VFM pixels that intersect the mask
        rx = mask.shape[0] / feat_img.shape[2]
        ry = mask.shape[1] / feat_img.shape[3]

        mask_px_list = torch.argwhere(mask)
        mask_feat_px_list = mask_px_list.clone().float()
        mask_feat_px_list[:, 0] /= rx
        mask_feat_px_list[:, 1] /= ry

        nonmask_px_list = torch.argwhere(~mask)
        nonmask_feat_px_list = nonmask_px_list.clone().float()
        nonmask_feat_px_list[:, 0] /= rx
        nonmask_feat_px_list[:, 1] /= ry

        #note that these lists can overlap for px on the mask boundary
        px_in_mask = torch.unique(mask_feat_px_list.long(), dim=0)
        px_not_in_mask = torch.unique(nonmask_feat_px_list.long(), dim=0)

        all_px = torch.cat([px_in_mask, px_not_in_mask], dim=0)
        uniq, inv, cnt = torch.unique(all_px, dim=0, return_inverse=True, return_counts=True)
        px_only_in_mask = px_in_mask[cnt[inv[:px_in_mask.shape[0]]] == 1]

        # [K x C]
        mask_feats = feat_img[0][:, px_only_in_mask[:, 0], px_only_in_mask[:, 1]].T
        all_feats = feat_img[0].reshape(feat_img.shape[0], -1).T

        #this is probably bad but will work for now
        ptype = mask_feats.mean(dim=0)
        prototypes['nonobstacle'].append({
            'label': ptype_name,
            'ptype': ptype
        })

    torch.save(prototypes, 'prototypes.pt')

    prototypes = torch.load('prototypes.pt')

    ## Viz ##
    if args.viz_dir is None:
        viz_dir = img_dir
    else:
        viz_dir = args.viz_dir

    viz_img_fps = os.listdir(viz_dir)

    for img_fp in viz_img_fps:
        img_fp = os.path.join(viz_dir, img_fp)

        img = cv2.imread(img_fp)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB) / 255.0
        img = torch.tensor(img).float().to(config['device'])
        img = img.unsqueeze(0).permute(0, 3, 1, 2)

        intrinsics = torch.zeros(1, 3, 3) #intrinsics not relevant for this script

        feat_img, _ = image_pipeline.run(
            img, intrinsics
        )

        obs_csim, nonobs_csim = get_prototype_scores(feat_img, prototypes)

        obstacle_score = obs_csim.max(dim=0)[0]
        nonobstacle_score = nonobs_csim.max(dim=0)[0]
        # score = (obstacle_score + 1.) / (nonobstacle_score + 1.)

        score = obstacle_score - nonobstacle_score

        img_viz = img[0].permute(1,2,0).cpu().numpy()
        feat_img_viz = normalize_dino(feat_img[0].permute(1,2,0)).cpu().numpy()

        extent = (0, img_viz.shape[1], 0, img_viz.shape[0])

        fig, axs = plt.subplots(1, 5, figsize=(25, 4))
        axs[0].set_title('img')
        axs[0].imshow(img_viz, extent=extent)

        axs[1].set_title('feat img')
        axs[1].imshow(feat_img_viz, extent=extent)

        axs[2].set_title('obstacle sim')
        axs[2].imshow(obstacle_score.cpu().numpy(), extent=extent, alpha=1.0, cmap='coolwarm', vmin=-1., vmax=1.)

        axs[3].set_title('traversable sim')
        axs[3].imshow(nonobstacle_score.cpu().numpy(), extent=extent, alpha=1.0, cmap='coolwarm', vmin=-1., vmax=1.)

        axs[4].set_title('score')
        axs[4].imshow(score.cpu().numpy(), extent=extent, alpha=1.0, cmap='jet')

        fig.suptitle(img_fp)
        plt.show()
