"""
PART 1: First run the below to generate outputs
python eval_davis_video_seg.py --dataroot /mnt/haiwen/datasets/DAVIS --model_type dinov2_vits14 --output_dir /mnt/haiwen/datasets/DAVIS/outputs/loftup --imsize 224 --upsampler_path /path/to/loftup_checkpoint.pth
PART 2: Then the below command computes the scores
python davis2017-evaluation/evaluation_method.py --davis_path /YOUR/PATH/TO/DAVIS --task semi-supervised --results_path davis_outputs/loftup/davis_vidseg_224 --imsize 224
"""
import os
import argparse
import time
import math

import torch
import torch.multiprocessing
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image, ImageFile, ImageColor
from torch.utils.data import DataLoader
from tqdm import tqdm
from torch import nn
from torchvision import transforms
from torchvision.datasets import ImageFolder
from urllib.request import urlopen

from tools.eval_video_segmentation import plot_video_features_davis, read_seg, read_frame_list
from featurizers import get_featurizer
from upsamplers import load_loftup_checkpoint

torch.multiprocessing.set_sharing_strategy('file_system')

class FeaturizerWithUpsampling(nn.Module):
    def __init__(self, model_type, upsampler_path=None, add_featup=False):
        super(FeaturizerWithUpsampling, self).__init__()
        self.featurizer, patch_size, dim = get_featurizer(model_type)
        self.featurizer = self.featurizer.to('cuda')
        self.dim = dim
        self.patch_size = patch_size
        self.featurizer.eval()
        if upsampler_path is not None:
            self.upsampler = load_loftup_checkpoint(upsampler_path, dim)
            self.upsampler.to('cuda')
            self.upsampler.eval()
        else:
            ## Assuming using pretrained FeatUp as the upsampler
            featup_upsampler = torch.hub.load("mhamilton723/FeatUp", 'dinov2')
            self.upsampler = featup_upsampler.upsampler
            self.upsampler.to('cuda')
            self.upsampler.eval()

    def forward(self, img, return_origianl_feat=False):
        feat = self.featurizer(img)
        ori_feat = feat.clone()
        if self.upsampler is not None:
            ### In DAVIS Experiments, we only upsample to img//2 size; so we use guidance image of size img//2 ###
            guidance_img = F.interpolate(img, size=(img.shape[-2]//2, img.shape[-1]//2), mode='bilinear', align_corners=False)
            up_feat = self.upsampler(feat, guidance_img)
        else:
            return feat, None
        up_feat = up_feat.reshape(up_feat.shape[0], up_feat.shape[1], -1)
        up_feat = up_feat.permute(0, 2, 1)
        if return_origianl_feat:
            return up_feat, ori_feat
        else:
            return up_feat, None
    

def run_video_segmentation(args):
    extractor = FeaturizerWithUpsampling(args.model_type, args.upsampler_path)
    if "siglip" in args.model_type:
        mean = (0.5, 0.5, 0.5)
        std = (0.5, 0.5, 0.5)
    else: # dino, dinov2, clip, etc
        mean = (0.485, 0.456, 0.406)
        std = (0.229, 0.224, 0.225)
    transform = transforms.Compose([
            transforms.Resize(args.imsize, interpolation=transforms.InterpolationMode.BICUBIC),
            transforms.CenterCrop(args.imsize),
            transforms.ToTensor(),
            transforms.Normalize(mean=mean, std=std)
        ])
    
    color_palette = []
    for line in urlopen("https://raw.githubusercontent.com/Liusifei/UVC/master/libs/data/palette.txt"):
        color_palette.append([int(i) for i in line.decode("utf-8").split('\n')[0].split(" ")])
    color_palette = np.asarray(color_palette, dtype=np.uint8).reshape(-1,3)
    video_list = open(os.path.join(args.dataroot, "ImageSets/2017/val.txt")).readlines()
    for i, video_name in enumerate(video_list):
        video_name = video_name.strip()
        print(f'[{i}/{len(video_list)}] Begin to segmentate video {video_name}.')
        video_dir = os.path.join(args.dataroot, "JPEGImages/480p/", video_name)
        frame_list = read_frame_list(video_dir)
        seg_path = frame_list[0].replace("JPEGImages", "Annotations").replace("jpg", "png")
        first_seg, seg_ori = read_seg(seg_path, factor=2, scale_size=[args.imsize, args.imsize]) ## 8x upsampling
        plot_video_features_davis(args, extractor, transform, frame_list, video_dir)


def parse_args():
    parser = argparse.ArgumentParser('DAVIS Object Segmentation Evaluation')
    ### BACKBONE ###
    parser.add_argument('--model_type', default='dino_vits8', type=str, help='backbone model type')
    ### DATASET ###
    parser.add_argument('--dataroot', default='path/to/DAVIS')
    parser.add_argument('--output_dir', default='temp_davis_test', help='dir to save metric plots to')
    parser.add_argument('--imsize', default=224, type=int, help='image resize size')
    parser.add_argument("--n_last_frames", type=int, default=7, help="number of preceeding frames")
    parser.add_argument("--size_mask_neighborhood", default=12, type=int,
                        help="We restrict the set of source nodes considered to a spatial neighborhood of the query node")
    parser.add_argument("--topk", type=int, default=5, help="accumulate label from top k neighbors")
    parser.add_argument("--upsampler_path", type=str, default=None, help="path of pretrained upsampler model to use. If not given, upsampling is not used")
    ###
    args = parser.parse_args()
    return args


if __name__ == '__main__':
    args = parse_args()
    start_time = time.time()
    run_video_segmentation(args)
    # print total time taken in minutes
    print(f'Total time taken: {(time.time() - start_time) / 60:.2f} minutes')
    # print max GPU memory used in GB
    print(f'Max GPU memory used: {torch.cuda.max_memory_allocated() / 1e9:.2f} GB')