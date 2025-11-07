import os
import copy
import glob
import queue
from urllib.request import urlopen
import argparse
import numpy as np
from tqdm import tqdm

import cv2
import torch
import torch.nn as nn
from torch.nn import functional as F
from PIL import Image
from torchvision import transforms
from einops import rearrange
from utils import pca
import matplotlib.pyplot as plt
@torch.no_grad()
def eval_video_tracking_davis(args, model, transform, frame_list, video_dir, first_seg, seg_ori, color_palette):
    """
    Evaluate tracking on a video given first frame & segmentation
    """
    os.makedirs(args.output_dir, exist_ok=True)
    os.makedirs(os.path.join(args.output_dir, f'davis_vidseg_224'), exist_ok=True)
    output_dir = os.path.join(args.output_dir, f'davis_vidseg_224')

    video_folder = os.path.join(output_dir, video_dir.split('/')[-1])
    os.makedirs(video_folder, exist_ok=True)

    # The queue stores the n preceeding frames
    que = queue.Queue(args.n_last_frames)

    # first frame
    frame1, ori_h, ori_w = read_frame(frame_list[0])
    # extract first frame feature

    frame1_feat, _ = extract_feature(args, model, transform, frame1, patch_size=model.patch_size, imsize=args.imsize) #  dim x h*w
    frame1_feat = frame1_feat.T # dim x h*w

    # saving first segmentation
    out_path = os.path.join(video_folder, "00000.png")
    imwrite_indexed(out_path, seg_ori, color_palette)
    mask_neighborhood = None
    for cnt in tqdm(range(1, len(frame_list))):
        frame_tar = read_frame(frame_list[cnt])[0]

        # we use the first segmentation and the n previous ones
        used_frame_feats = [frame1_feat] + [pair[0] for pair in list(que.queue)]
        used_segs = [first_seg] + [pair[1] for pair in list(que.queue)]

        frame_tar_avg, feat_tar, mask_neighborhood = label_propagation(args, model, transform, frame_tar, used_frame_feats, used_segs, mask_neighborhood)

        # pop out oldest frame if neccessary
        if que.qsize() == args.n_last_frames:
            que.get()
        # push current results into queue
        seg = copy.deepcopy(frame_tar_avg)
        que.put([feat_tar, seg])

        # upsampling & argmax
        if args.upsampler_path is not None:
            frame_tar_avg = F.interpolate(frame_tar_avg, scale_factor=2, mode='bilinear', align_corners=False, recompute_scale_factor=False)[0]
        else:
            frame_tar_avg = F.interpolate(frame_tar_avg, scale_factor=args.patch, mode='bilinear', align_corners=False, recompute_scale_factor=False)[0]
        frame_tar_avg = norm_mask(frame_tar_avg)
        _, frame_tar_seg = torch.max(frame_tar_avg, dim=0)

        # saving to disk
        frame_tar_seg = np.array(frame_tar_seg.squeeze().cpu(), dtype=np.uint8)
        frame_tar_seg = np.array(Image.fromarray(frame_tar_seg).resize((ori_w, ori_h), 0))
        frame_nm = frame_list[cnt].split('/')[-1].replace(".jpg", ".png")
        imwrite_indexed(os.path.join(video_folder, frame_nm), frame_tar_seg, color_palette)

@torch.no_grad()
def plot_video_features_davis(args, model, transform, frame_list, video_dir):
    """
    Plot the video features of the video
    """
    nonorm_transform = transforms.Compose([
            transforms.Resize(args.imsize, interpolation=transforms.InterpolationMode.BICUBIC),
            transforms.CenterCrop(args.imsize),
            transforms.ToTensor(),
        ])
    os.makedirs(args.output_dir, exist_ok=True)
    os.makedirs(os.path.join(args.output_dir, f'davis_vidfeat_224'), exist_ok=True)
    output_dir = os.path.join(args.output_dir, f'davis_vidfeat_224')

    video_folder = os.path.join(output_dir, video_dir.split('/')[-1])
    os.makedirs(video_folder, exist_ok=True)

    original_imgs = []
    original_features = []
    upsampled_features = []

    # first frame
    frame1, ori_h, ori_w = read_frame(frame_list[0])
    # extract first frame feature

    frame1_feat, frame1_original_feat = extract_feature(args, model, transform, frame1, patch_size=model.patch_size, imsize=args.imsize, return_origianl_feat=True) #  dim x h*w

    original_imgs.append(frame1) # format: list of PIL images
    original_features.append(frame1_original_feat) # format: dim x h*w
    upsampled_features.append(frame1_feat) # format: dim x h*w


    for cnt in tqdm(range(1, len(frame_list))):
        frame_tar = read_frame(frame_list[cnt])[0]

        frame_tar_feat, frame_tar_original_feat = extract_feature(args, model, transform, frame_tar, patch_size=model.patch_size, imsize=args.imsize, return_origianl_feat=True) #  dim x h*w

        original_imgs.append(frame_tar)
        original_features.append(frame_tar_original_feat)
        upsampled_features.append(frame_tar_feat)
    
    ## Perform PCA on the original features
    original_feats_pca, fit_pca = pca(original_features)
    upsampled_feats_pca, _ = pca(upsampled_features, fit_pca=fit_pca)


    def add_label(frame, text):
        labeled = frame.copy()
        height, width = labeled.shape[:2]
        
        font = cv2.FONT_HERSHEY_SIMPLEX
        font_scale = 1.0  # smaller font scale for small images
        thickness = 3  # thinner line for small image
        margin = 10  # margin from the top-left corner

        # Calculate the size of the text box
        text_size, _ = cv2.getTextSize(text, font, font_scale, thickness)
        text_width, text_height = text_size

        # Make sure text doesn't overflow
        if text_width + 2 * margin > width:
            font_scale = font_scale * (width - 2 * margin) / text_width
            text_size, _ = cv2.getTextSize(text, font, font_scale, thickness)
            text_width, text_height = text_size

        position = (margin, margin + text_height)

        cv2.putText(labeled, text, position, font, font_scale, (255, 255, 255), thickness, cv2.LINE_AA)
        return labeled

    def feature_to_rgb(feat, shape):
        feat = np.array(feat)
        feat -= feat.min()
        if feat.max() > 0:
            feat /= feat.max()
        feat = (feat * 255).astype(np.uint8)
        return feat.reshape(*shape, 3)

    final_frames = []
    for frame_idx, (orig_pil, f_small, f_big) in enumerate(zip(original_imgs, original_feats_pca, upsampled_feats_pca)):
        # Convert original image to RGB array

        orig = np.array(nonorm_transform(orig_pil)*255).astype(np.uint8).transpose(1, 2, 0)
        # import ipdb; ipdb.set_trace()
        # Feature shapes
        up_size = args.imsize // 2
        feat_size = args.imsize // model.patch_size

        img_size = args.imsize

        # Convert features to RGB images
        dino_rgb = feature_to_rgb(f_small, (feat_size, feat_size))
        loftup_rgb = feature_to_rgb(f_big, (up_size, up_size))

        # Resize small DINO feature to (H, W)
        dino_rgb_resized = cv2.resize(dino_rgb, (img_size, img_size), interpolation=cv2.INTER_NEAREST)
        loftup_rgb_resized = cv2.resize(loftup_rgb, (img_size, img_size), interpolation=cv2.INTER_NEAREST)

        # Add labels
        orig_labeled = add_label(orig, "Input")
        if args.model_type == "dinov2":
            dino_labeled = add_label(dino_rgb_resized, "DINOv2")
            if args.upsampler_path is not None:
                loftup_labeled = add_label(loftup_rgb_resized, "DINOv2 + LoftUp")
            else:
                loftup_labeled = add_label(loftup_rgb_resized, "DINOv2 + FeatUp")
        elif args.model_type == "siglip2":
            dino_labeled = add_label(dino_rgb_resized, "SigLIP2")
            loftup_labeled = add_label(loftup_rgb_resized, "SigLIP2 + LoftUp")
        elif args.model_type == "clip":
            dino_labeled = add_label(dino_rgb_resized, "CLIP")
            loftup_labeled = add_label(loftup_rgb_resized, "CLIP + LoftUp")

        # Stack vertically
        separator_thickness = 20  # pixels
        # separator = np.zeros((separator_thickness, orig_labeled.shape[1], 3), dtype=np.uint8)  # black separator
        # stacked = cv2.vconcat([orig_labeled, separator, dino_labeled, separator, loftup_labeled])
        separator = np.ones((orig_labeled.shape[0], separator_thickness, 3), dtype=np.uint8) * 0  # black separator
        stacked = cv2.hconcat([
                orig_labeled,
                separator,
                dino_labeled,
                separator,
                loftup_labeled
            ])
        # Also plot the stacked image
        plt.imshow(stacked)
        plt.axis('off')
        plt.savefig(os.path.join(video_folder, f'{frame_idx}.png'), bbox_inches='tight', pad_inches=0)
        plt.close()
        final_frames.append(stacked)

    # --- Write video ---
    out_height, out_width = final_frames[0].shape[:2]
    video_writer = cv2.VideoWriter(os.path.join(video_folder, "feature_visualization.mp4"), cv2.VideoWriter_fourcc(*'mp4v'), 10, (out_width, out_height))

    for f in final_frames:
        video_writer.write(cv2.cvtColor(f, cv2.COLOR_RGB2BGR))
    video_writer.release()    

       

def restrict_neighborhood(args, h, w):
    # We restrict the set of source nodes considered to a spatial neighborhood of the query node (i.e. ``local attention'')
    mask = torch.zeros(h, w, h, w)
    for i in range(h):
        for j in range(w):
            for p in range(2 * args.size_mask_neighborhood + 1):
                for q in range(2 * args.size_mask_neighborhood + 1):
                    if i - args.size_mask_neighborhood + p < 0 or i - args.size_mask_neighborhood + p >= h:
                        continue
                    if j - args.size_mask_neighborhood + q < 0 or j - args.size_mask_neighborhood + q >= w:
                        continue
                    mask[i, j, i - args.size_mask_neighborhood + p, j - args.size_mask_neighborhood + q] = 1

    mask = mask.reshape(h * w, h * w)
    return mask.cuda(non_blocking=True)


def norm_mask(mask):
    c, h, w = mask.size()
    for cnt in range(c):
        mask_cnt = mask[cnt,:,:]
        if(mask_cnt.max() > 0):
            mask_cnt = (mask_cnt - mask_cnt.min())
            mask_cnt = mask_cnt/mask_cnt.max()
            mask[cnt,:,:] = mask_cnt
    return mask


def label_propagation(args, model, transform, frame_tar, list_frame_feats, list_segs, mask_neighborhood=None):
    """
    propagate segs of frames in list_frames to frame_tar
    """
    ## we only need to extract feature of the target frame
    feat_tar, _, h, w = extract_feature(args, model, transform, frame_tar, return_h_w=True, patch_size = model.patch_size, imsize=args.imsize)

    # detach feat_tar
    feat_tar = feat_tar.detach().cpu()

    return_feat_tar = feat_tar.T # dim x h*w

    ncontext = len(list_frame_feats)
    feat_sources = torch.stack(list_frame_feats) # nmb_context x dim x h*w

    feat_sources = feat_sources.detach().cpu()

    feat_tar = F.normalize(feat_tar, dim=1, p=2)
    feat_sources = F.normalize(feat_sources, dim=1, p=2)

    feat_tar = feat_tar.unsqueeze(0).repeat(ncontext, 1, 1)
    aff = torch.exp(torch.bmm(feat_tar, feat_sources) / 0.1) # nmb_context x h*w (tar: query) x h*w (source: keys)

    if args.size_mask_neighborhood > 0:
        if mask_neighborhood is None:
            mask_neighborhood = restrict_neighborhood(args, h, w)
            mask_neighborhood = mask_neighborhood.unsqueeze(0).repeat(ncontext, 1, 1)
        mask_neighborhood = mask_neighborhood.detach().cpu()
        aff *= mask_neighborhood
    
    aff = aff.transpose(2, 1).reshape(-1, h * w) # nmb_context*h*w (source: keys) x h*w (tar: queries)
    tk_val, _ = torch.topk(aff, dim=0, k=args.topk)
    tk_val_min, _ = torch.min(tk_val, dim=0)
    aff[aff < tk_val_min] = 0

    aff = aff / torch.sum(aff, keepdim=True, axis=0)

    # list_segs = [s.cuda() for s in list_segs]
    ## Ensure all the tensors are on the same device
    list_segs = [s.detach().cpu()  if s.device != torch.device('cpu') else s for s in list_segs]
    segs = torch.cat(list_segs)
    nmb_context, C, h, w = segs.shape
    segs = segs.reshape(nmb_context, C, -1).transpose(2, 1).reshape(-1, C).T # C x nmb_context*h*w
    seg_tar = torch.mm(segs, aff)
    seg_tar = seg_tar.reshape(1, C, h, w)
    torch.cuda.empty_cache()
    return seg_tar.cuda(), return_feat_tar.cuda(), mask_neighborhood.cuda()


def extract_feature(args, model, transform, frame, return_h_w=False, patch_size=16, imsize=224, return_origianl_feat=False):
    """Extract one frame feature everytime."""
    with torch.no_grad():
        frame = transform(frame)
        out, original_out = model(frame.unsqueeze(0).cuda(), return_origianl_feat=return_origianl_feat)
    h, w = frame.shape[1]//2, frame.shape[2]//2

    if out.shape[-2] != h*w:
        out_h = int(np.sqrt(out.shape[-2]))
        out = out[0].reshape(out_h, out_h, -1).permute(2, 0, 1).unsqueeze(0)
        out = F.interpolate(out, size=(h, w), mode='bilinear', align_corners=False)
        out = out.permute(0, 2, 3, 1)
    dim = out.shape[-1]
    out = out[0].reshape(h, w, dim)
    out = out.reshape(-1, dim)
    if original_out is not None:
        original_out = rearrange(original_out, 'b c h w -> (b h w) c') ### Direct reshape will lead to bugs!!!   
    if return_h_w:
        return out, original_out, h, w
    return out, original_out


def imwrite_indexed(filename, array, color_palette):
    """ Save indexed png for DAVIS."""
    if np.atleast_3d(array).shape[2] != 1:
      raise Exception("Saving indexed PNGs requires 2D array.")

    im = Image.fromarray(array)
    im.putpalette(color_palette.ravel())
    im.save(filename, format='PNG')


def to_one_hot(y_tensor, n_dims=None):
    """
    Take integer y (tensor or variable) with n dims &
    convert it to 1-hot representation with n+1 dims.
    """
    if(n_dims is None):
        n_dims = int(y_tensor.max()+ 1)
    _,h,w = y_tensor.size()
    y_tensor = y_tensor.type(torch.LongTensor).view(-1, 1)
    n_dims = n_dims if n_dims is not None else int(torch.max(y_tensor)) + 1
    y_one_hot = torch.zeros(y_tensor.size()[0], n_dims).scatter_(1, y_tensor, 1)
    y_one_hot = y_one_hot.view(h,w,n_dims)
    return y_one_hot.permute(2, 0, 1).unsqueeze(0)


def read_frame_list(video_dir):
    frame_list = [img for img in glob.glob(os.path.join(video_dir,"*.jpg"))]
    frame_list = sorted(frame_list)
    return frame_list


def read_frame(frame_dir):
    img = Image.open(frame_dir)
    ori_w, ori_h = img.size
    return img, ori_h, ori_w


def read_seg(seg_dir, factor, scale_size=[224,224], custom=None):
    seg = Image.open(seg_dir)
    _w, _h = seg.size # note PIL.Image.Image's size is (w, h)
    if len(scale_size) == 1:
        if(_w > _h):
            _th = scale_size[0]
            _tw = (_th * _w) / _h
            _tw = int((_tw // factor) * factor)
        else:
            _tw = scale_size[0]
            _th = (_tw * _h) / _w
            _th = int((_th // factor) * factor)
    else:
        _th = scale_size[1]
        _tw = scale_size[0]
    small_seg = np.array(seg.resize((_tw // factor, _th // factor), 0))
    if(custom is not None):
        small_seg = np.array(seg.resize((custom, custom), 0))
    small_seg = torch.from_numpy(small_seg.copy()).contiguous().float().unsqueeze(0)
    return to_one_hot(small_seg), np.asarray(seg)


def color_normalize(x, mean=[0.485, 0.456, 0.406], std=[0.228, 0.224, 0.225]):
    for t, m, s in zip(x, mean, std):
        t.sub_(m)
        t.div_(s)
    return x