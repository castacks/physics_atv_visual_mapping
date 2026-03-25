import torch
from torch.utils.data import Dataset
import os
import numpy as np
import yaml
import os
import cv2
from scipy.spatial.transform import Rotation as R
from PIL import Image
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import torch
# import torchdiffeq
# import torchsde
# from torchdyn.core import NeuralODE
import torchvision
from torchvision import datasets, transforms
from torchvision.transforms import ToPILImage
from torchvision.utils import make_grid
from torchvision.transforms.functional import hflip
from tqdm import tqdm
# from efficientnet_pytorch import EfficientNet
from typing import List, Dict, Optional, Tuple, Callable

import rasterio
from torch.utils.data import ConcatDataset
from torch.optim.lr_scheduler import SequentialLR, LinearLR, CosineAnnealingWarmRestarts
import torch
import torch.nn as nn
import torch.nn.functional as F

from torch.nn.functional import grid_sample


from pathlib import Path
import torch
import torchvision.transforms as T
import os
from hydra.utils import instantiate
import time
from utils.notebooks import load_model
from utils.img import unnormalize
from utils.visualization import plot_feats, viz_pca
import torchvision

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
project_root = str(Path().absolute())

print(project_root)
backbone = "vit_small_patch14_dinov2.lvd142m"

model, backbone = load_model(backbone, project_root)


# Load an image
img_size = 224

data_dir = '/media/striest/offroad/datasets/yamaha_kitti/20250808/teleop/01_garage_to_turnpike_long/image'
#warehouse cone
idx = 242
qcoords = np.array([[100,300],[500,800]])

#
idx = 981
qcoords = np.array([[50,250],[300,500]])


idx_str = f"{idx:08d}"

img_path = os.path.join(data_dir, f"{idx_str}.png")
og_img = Image.open(img_path).convert("RGB")
og_img = np.array(og_img).astype(np.float32)/255.
raw_img = og_img.copy()
og_img = np.transpose(og_img, [2,0,1])
og_img = torch.from_numpy(og_img).unsqueeze(0).cuda()

# Transform the image to match the input requirements of the model
mean = torch.tensor([0.485, 0.456, 0.406], device=device)
std = torch.tensor([0.229, 0.224, 0.225], device=device)
transform = T.Compose(
    [
        # T.Resize(img_size),
        # T.CenterCrop(img_size),
        # T.ToTensor(),  # Convert to tensor
        T.Normalize(mean=mean, std=std),  # Normalize
    ]
)
og_img = transform(og_img).to(device)  # Add batch dimension and move to device

raw_subquery = raw_img[qcoords[0,0]:qcoords[0,1], qcoords[1,0]:qcoords[1,1]]
sq = og_img[:,:,qcoords[0,0]:qcoords[0,1], qcoords[1,0]:qcoords[1,1]]


og_img = torchvision.transforms.functional.resize(og_img,(img_size,img_size))
sq = torchvision.transforms.functional.resize(sq,(img_size,img_size))


print(og_img.shape)
print(sq.shape)

fig, ax = plt.subplots(1,2)
ax[0].imshow(raw_img)
ax[1].imshow(raw_subquery)
plt.show()

# print(og_img.min(), og_img.max())
image_intrinsics = torch.eye(4).unsqueeze(0)
with torch.no_grad():

    lr_feats, _ = backbone(og_img)
    feat_img = model(og_img, lr_feats, (img_size, img_size))

    lr_feats, _ = backbone(sq)
    feat_sq = model(sq, lr_feats, (img_size, img_size))



    feat_img = feat_img[:,:3]
    feat_sq = feat_sq[:,:3]

    # feat_sq = feat_sq[0].cpu().numpy()

    print(feat_img.shape)
    feat_img = F.interpolate(
        feat_img,  # add batch dim
        size=raw_img.shape[:2], # (H, W)
        mode="bilinear",
        align_corners=False
    )[0].cpu().numpy()  # (C, H, W)

    feat_sq = F.interpolate(
        feat_sq,
        size=raw_subquery.shape[:2],
        mode="bilinear",
        align_corners=False
    )[0].cpu().numpy()

    # print(feat_img.shape)

    stitched = feat_img.copy()
    stitched[:,qcoords[0,0]:qcoords[0,1], qcoords[1,0]:qcoords[1,1]] = feat_sq

    mins = feat_img.min(axis=(1,2), keepdims=True)   # shape (1,1,3)
    maxs = feat_img.max(axis=(1,2), keepdims=True)
    viz_img = (feat_img - mins)/(maxs-mins)
    # viz_img = normalize_dino(torch.from_numpy(feat_img)).numpy()

    viz_sq = (feat_sq - mins)/(maxs-mins)

    viz_stitched = (stitched - mins)/(maxs-mins)

    viz_unstitched = viz_img[:,qcoords[0,0]:qcoords[0,1], qcoords[1,0]:qcoords[1,1]]

    print(viz_img.shape, viz_unstitched.shape)

    fig, axs = plt.subplots(2,3)
    axs[0,0].imshow(raw_img)
    axs[1,0].imshow(raw_subquery)
    axs[0,1].imshow(np.transpose(viz_img[0:3],[1,2,0]))
    axs[0,2].imshow(np.transpose(viz_unstitched[0:3],[1,2,0]))
    axs[1,2].imshow(np.transpose(viz_sq[0:3],[1,2,0]))
    axs[1,1].imshow(np.transpose(viz_stitched[0:3],[1,2,0]))

    for ax in axs.flatten():
        ax.axis('off')

    axs[0,0].set_title("Original Image")
    axs[0,1].set_title("Original Loftup")
    axs[0,2].set_title("Original Loftup Crop")
    axs[1,0].set_title("Cropped Input Image")
    axs[1,1].set_title("Stitched")
    axs[1,2].set_title("Cropped Loftup")

    plt.show()