from upsamplers import load_loftup_checkpoint, norm, unnorm
from featurizers import get_featurizer
from utils import plot_feats

import torch
import torch.nn.functional as F
from PIL import Image, ImageOps
import torchvision.transforms as T
import torchvision.transforms.functional as TF

featurizer_class = "dinov2" # "dinov2", "dinov2b", "dinov2s_reg", "dinov2b_reg", "clip", "siglip", "siglip2"
torch_hub_name = "loftup_dinov2s" # "loftup_dinov2s", "loftup_dinov2b", "loftup_dinov2s_reg", "loftup_dinov2b_reg", "loftup_clip", "loftup_siglip", "loftup_siglip2"

model, patch_size, dim = get_featurizer(featurizer_class)
model = model.to('cuda')
kernel_size = patch_size 
lr_size = 224 // patch_size
load_size = 224

upsampler = torch.hub.load('andrehuang/loftup', torch_hub_name, pretrained=True)
upsampler = upsampler.to('cuda')

image_path = "examples/sa_1.jpg"

transform = T.Compose([
        T.Resize(load_size, T.InterpolationMode.BILINEAR),
        T.CenterCrop(load_size), # Depending on whether you want a center crop
        T.ToTensor(),
        norm])
img = Image.open(image_path).convert("RGB")
normalized_img_tensor = transform(img).unsqueeze(0).to('cuda')
lr_feats = model(normalized_img_tensor) # 1, dim, lr_size, lr_size

## Upsampling step
hr_feats = upsampler(lr_feats, normalized_img_tensor) # 1, dim, 224, 224

# ## You can also upsample to any shape you want; Just change the guidance image shape
# H, W = 112, 112
# img_tensor_112 = F.interpolate(normalized_img_tensor, size=(H, W), mode='bilinear', align_corners=False)
# hr_feats_112 = upsampler(lr_feats, img_tensor_112) # 1, 384, 112, 112

plot_feats(unnorm(normalized_img_tensor)[0], lr_feats[0], hr_feats[0], 'examples/feats.png')
