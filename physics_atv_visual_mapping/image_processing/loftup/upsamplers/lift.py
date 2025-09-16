"""
LiFT Module for ViT feature upsampling.

Code by: Saksham Suri and Matthew Walmer
"""
import torch
import torch.nn as nn
import torch.nn.functional as F


class DoubleConv(nn.Module):
    """(convolution => [BN] => ReLU) * 2"""

    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.double_conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        return self.double_conv(x)


class Up(nn.Module):
    """Upscaling then double conv"""

    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.up = nn.ConvTranspose2d(in_channels, in_channels//2, kernel_size=2, stride=2)
        self.conv_1 = DoubleConv(in_channels//2+32, out_channels//2)

    def forward(self, x, imgs_1):
        x = self.up(x)
        x = torch.cat([x, imgs_1], dim=1)
        x = self.conv_1(x)
        return x


"""
in_channels: number of channels in the features from the ViT backbone

patch_size: size of patches used by the ViT backbone

pre_shape: enable/disable reshaping of feature inputs, altering the expected input shape 
    True - will accept input shape [B, T, C] (ViT standard), which it will convert to [B, C, H, W]
    False - will accept input shape [B, C, H, W], conversion already performed

post_shape: enable/disable reshaping of feature outputs
    True - will return output in shape [B, T, C] (ViT standard)
    False - will return output in shape [B, C, H, W]
"""
class LiFT(nn.Module):
    def __init__(self, in_channels, patch_size, pre_shape=False, post_shape=False):
        super(LiFT, self).__init__()
        self.patch_size = patch_size
        self.pre_shape = pre_shape
        self.post_shape = post_shape
        
        self.up1 = (Up(in_channels+32, in_channels))
        self.outc = nn.Conv2d(in_channels//2, in_channels, kernel_size=1)
        self.image_convs_1 = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=3, padding=1, stride=2),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 32, kernel_size=3, padding=1, stride=2),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
        )
        if patch_size == 8:
            self.scale_adapter = nn.Identity()
        elif patch_size == 16:
            self.scale_adapter = nn.MaxPool2d(2, 2)
        elif patch_size == 14:
            self.scale_adapter = None
            # self.scale_adapter_14 = F.adaptive_max_pool2d((14, 14))
            # self.scale_adapter_7 = F.adaptive_max_pool2d((7, 7))
            # self.scale_adapter_28 = F.adaptive_max_pool2d((28, 28))
        else:
            print('ERROR: patch size %i not currently supported'%patch_size)
            exit()
        self.image_convs_2 = nn.Sequential(
            nn.Conv2d(32, 32, kernel_size=3, padding=1, stride=2),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
        )


    # [B, T, C] --> [B, C, H, W]
    def run_pre_shape(self, imgs, x):
        H = int(imgs.shape[2] / self.patch_size)
        W = int(imgs.shape[3] / self.patch_size)
        x = x.permute(0, 2, 1)
        x = x.reshape(x.shape[0], -1, H, W)
        return x


    # [B, C, H, W] --> [B, T, C]
    def run_post_shape(self, x):
        x = x.reshape(x.shape[0], x.shape[1], -1)
        x = x.permute(0, 2, 1)
        return x


    def forward(self, x, imgs):
        if self.pre_shape: x = self.run_pre_shape(imgs, x)
        imgs_1 = self.image_convs_1(imgs)
        imgs_1 = F.adaptive_max_pool2d(imgs_1, (x.shape[2]*2, x.shape[3]*2))
        # imgs_1 = self.scale_adapter(imgs_1)
        imgs_2 = self.image_convs_2(imgs_1)
        # Enable the following if working with both --imsize 56 and --patch_size 16
        # if(x.shape[2] != imgs_2.shape[2]):
        #     imgs_1 = self.image_convs_1(imgs[:,:,2:-2,2:-2])
        #     imgs_1 = self.scale_adapter(imgs_1)
        #     imgs_2 = self.image_convs_2(imgs_1)
        x = torch.cat([x, imgs_2], dim=1)
        x = self.up1(x, imgs_1)
        logits = self.outc(x)
        if self.post_shape: logits = self.run_post_shape(logits)
        return logits

def load_lift_checkpoints(lift_path, channel=384, patch=14):
        lift = LiFT(channel, patch)
        state_dict = torch.load(lift_path)
            # if "module." in state_dict, remove it
        for k in list(state_dict.keys()):
            if k.startswith('module.'):
                state_dict[k[7:]] = state_dict[k]
                del state_dict[k]
        lift.load_state_dict(state_dict)
        lift.to("cuda")
        print('Loaded LiFT module from: ' + lift_path)
        return lift