from physics_atv_visual_mapping.image_processing.processing_blocks.base import ImageProcessingBlock
from physics_atv_visual_mapping.image_processing.third_party.clip_utils import clip
from physics_atv_visual_mapping.image_processing.third_party.clip_utils.imagenet_template import openai_imagenet_template
import torch

class NAClipBlock(ImageProcessingBlock):
    """
    Image processing block that runs NAClip image encoder on the image
    """
    def __init__(self, image_insize, device):
        self.image_insize = image_insize
        self.naclip_model, _ = clip.load("ViT-B/16", device=device, jit=False)
        self.naclip_model.eval()
        self.naclip_model.visual.set_params(arch="reduced", attn_strategy="naclip", gaussian_std=5.)
        
        self.device = device

    def run(self, image, intrinsics, image_orig):

        image_features = self.naclip_model.encode_image(image, return_all=True).detach()
        image_features = image_features[:, 1:]
        image_features /= image_features.norm(dim=-1, keepdim=True)
        
        patch_window = 16
        B = image_features.shape[0]
        H, W = image.shape[-2] // patch_window, image.shape[-1] // patch_window
        C = image_features.shape[-1]
        img_out = image_features.reshape(B, H, W, C).permute(0, 3, 1, 2)

        print("Image shape", image.shape)
        print("Image out shape", img_out.shape)
        
        ix = image.shape[3]
        dx = img_out.shape[3]
        iy = image.shape[2]
        dy = img_out.shape[2]

        intrinsics[:, 0, 0] *= (dx/ix)
        intrinsics[:, 0, 2] *= (dx/ix)

        intrinsics[:, 1, 1] *= (dy/iy)
        intrinsics[:, 1, 2] *= (dy/iy)

        return img_out, intrinsics
    
    def default_text_feats(self, labels):
        
        text_features = []
        
        for qw in labels:
            prompts = [qw]
            query = clip.tokenize(prompts).to(self.device)
            feature = self.naclip_model.encode_text(query)
            feature /= feature.norm(dim=-1, keepdim=True)
            feature = feature.mean(dim=0)
            feature /= feature.norm()
            text_features.append(feature.unsqueeze(0))
        
        # query = clip.tokenize(labels).to(self.device)
        # feature = self.naclip_model.encode_text(query)
        # feature /= feature.norm(dim=-1, keepdim=True)
        # feature = feature.mean(dim=0)
        # feature /= feature.norm()
        # text_features.append(feature.unsqueeze(0))
        
        text_features = torch.cat(text_features, dim=0)
            
        return text_features