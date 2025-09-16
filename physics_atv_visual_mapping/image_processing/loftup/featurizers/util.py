import torch

def get_featurizer(name, models_dir, activation_type="token", **kwargs):
    name = name.lower()
    if name == "dinov2":
        from .DINOv2 import DINOv2Featurizer
        patch_size = 14
        model = DINOv2Featurizer(models_dir, "dinov2_vits14", patch_size, activation_type)
        dim = 384
    elif name == "dinov2b":
        from .DINOv2 import DINOv2Featurizer
        patch_size = 14
        model = DINOv2Featurizer(models_dir, "dinov2_vitb14", patch_size, activation_type)
        dim = 768
    elif name == "dinov2s_reg":
        from .DINOv2 import DINOv2Featurizer
        patch_size = 14
        model = DINOv2Featurizer(models_dir, "dinov2_vits14_reg", patch_size, activation_type)
        dim = 384
    elif name == "dinov2b_reg":
        from .DINOv2 import DINOv2Featurizer
        patch_size = 14
        model = DINOv2Featurizer(models_dir, "dinov2_vitb14_reg", patch_size, activation_type)
        dim = 768
    elif name == "clip":
        from .CLIP import CLIPFeaturizer
        patch_size = 16
        model = CLIPFeaturizer()
        dim = 512
    elif name == "siglip":
        from .SigLIP import SigLIPFeaturizer
        patch_size = 16
        model = SigLIPFeaturizer("hf-hub:timm/ViT-B-16-SigLIP", patch_size)
        dim = 768
    elif name == "siglip2":
        from .SigLIP import SigLIPFeaturizer
        patch_size = 16
        model = SigLIPFeaturizer("hf-hub:timm/ViT-B-16-SigLIP2", patch_size)
        dim = 768
    else:
        raise ValueError("unknown model: {}".format(name))
    return model, patch_size, dim
