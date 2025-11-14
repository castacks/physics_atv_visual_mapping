import torch
import os

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
    elif name == "anythermal":
        from .DINOv2 import DINOv2Featurizer
        patch_size = 14
        models_top_dir = models_dir.split('/')[0]
        model_file = os.path.join('physics_atv_visual_mapping', 'anythermal.pth')
        model_data = torch.load(os.path.join('/home/tartandriver/tartandriver_ws/models', model_file))
        model_type = model_data['student_model_type']
        model = DINOv2Featurizer(models_dir, model_type, patch_size, activation_type)
        state_dictionary = {f"model.{k}": v for k, v in model_data['student_model_state_dict']['backbone_model_state_dict'].items()}
        model.load_state_dict(state_dictionary)
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
