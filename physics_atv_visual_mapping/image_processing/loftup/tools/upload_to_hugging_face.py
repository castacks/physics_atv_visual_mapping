import torch
import torch.nn as nn
from huggingface_hub import PyTorchModelHubMixin
from upsamplers.layers import ChannelNorm
from upsamplers.upsamplers import LoftUp, UpsamplerwithChannelNorm
from huggingface_hub import hf_hub_download


class UpsamplerwithChannelNormHub(nn.Module, PyTorchModelHubMixin):
    def __init__(self, upsampler, channelnorm):
        super(UpsamplerwithChannelNormHub, self).__init__()
        self.upsampler = upsampler
        self.channelnorm = channelnorm

    def forward(self, lr_feats, img):
        lr_feats = self.channelnorm(lr_feats)
        return self.upsampler(lr_feats, img)

    def _save_pretrained(self, save_directory, **kwargs):
        # Save weights
        torch.save(self.state_dict(), f"{save_directory}/pytorch_model.bin")

    @classmethod
    def _from_pretrained(cls, model_id, **kwargs):
        # Recreate the model structure manually
        save_directory = model_id
        upsampler = LoftUp(384, lr_pe_type="sine", lr_size=16)
        channelnorm = ChannelNorm(384)
        model = cls(upsampler, channelnorm)
        state_dict = torch.load(f"{save_directory}/pytorch_model.bin", map_location="cpu")
        model.load_state_dict(state_dict)
        return model

## Load weights
def load_loftup_checkpoint(upsampler_path, n_dim, lr_pe_type="sine", lr_size=16):
    channelnorm = ChannelNorm(n_dim)
    upsampler = LoftUp(n_dim, lr_pe_type=lr_pe_type, lr_size=lr_size)
    ckpt_weight = torch.load(upsampler_path)['state_dict']
    channelnorm_checkpoint = {k: v for k, v in ckpt_weight.items() if 'model.1' in k} # dict_keys(['model.1.norm.weight', 'model.1.norm.bias'])
    # change the key names
    channelnorm_checkpoint = {k.replace('model.1.', ''): v for k, v in channelnorm_checkpoint.items()}
    # if the key starts with upsampler, remove the upsampler.
    upsampler_ckpt_weight = {k: v for k, v in ckpt_weight.items() if k.startswith('upsampler')}
    upsampler_ckpt_weight = {k.replace('upsampler.', ''): v for k, v in upsampler_ckpt_weight.items()}
    upsampler.load_state_dict(upsampler_ckpt_weight)
    channelnorm.load_state_dict(channelnorm_checkpoint)
    for param in upsampler.parameters():
        param.requires_grad = False
    for param in channelnorm.parameters():
        param.requires_grad = False
    # return channelnorm, upsampler
    return UpsamplerwithChannelNormHub(upsampler, channelnorm)

upsampler_path = "/mnt/haiwen/pretrained_models/FeatUp/loftup_siglip_stage2.ckpt"
model = load_loftup_checkpoint(upsampler_path, 768, "sine", 14)
## Load weights

# # save locally


# push to the hub
model.push_to_hub("haiwen/loftup-siglip")

# # reload

model_path = hf_hub_download(repo_id="haiwen/loftup-siglip", filename="pytorch_model.bin")
state_dict = torch.load(model_path, map_location="cpu")

upsampler = LoftUp(768, lr_pe_type="sine", lr_size=14)
channelnorm = ChannelNorm(768) 
model = UpsamplerwithChannelNorm(upsampler, channelnorm)
model.load_state_dict(state_dict)
print(model)