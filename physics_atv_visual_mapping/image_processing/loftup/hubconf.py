import torch
import torch.nn as nn
from upsamplers.layers import ChannelNorm
from upsamplers.upsamplers import LoftUp, UpsamplerwithChannelNorm
from huggingface_hub import hf_hub_download


def loftup_dinov2s(pretrained=True, progress=True, **kwargs):
    upsampler = LoftUp(384, lr_pe_type="sine", lr_size=16)
    channelnorm = ChannelNorm(384)
    model = UpsamplerwithChannelNorm(upsampler, channelnorm)
    if pretrained:
        model_path = hf_hub_download(repo_id="haiwen/loftup-dinov2s", filename="pytorch_model.bin")
        checkpoint = torch.load(model_path, map_location="cpu")
        model.load_state_dict(checkpoint)
    return model

def loftup_siglip2(pretrained=True, progress=True, **kwargs):
    upsampler = LoftUp(768, lr_pe_type="sine", lr_size=14)
    channelnorm = ChannelNorm(768)
    model = UpsamplerwithChannelNorm(upsampler, channelnorm)
    if pretrained:
        model_path = hf_hub_download(repo_id="haiwen/loftup-siglip2", filename="pytorch_model.bin")
        checkpoint = torch.load(model_path, map_location="cpu")
        model.load_state_dict(checkpoint)
    return model

def loftup_siglip(pretrained=True, progress=True, **kwargs):
    upsampler = LoftUp(768, lr_pe_type="sine", lr_size=14)
    channelnorm = ChannelNorm(768)
    model = UpsamplerwithChannelNorm(upsampler, channelnorm)
    if pretrained:
        model_path = hf_hub_download(repo_id="haiwen/loftup-siglip", filename="pytorch_model.bin")
        checkpoint = torch.load(model_path, map_location="cpu")
        model.load_state_dict(checkpoint)
    return model

def loftup_dinov2b(pretrained=True, progress=True, **kwargs):
    upsampler = LoftUp(768, lr_pe_type="sine", lr_size=14)
    channelnorm = ChannelNorm(768)
    model = UpsamplerwithChannelNorm(upsampler, channelnorm)
    if pretrained:
        model_path = hf_hub_download(repo_id="haiwen/loftup-dinov2b", filename="pytorch_model.bin")
        checkpoint = torch.load(model_path, map_location="cpu")
        model.load_state_dict(checkpoint)
    return model

def loftup_dinov2s_reg(pretrained=True, progress=True, **kwargs):
    upsampler = LoftUp(384, lr_pe_type="sine", lr_size=16)
    channelnorm = ChannelNorm(384)
    model = UpsamplerwithChannelNorm(upsampler, channelnorm)
    if pretrained:
        model_path = hf_hub_download(repo_id="haiwen/loftup-dinov2s_reg", filename="pytorch_model.bin")
        checkpoint = torch.load(model_path, map_location="cpu") 
        model.load_state_dict(checkpoint)
    return model

def loftup_dinov2b_reg(pretrained=True, progress=True, **kwargs):
    upsampler = LoftUp(768, lr_pe_type="sine", lr_size=14)
    channelnorm = ChannelNorm(768)
    model = UpsamplerwithChannelNorm(upsampler, channelnorm)
    if pretrained:
        model_path = hf_hub_download(repo_id="haiwen/loftup-dinov2b_reg", filename="pytorch_model.bin")
        checkpoint = torch.load(model_path, map_location="cpu")
        model.load_state_dict(checkpoint)
    return model

def loftup_clip(pretrained=True, progress=True, **kwargs):
    upsampler = LoftUp(512, lr_pe_type="sine", lr_size=14)
    channelnorm = ChannelNorm(512)
    model = UpsamplerwithChannelNorm(upsampler, channelnorm)
    if pretrained:
        model_path = hf_hub_download(repo_id="haiwen/loftup-clip", filename="pytorch_model.bin")
        checkpoint = torch.load(model_path, map_location="cpu")
        model.load_state_dict(checkpoint)
    return model
