"""
This script can be used to push the LoftUp upsamplers to the Hugging Face Hub.
"""

import torch

# push loftup-dinov2s
model = torch.hub.load('nielsrogge/loftup:add_hf', 'loftup_dinov2s', pretrained=True)
model.upsampler.push_to_hub("haiwen/loftup-dinov2s")

# push loftup-siglip2
model = torch.hub.load('nielsrogge/loftup:add_hf', 'loftup_siglip2', pretrained=True)
model.upsampler.push_to_hub("haiwen/loftup-siglip2")

# push loftup-siglip
model = torch.hub.load('nielsrogge/loftup:add_hf', 'loftup_siglip', pretrained=True)
model.upsampler.push_to_hub("haiwen/loftup-siglip")

# push loftup-dinov2b
model = torch.hub.load('nielsrogge/loftup:add_hf', 'loftup_dinov2b', pretrained=True)
model.upsampler.push_to_hub("haiwen/loftup-dinov2b")

# push loftup-dinov2s_reg
model = torch.hub.load('nielsrogge/loftup:add_hf', 'loftup_dinov2s_reg', pretrained=True)
model.upsampler.push_to_hub("haiwen/loftup-dinov2s_reg")

# push loftup-dinov2b_reg
model = torch.hub.load('nielsrogge/loftup:add_hf', 'loftup_dinov2b_reg', pretrained=True)
model.upsampler.push_to_hub("haiwen/loftup-dinov2b_reg")

# push loftup-clip
model = torch.hub.load('nielsrogge/loftup:add_hf', 'loftup_clip', pretrained=True)
model.upsampler.push_to_hub("haiwen/loftup-clip")
