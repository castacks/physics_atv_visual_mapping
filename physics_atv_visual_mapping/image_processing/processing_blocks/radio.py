import os
import torch
import rospkg
import torchvision
import torch.nn.functional as F
# from ptflops import get_model_complexity_info
# from thop import profile

from physics_atv_visual_mapping.image_processing.processing_blocks.base import ImageProcessingBlock

class RadioBlock(ImageProcessingBlock):
    """
    Image processing block that runs dino on the image
    """
    def __init__(self, radio_type, image_insize, device):
        self.input_size = image_insize
        self.output_size = (int(image_insize[0]/16),int(image_insize[1]/16))
        self.radio_type = radio_type
        
        radio = torch.hub.load('NVlabs/RADIO', 'radio_model', version=radio_type, progress=True, skip_validation=True) #  force_reload=True
        if "e-radio" in radio_type:
            radio.model.set_optimal_window_size([image_insize[1], image_insize[0]])

        self.radio = radio.to(device).eval()
        
        # input_tensor = torch.randn(1, 3, self.input_size[1], self.input_size[0]) .to(device)  # Replace with appropriate input size
        # flops, params = profile(self.radio, inputs=(input_tensor,))
        # print(f"FLOPs: {flops}")
        # print(f"Params: {params}")
        
        # input_size = (3, self.input_size[1], self.input_size[0]) 
        # flops, params = get_model_complexity_info(self.radio, input_size, as_strings=True, print_per_layer_stat=True)

        # print(f"Radio Type: {radio_type}")
        # print(f"FLOPs: {flops}")
        # print(f"Params: {params}")
        
    def preprocess(self, img):
        assert len(img.shape) == 4, 'need to batch images'
        assert img.shape[1] == 3, 'expects channels-first'
        img = img.cuda().float()
        img = torchvision.transforms.functional.resize(img,(self.input_size[1],self.input_size[0]))
        return img

    def run(self, image, intrinsics, image_orig):

        with torch.no_grad():
            img = self.preprocess(image)
            summary, img = self.radio(img)
            # print("radio shape, dtype:", img.shape, img.dtype)
            # print("radio min value:", torch.min(img[0,:,:]))
            # print("radio max value:", torch.max(img[0,:,:]))
            # print("radio mean value:", torch.mean(img[0,:,:]))
            # print("radio std value:", torch.std(img[0,:,:]))      
            #TODO should we normalize? # NO! already layer normed
            img = F.normalize(img, dim=-1)
            # print("radio min value normalised:", torch.min(img[0,:,:]))
            # print("radio max value normalised:", torch.max(img[0,:,:]))
            # print("radio mean value normalised:", torch.mean(img[0,:,:]))
            # print("radio std value normalised:", torch.std(img[0,:,:]))
            img_out = img.view(img.shape[0], self.output_size[1], self.output_size[0], -1).permute(0,3,1,2)

        ix = image.shape[3]
        dx = img_out.shape[3]
        iy = image.shape[2]
        dy = img_out.shape[2]

        intrinsics[:, 0, 0] *= (dx/ix)
        intrinsics[:, 0, 2] *= (dx/ix)

        intrinsics[:, 1, 1] *= (dy/iy)
        intrinsics[:, 1, 2] *= (dy/iy)

        return img_out, intrinsics