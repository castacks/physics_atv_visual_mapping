import os
import torch
import rospkg
import torchvision
import torch.nn.functional as F

from physics_atv_visual_mapping.image_processing.anyloc_utils import DinoV2ExtractFeatures
from physics_atv_visual_mapping.image_processing.processing_blocks.base import ImageProcessingBlock

class RadioBlock(ImageProcessingBlock):
    """
    Image processing block that runs dino on the image
    """
    def __init__(self, image_insize, device):
        rp = rospkg.RosPack()
        dino_dir = os.path.join(rp.get_path("physics_atv_visual_mapping"), "models/hub")

        # self.radio = DinoV2ExtractFeatures(dino_dir,
        #     dino_model=dino_type,
        #     layer=dino_layer,
        #     input_size=image_insize,
        #     facet=desc_facet,
        #     device=device
        # )
        self.input_size = image_insize
        self.output_size = (int(image_insize[0]/16),int(image_insize[1]/16))
        # model_version="radio_v2.1" # for RADIO
        model_version="e-radio_v2" # for E-RADIO
        radio = torch.hub.load('NVlabs/RADIO', 'radio_model', version=model_version, progress=True, skip_validation=True)
        # print(self.output_size)
        radio.model.set_optimal_window_size([image_insize[1], image_insize[0]])

        self.radio = radio.to(device)#.eval()


    def preprocess(self, img):
        assert len(img.shape) == 4, 'need to batch images'
        assert img.shape[1] == 3, 'expects channels-first'
        img = img.cuda().float()
        img = torchvision.transforms.functional.resize(img,(self.input_size[1],self.input_size[0]))
        return img

    def run(self, image, intrinsics, image_orig):

        with torch.no_grad():
            img = self.preprocess(image)
            # print(img.dtype)
            summary, img = self.radio(img)
            # print(img.dtype)
            # print(img.shape)
            img = F.normalize(img, dim=-1)
            #TODO should we normalize?
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
