import os
import torch
import torchvision
import torch.nn.functional as F

# from ptflops import get_model_complexity_info
# from thop import profile

from physics_atv_visual_mapping.image_processing.processing_blocks.base import (
    ImageProcessingBlock,
)


class RadioBlock(ImageProcessingBlock):
    """
    Image processing block that runs dino on the image
    """

    def __init__(self, radio_type, image_insize, models_dir, device):
        self.input_size = image_insize
        self.output_size = (int(image_insize[0] / 16), int(image_insize[1] / 16))
        self.radio_type = radio_type

        # call this to run from local
        torch.hub.set_dir(os.path.join(models_dir, "torch_hub"))
        radio_fp = os.path.join(models_dir, "torch_hub", "NVlabs_RADIO_main")
        radio = torch.hub.load(
            radio_fp,
            "radio_model",
            version=radio_type,
            progress=True,
            skip_validation=True,
            source="local",
        )  #  force_reload=True

        # call this to download (TODO: make a download models script)
        # radio = torch.hub.load('NVlabs/RADIO', 'radio_model', version=radio_type, progress=True, skip_validation=True, source='github') #  force_reload=True

        if "e-radio" in radio_type:
            radio.model.set_optimal_window_size([image_insize[1], image_insize[0]])

        self.radio = radio.to(device).eval()

    def preprocess(self, img):
        assert len(img.shape) == 4, "need to batch images"
        assert img.shape[1] == 3, "expects channels-first"
        img = img.cuda().float()
        img = torchvision.transforms.functional.resize(
            img, (self.input_size[1], self.input_size[0])
        )
        return img

    def run(self, image, intrinsics, image_orig):

        with torch.no_grad():
            img = self.preprocess(image)
            summary, img = self.radio(img)
            img = F.normalize(img, dim=-1)
            img_out = img.view(
                img.shape[0], self.output_size[1], self.output_size[0], -1
            ).permute(0, 3, 1, 2)

        ix = image.shape[3]
        dx = img_out.shape[3]
        iy = image.shape[2]
        dy = img_out.shape[2]

        intrinsics[:, 0, 0] *= dx / ix
        intrinsics[:, 0, 2] *= dx / ix

        intrinsics[:, 1, 1] *= dy / iy
        intrinsics[:, 1, 2] *= dy / iy

        return img_out, intrinsics
