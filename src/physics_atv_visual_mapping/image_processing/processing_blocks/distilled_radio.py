import os
import yaml
import torch
import rospkg
from torchvision import transforms
import torch.nn.functional as F

from physics_atv_visual_mapping.image_processing.processing_blocks.base import ImageProcessingBlock
# import visual_feature_distiller

class DistilledRadioBlock(ImageProcessingBlock):
    """
    Image processing block that runs dino on the image
    """
    def __init__(self, run_path, best, device):
        rp = rospkg.RosPack()
        self.run_path = run_path
        model_path = os.path.join(run_path, f"weights/{'best' if best else 'last'}.pth")
        
        train_config_path = os.path.join(run_path, "train_config.yaml")
        train_config = yaml.safe_load(open(train_config_path, 'r'))
        
        image_size = train_config['image_size']
        self.crop_w = train_config['crop_w']
        self.crop_h_low = train_config['crop_h_low']
        self.crop_h_high = train_config['crop_h_high']
        image_insize = train_config['image_insize']
        
        # HACK lifted from train_util.py in visual distllation 
        class CustomCropTransform(object):
            def __init__(self, crop_w, crop_h_low, crop_h_high):
                self.crop_w = crop_w
                self.crop_h_low = crop_h_low
                self.crop_h_high = crop_h_high

            def __call__(self, img):
                c, h, w = img.shape
                img_cropped = img[:, self.crop_h_low:h-self.crop_h_high, self.crop_w:w-self.crop_w]
                return img_cropped
                    
        self.transform = transforms.Compose([
            # transforms.ToTensor(),
            CustomCropTransform(crop_w=self.crop_w, crop_h_low=self.crop_h_low, crop_h_high=self.crop_h_high),
            transforms.Resize((image_insize[1], image_insize[0]))
        ])
        
        self.device = device
        
        self.student_model = torch.load(model_path).to(device)
        self.student_model.eval()


    def preprocess(self, img):
        assert len(img.shape) == 4, 'need to batch images'
        assert img.shape[1] == 3, 'expects channels-first'
        img = img.cuda().float()
        img = self.transform(img[0])
        return img.unsqueeze(0)

    def run(self, image, intrinsics, image_orig):

        with torch.no_grad():
            img = self.preprocess(image)
            efficientnet_output, student_features, student_img = self.student_model(img)
            student_features = F.normalize(student_features, dim=-1)

        # crop intrinsics
        intrinsics[:, 0, 2] -= self.crop_w
        intrinsics[:, 1, 2] -= self.crop_h_low
        
        # scale intrinsics
        ix = image.shape[3] - 2 * self.crop_w
        dx = student_features.shape[3]
        iy = image.shape[2] - self.crop_h_low - self.crop_h_high
        dy = student_features.shape[2]

        intrinsics[:, 0, 0] *= (dx/ix)
        intrinsics[:, 0, 2] *= (dx/ix)

        intrinsics[:, 1, 1] *= (dy/iy)
        intrinsics[:, 1, 2] *= (dy/iy)

        return student_features, intrinsics
