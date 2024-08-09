import os
import yaml
import torch
import rospkg
import torchvision
from torchvision import transforms
import torch.nn.functional as F
# from ptflops import get_model_complexity_info
# from thop import profile

from physics_atv_visual_mapping.image_processing.processing_blocks.base import ImageProcessingBlock
# import visual_feature_distiller

class DistilledModelBlock(ImageProcessingBlock):
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
        self.image_insize = train_config['image_insize']
        self.normalise_rgb = train_config.get('normalise_rgb', False)
        self.teacher_type = train_config['image_processing'][0]['type']
        
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
            transforms.Resize((self.image_insize[1], self.image_insize[0]))
        ])
        
        self.device = device
        
        self.student_model = torch.load(model_path).to(device)
        self.student_model.eval()
        
        # input_tensor = torch.randn(1, 3, self.image_insize[1], self.image_insize[0]) .to(device)  # Replace with appropriate input size
        # flops, params = profile(self.student_model, inputs=(input_tensor,))
        # print(f"FLOPs: {flops}")
        # print(f"Params: {params}")
        
        # input_size = (3, self.image_insize[1], self.image_insize[0]) 
        # flops, params = get_model_complexity_info(self.student_model, input_size, as_strings=True, print_per_layer_stat=True)

        # print("Distilled Radio")
        # print(f"FLOPs: {flops}")
        # print(f"Params: {params}")


    def preprocess(self, img):
        assert len(img.shape) == 4, 'need to batch images'
        assert img.shape[1] == 3, 'expects channels-first'
        img = img.cuda().float()
        img = torchvision.transforms.functional.resize(img,(self.image_insize[1],self.image_insize[0]))
        return img
        # img = self.transform(img[0])
        # return img.unsqueeze(0)
        # transformed_images = []
        # # HACK but realistically only 1 image at a time
        # for single_img in img:
        #     transformed_img = self.transform(single_img)
        #     transformed_images.append(transformed_img)

        # transformed_images_batch = torch.stack(transformed_images)
        # return transformed_images_batch

    def run(self, image, intrinsics, image_orig):
        with torch.no_grad():
            img = self.preprocess(image)
            if self.normalise_rgb:
                normalised_images = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])(img)
                student_features = self.student_model(normalised_images)
            else:
                student_features = self.student_model(img)

            if self.teacher_type == 'dino' or self.teacher_type == 'radio':
                student_features = F.normalize(student_features, dim=1)

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
