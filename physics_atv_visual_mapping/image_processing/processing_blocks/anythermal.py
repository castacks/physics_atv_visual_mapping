import os
import torch
import cv2
import numpy as np
import torchvision

from physics_atv_visual_mapping.image_processing.processing_blocks.base import ImageProcessingBlock
from physics_atv_visual_mapping.image_processing.loftup.featurizers import get_featurizer
from physics_atv_visual_mapping.feature_key_list import FeatureKeyList

class AnyThermalBlock(ImageProcessingBlock):
    """
    Image processing block that encodes image using AnyThermal
    """

    def __init__(self, input_size, device, models_dir):
        self.device = device
        # load model
        dino_dir = os.path.join(models_dir, "torch_hub", "facebookresearch_dinov2_main")
        model, patch_size, dim = get_featurizer("anythermal", dino_dir)
        self.model = model.to('cuda')
        self.input_size = input_size

        # for saving descriptors for fitting VLAD
        self.save_samples = []
        self.save_count = 1
    
    def preprocess(self, img):
        assert len(img.shape) == 4, 'need to batch images'
        assert img.shape[1] == 3, 'expects channels-first'
        img = img.cuda().float()
        img = torchvision.transforms.functional.resize(img,(self.input_size[1],self.input_size[0]))
        return img
    
    def run(self, image, intrinsics, image_orig):
        with torch.no_grad():
            img = self.preprocess(image)
            img_out = self.model(img)

        ix = image.shape[3]
        dx = img_out.shape[3]
        iy = image.shape[2]
        dy = img_out.shape[2]

        intrinsics[:, 0, 0] *= (dx/ix)
        intrinsics[:, 0, 2] *= (dx/ix)

        intrinsics[:, 1, 1] *= (dy/iy)
        intrinsics[:, 1, 2] *= (dy/iy)

        # save descriptors for vlad
        # save_samples = img_out[0,:,2:-1,:].cpu().flatten(1).numpy() #assuming 16x16
        # save_samples = save_samples.T
        # np.random.seed(42)
        # sample_ids = np.random.choice(len(save_samples), size=100)
        # save_samples = save_samples[sample_ids]
        # self.save_samples.append(save_samples)
        # if self.save_count % 10 == 0:
        #     np.save("/home/tartandriver/tartandriver_ws/anythermal_thermal_16x16_feats", np.array(self.save_samples))
        #     print("_________________________________________", self.save_count)
        # self.save_count += 1
        
        return img_out, intrinsics
    
    @property
    def output_feature_keys(self):
        embed_dim = 768
        return FeatureKeyList(
            label=[f"anythermal_{i}" for i in range(embed_dim)],
            metainfo=["vfm" for i in range(embed_dim)]
        )

if __name__ == '__main__':
    models_dir = '/home/tartandriver/tartandriver_ws/models'
    dino_dir = os.path.join(models_dir, "torch_hub", "facebookresearch_dinov2_main")
    model_file = 'physics_atv_visual_mapping/anythermal.pth'
    model_data = torch.load(os.path.join(models_dir, model_file))
    model_type = model_data['student_model_type']

    # model = torch.hub.load(dino_dir, model_type, source='local', pretrained=False)
    # model.load_state_dict(model_data['student_model_state_dict']['backbone_model_state_dict'])
    # model.to("cuda")

    # print(type(model))

    img = cv2.imread('00000001.png', cv2.IMREAD_UNCHANGED)
    img = np.transpose(img[np.newaxis, :], (0, 3, 1, 2))
    img = torch.from_numpy(img).cuda().float()
    h_resize = ((img.shape[2] // 14) + 1) * 14
    w_resize = ((img.shape[3] // 14) + 1) * 14
    img = torchvision.transforms.functional.resize(img, (h_resize, w_resize))
    img.to("cuda")
    # output = model(img)

    # print(output.shape)

    # print("Model's state_dict:")
    # for param_tensor in model.state_dict():
    #     print(param_tensor, "\t", model.state_dict()[param_tensor].size())
    
    
    # model.load_state_dict(torch.load(os.path.join(models_dir, model_name), weights_only=True))

    # print(model.keys())

    # print(type(model['student_model_state_dict']['backbone_model_state_dict']))

    # print(model['student_model_type'])

    # print(model['student_model_state_dict']['backbone_model_state_dict'].keys())

    # print(model.state_dict().keys())

    model, patch_size, dim = get_featurizer("anythermal", dino_dir)

    model = model.to("cuda")

    output = model(img)

    print(output.shape)


