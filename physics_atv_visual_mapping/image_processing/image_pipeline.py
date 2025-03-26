import torch
import numpy as np

from physics_atv_visual_mapping.image_processing.processing_blocks.dino import (
    Dinov2Block,
)
from physics_atv_visual_mapping.image_processing.processing_blocks.radio import (
    RadioBlock,
)
from physics_atv_visual_mapping.image_processing.processing_blocks.pca import PCABlock
from physics_atv_visual_mapping.image_processing.processing_blocks.vlad import VLADBlock
from physics_atv_visual_mapping.image_processing.processing_blocks.pca_vlad import (
    PCAVLADBlock,
)
# from physics_atv_visual_mapping.image_processing.processing_blocks.ganav import GANavBlock

from physics_atv_visual_mapping.utils import normalize_dino
import os


def setup_image_pipeline(config):
    blocks = []
    pipeline = ImagePipeline(blocks, config["device"])

    for block_config in config["image_processing"]:
        btype = block_config["type"]
        block_config["args"]["device"] = config["device"]
        if btype == "dino":
            block = Dinov2Block(**block_config["args"], models_dir=config["models_dir"])
        elif btype == "sam":
            block = SAMBlock(**block_config["args"], models_dir=config["models_dir"])
        elif btype == "radio":
            block = RadioBlock(**block_config["args"], models_dir=config["models_dir"])
        elif btype == "pca":
            block = PCABlock(**block_config["args"], models_dir=config["models_dir"])
        elif btype == "vlad":
            block = VLADBlock(**block_config["args"], models_dir=config["models_dir"])
        elif btype == "pca_vlad":
            block = PCAVLADBlock(
                **block_config["args"], models_dir=config["models_dir"]
            )
        # elif btype == 'ganav':
        #     block = GANavBlock(**block_config['args'])
        else:
            print("Unsupported visual block type {}".format(btype))
            exit(1)

        blocks.append(block)

    return pipeline


class ImagePipeline:
    """
    Wrapper class for image proc. Essentially a cascaded set of image proc fns
    """

    def __init__(self, blocks, device):
        self.blocks = blocks
        self.device = device

    def run(self, image, intrinsics):
        image = image.to(self.device).float()
        image_feats = image.clone()
        intrinsics_out = intrinsics.clone()

        for block in self.blocks:
            image_feats, intrinsics_out = block.run(image_feats, intrinsics_out, image)

        return image_feats, intrinsics_out

    def __repr__(self):
        out = "ImagePipeline with:\n"
        for block in self.blocks:
            out += "\t" + str(block) + "\n"
        return out


if __name__ == "__main__":
    import os
    import cv2
    import yaml
    import time
    import matplotlib.pyplot as plt

    #    config_fp = '/home/physics_atv/physics_atv_ws/src/perception/physics_atv_visual_mapping/config/ros/debug_frontends.yaml'
    config_fp = "/home/physics_atv/physics_atv_ws/src/perception/physics_atv_visual_mapping/config/ros/sam_pca.yaml"
    config = yaml.safe_load(open(config_fp, "r"))

    #    img_dir = '/home/physics_atv/workspace/datasets/dino_contrastive_eval/turnpike_flat/image_left_color/'
    img_dir = "/home/physics_atv/workspace/datasets/dino_contrastive_test/fig8/image_left_color/"
    img_fps = sorted(os.listdir(img_dir))

    K_orig = torch.tensor(config["intrinsics"]["K"]).reshape(1, 3, 3)

    pipeline = setup_image_pipeline(config)

    for ifp in img_fps[::100]:
        img = cv2.imread(os.path.join(img_dir, ifp))
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB) / 255.0
        img = torch.tensor(img).unsqueeze(0).permute(0, 3, 1, 2)

        t1 = time.time()
        img_feats, K = pipeline.run(img, K_orig)
        t2 = time.time()

        print("img_size_orig:", img.shape)
        print("img_size_out:", img_feats.shape)
        print("intrinsics_orig:", K_orig)
        print("intrinsics_out:", K)

        fig, axs = plt.subplots(1, 2)
        axs[0].imshow(img[0].permute(1, 2, 0))
        axs[1].imshow(normalize_dino(img_feats[0].permute(1, 2, 0)).cpu())
        #        axs[1].imshow(img_feats[0].argmin(dim=0).cpu(), cmap='tab10')
        #        axs[1].imshow(img_feats[0].argmax(dim=0).cpu(), cmap='tab10')
        fig.suptitle("proc took {:.4f}s".format(t2 - t1))
        plt.show()
