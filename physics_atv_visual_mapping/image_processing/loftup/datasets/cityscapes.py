import random
from os.path import join

import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset
import glob
import torchvision.transforms.functional as F

class Cityscapes(Dataset):
    def __init__(self,
                 root,
                 split,
                 transform,
                 target_transform,
                 include_labels=True,
                ):
        super(Cityscapes, self).__init__()
        self.split = split
        self.root = join(root, "cityscapes")
        self.transform = transform
        self.label_transform = target_transform
        self.include_labels = include_labels
        # image file: leftImg8bit/split/*.png
        # semseg file: gtFine/split/*/*labelTrainIds.png
        assert self.split in ["train", "val", "train+val"]
        split_dirs = {
            "train": ["train"],
            "val": ["val"],
            "train+val": ["train", "val"]
        }

        self.image_files = []
        self.label_files = []
        for split_dir in split_dirs[self.split]:
            for img in glob.glob(join(self.root, "leftImg8bit", split_dir, "*", "*.png")):
                self.image_files.append(img)
                label = img.replace("leftImg8bit", "gtFine").split(".")[0] + "_labelTrainIds.png"
                self.label_files.append(label)

    def __len__(self):
        return len(self.image_files)

    def __getitem__(self, index):
        image_path = self.image_files[index]
        label_path = self.label_files[index]
        seed = np.random.randint(2147483647)
        batch = {}

        random.seed(seed)
        torch.manual_seed(seed)
        img = self.transform(Image.open(image_path).convert("RGB"))
        batch["img"] = img
        batch["img_path"] = image_path
        batch["label_path"] = label_path               
        if self.include_labels:
            label = Image.open(label_path)
            label = self.label_transform(label).squeeze(0).to(torch.int32)
            label[label == 255] = -1
            batch["label"] = label

        return batch
