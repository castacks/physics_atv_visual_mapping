import numpy as np
from torch.utils.data import Dataset
import torch

class EmbeddingFile(Dataset):
    """
    modified from: https://pytorch.org/docs/stable/_modules/torchvision/datasets/folder.html#ImageFolder
    uses cached directory listing if available rather than walking directory
     Attributes:
        classes (list): List of the class names.
        class_to_idx (dict): Dict with items (class_name, class_index).
        samples (list): List of (sample path, class_index) tuples
        targets (list): The class_index value for each image in the dataset
    """

    def __init__(self, file, loading_imgs, num_limit=60000):
        super(Dataset, self).__init__()
        self.file = file
        loaded = np.load(file)
        self.feats = loaded["feats"][:num_limit]
        self.labels = loaded["labels"][:num_limit]
        if loading_imgs:
            self.imgs = loaded["imgs"][:num_limit]
        else:
            self.imgs = [0] * len(self.labels)

    def dim(self):
        return self.feats.shape[1]

    def num_classes(self):
        return self.labels.max() + 1

    def __getitem__(self, index):
        return self.imgs[index], self.feats[index], self.labels[index]

    def __len__(self):
        return len(self.labels)


class EmbeddingAndImage(Dataset):
    def __init__(self, file, dataset):
        super(Dataset, self).__init__()
        self.file = file
        loaded = np.load(file)
        self.feats = loaded["feats"]
        self.labels = loaded["labels"]
        num_imgs = len(dataset)
        img_shape = dataset[0]["img"].shape
        self.imgs = torch.empty((num_imgs, *img_shape))
        # if dataset[0] is a dict, then only use the "img" key to create a list
        for i, d in enumerate(dataset):
            self.imgs[i] = d["img"]
        ### NOTE: TOO SLOW...

    def dim(self):
        return self.feats.shape[1]

    def num_classes(self):
        return self.labels.max() + 1

    def __getitem__(self, index):
        return self.feats[index], self.labels[index], self.imgs[index]

    def __len__(self):
        return len(self.labels)
