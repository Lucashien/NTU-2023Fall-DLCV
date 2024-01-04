import os
import torch
import torchvision.io as tvio
import torchvision.transforms.functional as fn
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as transforms
from torchvision.transforms import ToPILImage
import matplotlib.pyplot as plt
import random


class ImageDataset(Dataset):
    ## 初始化
    def __init__(
        self,
        img_dir,
        dataset_img_list,
        dataset_mask,
        img_transform=None,
        mask_transform=None,
    ):
        self.dataset_img_list = dataset_img_list
        self.masks = torch.Tensor(dataset_mask)
        self.img_dir = img_dir
        self.img_transform = img_transform
        self.mask_transform = mask_transform
        self.random = 0
        self.isprocessed = []

    ### dataset 長度
    def __len__(self):
        self.datasetlength = len(self.dataset_img_list)
        return self.datasetlength

    def __getitem__(self, idx):
        img_name = self.dataset_img_list[idx]
        self.img_name = img_name
        img_path = os.path.join(self.img_dir, img_name)
        img = tvio.read_image(img_path)
        mask = self.masks[idx]

        if self.img_transform:
            self.random += 10
            random.seed(self.random)
            to_pil = ToPILImage()
            img_pil = to_pil(img)
            img = self.img_transform(img_pil)

        return img, mask
