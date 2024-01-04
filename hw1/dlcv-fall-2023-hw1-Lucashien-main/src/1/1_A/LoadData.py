import os
import torch
import pandas as pd
import torchvision.io as tvio
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader


class ImageDataset(Dataset):
    ## 初始化
    def __init__(self, img_dir, dataset_list, transform=None, target_transform=None):
        self.dataset_list = dataset_list
        self.img_dir = img_dir
        self.transform = transform
        self.target_transform = target_transform

    ### dataset 長度
    def __len__(self):
        self.datasetlength = len(self.dataset_list)
        return self.datasetlength

    def __getitem__(self, idx):
        img_name = self.dataset_list[idx]
        img_path = os.path.join(self.img_dir, img_name)
        img = tvio.read_image(img_path)
        img = F.interpolate(img, 232)
        label = img_name.split("_")[0]
        label = torch.tensor(int(label), dtype=torch.float32)
        return img, label
