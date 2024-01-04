import os
import torch
import torchvision.io as tvio
import torchvision.transforms.functional as fn
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as transforms
from torchvision.transforms import ToPILImage
import matplotlib.pyplot as plt


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
        self.img_name = img_name
        img_path = os.path.join(self.img_dir, img_name)
        img = tvio.read_image(img_path)
        label = img_name.split("_")[0]
        label = torch.tensor(int(label), dtype=torch.float32)

        # img_pil.save(f"DLCV_hw1/image_output/{label}original.png", "PNG")
        if self.transform:
            to_pil = ToPILImage()
            img_pil = to_pil(img)
            # print("trans:",self.transform)
            img = self.transform(img_pil)
            # img_pil = to_pil(img)
            # img_pil.save("DLCV_hw1/image_output/trans.png", "PNG")

        return img, label
