import os
import torchvision.io as tvio
import torchvision.transforms.functional as fn
from torch.utils.data import Dataset
from torchvision.transforms import ToPILImage
import torch
from PIL import Image
import torchvision.transforms as trns


class MiniDataset(Dataset):
    ## 初始化
    def __init__(self, img_dir, dataset_list, transform, target_transform=None):
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
        if self.transform:
            to_pil = ToPILImage()
            img_pil = to_pil(img)
            # img_pil.save("DLCV_hw1/2/C/b_image.png")
            img = self.transform(img_pil)
            to_pil = ToPILImage()
            img_pil = to_pil(img)
            # img_pil.save("DLCV_hw1/2/C/a_image.png")

        return img


class ImageDataset(Dataset):
    def __init__(self, file_path, transform=None):
        self.path = file_path
        self.transform = transform
        if transform:
            self.transform = transform
        else:
            self.transform = trns.Compose(
                [
                    trns.Resize([128, 128]),
                    trns.ToTensor(),
                    trns.Normalize(
                        mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
                    ),
                ]
            )

        self.files = sorted([x for x in os.listdir(self.path) if x.endswith(".jpg")])
        self.labels = []
        self.data = []
        for file in self.files:
            if file.endswith(".jpg") and len(file.split("_")) > 1:
                label = file.split("_")[0]
                self.labels.append(int(label))
            else:
                self.labels.append(0)

            self.data.append(Image.open(os.path.join(self.path, file)).copy())

    def __getitem__(self, idx):
        data = Image.open(os.path.join(self.path, self.files[idx]))
        data = self.transform(data)

        if len(self.labels):
            return data, self.labels[idx]
        else:
            return data

    def __len__(self):
        return len(self.files)
