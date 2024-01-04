import os
import torch
import torchvision.io as tvio
import torchvision.transforms.functional as fn
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as trns
from torchvision.transforms import ToPILImage
import matplotlib.pyplot as plt
import random
import torchvision.transforms.functional as TF
import numpy as np
import imageio.v2 as imageio
from PIL import Image


class ImageDataset(Dataset):
    def __init__(self, file_path, img_transform=None, isFlip=True):
        self.path = file_path
        self.data = []

        self.imgfile = sorted(
            [img for img in os.listdir(self.path) if img.endswith(".jpg")]
        )
        self.maskfile = sorted(
            [mask for mask in os.listdir(self.path) if mask.endswith(".png")]
        )
        self.pmask = np.empty((len(self.maskfile), 512, 512))

        if len(self.maskfile):
            for i, (img, mask) in enumerate(zip(self.imgfile, self.maskfile)):
                path = os.path.join(self.path, img)
                self.data.append(Image.open(path).copy())
                mask = imageio.imread(os.path.join(self.path, mask)).copy()
                mask = (mask >= 128).astype(int)
                mask = 4 * mask[:, :, 0] + 2 * mask[:, :, 1] + mask[:, :, 2]
                self.pmask[i, mask == 3] = 0  # (Cyan: 011) Urban land
                self.pmask[i, mask == 6] = 1  # (Yellow: 110) Agriculture land
                self.pmask[i, mask == 5] = 2  # (Purple: 101) Rangeland
                self.pmask[i, mask == 2] = 3  # (Green: 010) Forest land
                self.pmask[i, mask == 1] = 4  # (Blue: 001) Water
                self.pmask[i, mask == 7] = 5  # (White: 111) Barren land
                self.pmask[i, mask == 0] = 6  # (Black: 000) Unknown

            else:
                for img in self.imgfile:
                    self.data.append(Image.open(os.path.join(self.path, img)).copy())
        self.transform = img_transform
        self.isflip = isFlip

    def __getitem__(self, idx):
        data = self.data[idx]
        data = self.transform(data)

        if len(self.pmask):
            pmask = self.pmask[idx].copy()
            if np.random.rand() > 0.5 and self.isflip:
                pmask = np.flip(pmask, axis=1)
                data = trns.functional.hflip(data)
            if np.random.rand() > 0.5 and self.isflip:
                pmask = np.flip(pmask, axis=0)
                data = trns.functional.vflip(data)
            return data, pmask.copy()
        else:
            return data

    def __len__(self):
        return len(self.imgfile)
