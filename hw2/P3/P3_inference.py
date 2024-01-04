import csv
import torch
import os, sys
import numpy as np
from PIL import Image
from tqdm import tqdm
from torchvision import datasets
from torchvision import transforms
from torch.utils.data import DataLoader, Dataset

from models import (
    FeatureExtractor,
    Classifier,
    DomainRegressor,
    GradReverse,
    DANN,
    BaseNet,
)

device = "cuda" if torch.cuda.is_available() else "cpu"


class ImageDataset(Dataset):
    def __init__(self, file_path, transform=None, datatype=None):
        print("Load data from " + file_path)
        self.path = file_path
        self.transform = transform
        self.data = []

        self.imgfile = sorted(
            [img for img in os.listdir(self.path) if img.endswith(".png")]
        )

        for x in self.imgfile:
            self.data.append(os.path.join(self.path, x))

    def __getitem__(self, idx):
        data = Image.open(self.data[idx])
        data = self.transform(data)
        data_3D = torch.zeros(3, 28, 28)

        if data.shape == (1, 28, 28):
            data_3D[0, :, :] = data
            data_3D[1, :, :] = data
            data_3D[2, :, :] = data
            return data_3D, self.imgfile[idx]
        else:
            return data, self.imgfile[idx]

    def __len__(self):
        return len(self.imgfile)


transform = transforms.Compose(
    [
        transforms.Resize(28),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5], std=[0.5]),
    ]
)

dataset_path = sys.argv[1]
csv_path = sys.argv[2]
pt_path = ""

if "svhn" in dataset_path:
    pt_path = "P3/svhn.pt" 
elif "usps" in dataset_path:
    pt_path = "P3/usps.pt" 

target_val = ImageDataset(file_path=dataset_path, transform=transform, datatype="val")

batch_size = 256

target_val_loader = DataLoader(
    dataset=target_val,
    batch_size=batch_size,
    shuffle=False,
    drop_last=False,
    num_workers=4,
)

# %% DANN part
dann = DANN(FeatureExtractor(), DomainRegressor(), Classifier()).to(device)
dann.load_state_dict(torch.load(pt_path))

predict_list = []
img_name_list = []
with torch.no_grad():
    for target_data in tqdm(
        target_val_loader,
    ):
        target_imgs, img_name = target_data[0].to(device), target_data[1]
        target_outputs, _ = dann(target_imgs)
        _, target_predicted = torch.max(target_outputs.data, 1)
        predict_list.extend(target_predicted.flatten().detach().tolist())
        img_name_list.extend(img_name)


np_img_name = np.array(img_name_list, dtype=str)
np_predict = np.array(predict_list, dtype=np.uint8)

with open(csv_path, "w", newline="") as file:
    writer = csv.writer(file)
    writer.writerow(("filename", "label"))
    for data in zip(np_img_name, np_predict):
        writer.writerow(data)

print(f"done")
