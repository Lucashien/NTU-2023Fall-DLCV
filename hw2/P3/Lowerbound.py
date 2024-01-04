import torch
import torchvision
import torch.nn as nn

from torch.utils.data import DataLoader, Dataset
from torchvision import datasets
from torchvision import transforms

import numpy as np
import os, sys
from PIL import Image
import csv
from tqdm import tqdm
import torch.optim as optim
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
        self.path = file_path + "data"
        self.transform = transform
        self.data = []

        self.imgfile = sorted(
            [img for img in os.listdir(self.path) if img.endswith(".png")]
        )
        if datatype == "train":
            self.csv_path = file_path + "train.csv"
        elif datatype == "val":
            self.csv_path = file_path + "val.csv"
        elif datatype == "test":
            self.csv_path = file_path + "tests.csv"
        else:
            self.csv_path = None

        self.imgname_csv = []
        self.labels_csv = []
        self.files = []
        self.labels = []
        with open(self.csv_path, "r", newline="") as file:
            reader = csv.reader(file, delimiter=",")
            next(reader)
            for row in reader:
                img_name, label = row
                self.imgname_csv.append(img_name)
                self.labels_csv.append(torch.tensor(int(label)))

        for x in os.listdir(self.path):
            if x.endswith(".png") and x in self.imgname_csv:
                self.files.append(os.path.join(self.path, x))
                self.labels.append(self.labels_csv[self.imgname_csv.index(x)])

    def __getitem__(self, idx):
        data = Image.open(self.files[idx])
        data = self.transform(data)
        data_3D = torch.zeros(3, 28, 28)

        if data.shape == (1, 28, 28):
            data_3D[0, :, :] = data
            data_3D[1, :, :] = data
            data_3D[2, :, :] = data
            return data_3D, self.labels[idx]
        else:
            return data, self.labels[idx]

    def __len__(self):
        return len(self.files)


transform = transforms.Compose(
    [
        transforms.Resize(28),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5], std=[0.5]),
    ]
)

mnist_dir = "hw2_data/digits/mnistm/"
svhn_dir = "hw2_data/digits/svhn/"
usps_dir = "hw2_data/digits/usps/"

src_train = ImageDataset(file_path=mnist_dir, transform=transform, datatype="train")
src_val = ImageDataset(file_path=mnist_dir, transform=transform, datatype="val")

target_train = ImageDataset(file_path=usps_dir, transform=transform, datatype="train")
target_val = ImageDataset(file_path=usps_dir, transform=transform, datatype="val")


batch_size = 256
src_train_loader = DataLoader(
    dataset=src_train,
    batch_size=batch_size,
    shuffle=True,
    drop_last=True,
    num_workers=4,
)
src_val_loader = DataLoader(
    dataset=src_val,
    batch_size=batch_size,
    shuffle=False,
    drop_last=False,
    num_workers=4,
)

target_train_loader = DataLoader(
    dataset=target_train,
    batch_size=batch_size,
    shuffle=True,
    drop_last=True,
    num_workers=4,
)
target_val_loader = DataLoader(
    dataset=target_val,
    batch_size=batch_size,
    shuffle=False,
    drop_last=False,
    num_workers=4,
)

basenet = BaseNet(FeatureExtractor(), Classifier()).to(device)
criterion = nn.NLLLoss()
optimizer = optim.SGD(basenet.parameters(), lr=0.001, momentum=0.9)

total_train, correct_train = 0, 0

for epoch in range(100):
    print(f"Training epoch {epoch}...")

    for data in tqdm(src_train_loader, desc=f"Epoch = {epoch}"):
        inputs, labels = data[0].to(device), data[1].to(device)

        outputs = basenet(inputs)
        loss = criterion(outputs, labels)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        _, predicted = torch.max(outputs.data, 1)
        total_train += len(labels)
        correct_train += (predicted == labels).sum().item()

    train_accuracy = 100 * correct_train / float(total_train)

    total_val, correct_val = 0, 0
    with torch.no_grad():
        for data in tqdm(src_val_loader, desc=f"Epoch = {epoch}"):
            inputs, labels = data[0].to(device), data[1].to(device)
            outputs = basenet(inputs)
            _, predicted = torch.max(outputs.data, 1)
            total_val += len(labels)
            correct_val += (predicted == labels).sum().item()
        val_accuracy = 100 * correct_val / float(total_val)

    with open("P3/basenet_output.txt", "a") as file:
        sys.stdout = file
        print(
            f"Epoch:{epoch} | src train acc: {train_accuracy:3f}%, target acc: {val_accuracy:3f}%"
        )

    if val_accuracy >90:
        torch.save(basenet.state_dict(), f"P3/basenet_{val_accuracy:.3f}.pt")

    sys.stdout = sys.__stdout__
    print(f"src train acc: {train_accuracy:3f}%, target acc: {val_accuracy:3f}%")
