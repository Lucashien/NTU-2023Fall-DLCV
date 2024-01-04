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
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt

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
        return data, self.labels[idx]

    def __len__(self):
        return len(self.files)


transform = transforms.Compose(
    [
        # transforms.Grayscale(1),
        transforms.
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

target_train = ImageDataset(file_path=svhn_dir, transform=transform, datatype="train")
target_val = ImageDataset(file_path=svhn_dir, transform=transform, datatype="val")


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
# %% DANN part
dann = DANN(FeatureExtractor(), DomainRegressor(), Classifier()).to(device)

# Number of epochs
NUM_EPOCH = 101
# Length of an epoch
LEN_EPOCH = min(len(src_train_loader), len(target_train_loader))

# Total steps in the training
total_steps = NUM_EPOCH * LEN_EPOCH

# Define criterions
criterion_classifier = nn.CrossEntropyLoss()
criterion_domain_regressor = nn.CrossEntropyLoss()

# SGD optimizer
optimizer = optim.SGD(
    [
        {"params": dann.feature_extractor.parameters()},
        {"params": dann.classifier.parameters()},
        {"params": dann.domain_regressor.parameters()},
    ],
    lr=0.01,
    momentum=0.9,
)


# Learning rate scheduler
def mu_p(step):
    alpha = 10
    beta = 0.75
    mu_p = 1 / (1 + alpha * step / total_steps) ** beta
    return mu_p


# Virtual learning rate for the domain regressor
def domain_regressor_lr_scheduler(step):
    gamma = 10

    # If step=0, just returns mu_p to avoid division by zero
    if step == 0:
        lambda_p = 1
    else:
        # Compute progress
        p = step / total_steps

        lambda_p = 2 / (1 + np.exp(-gamma * p)) - 1

    return mu_p(step) / lambda_p


# Learning rate scheduler
scheduler = torch.optim.lr_scheduler.LambdaLR(
    optimizer, [mu_p, mu_p, domain_regressor_lr_scheduler]
)

# Initialize progress
p = 0

# Domain targets
src_labels_domain = torch.zeros(batch_size).long().to(device)
target_labels_domain = torch.ones(batch_size).long().to(device)
labels_domain = torch.cat((src_labels_domain, target_labels_domain))
best_acc = 0
for epoch in range(NUM_EPOCH):
    for src_data, target_data in tqdm(
        zip(src_train_loader, target_train_loader),
        desc=f"Epoch {epoch}",
        total=min(len(src_train_loader), len(target_train_loader)),
    ):
        # Update progress
        p += 1 / total_steps

        # Compute the regularization term
        gamma = 10
        lambda_p = 2 / (1 + np.exp(-gamma * p)) - 1

        # Split and transfer to GPU
        src_imgs, src_labels = src_data[0].to(device), src_data[1].to(device)
        target_imgs, target_labels = target_data[0].to(device), target_data[1].to(
            device
        )

        # Source forward pass
        src_class, src_domain = dann(src_imgs)

        # Classifier loss
        class_loss = criterion_classifier(src_class, src_labels)
        # Target forward pass
        _, target_domain = dann(target_imgs)

        # Domain Loss
        preds_domain = torch.cat((src_domain, target_domain))
        domain_loss = criterion_domain_regressor(preds_domain, labels_domain)

        # Total loss
        loss = class_loss.cpu() + lambda_p * domain_loss.cpu()

        # Backward and Optimize
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # Scheduler step
        scheduler.step()

    src_total_train = 0
    target_total_train = 0
    src_correct_train = 0
    target_correct_train = 0
    with torch.no_grad():
        for target_data in tqdm(target_val_loader):
            target_imgs, target_labels = target_data[0].to(device), target_data[1].to(
                device
            )
            target_outputs, _ = dann(target_imgs)
            _, target_predicted = torch.max(target_outputs.data, 1)

            target_total_train += len(target_labels)
            target_correct_train += (target_predicted == target_labels).sum().item()
    target_val_accuracy = 100 * target_correct_train / float(target_total_train)

    with torch.no_grad():
        for src_data in tqdm(
            src_val_loader,
        ):
            src_imgs, src_labels = src_data[0].to(device), src_data[1].to(device)
            src_outputs, _ = dann(src_imgs)
            _, src_predicted = torch.max(src_outputs.data, 1)

            src_total_train += len(src_labels)
            src_correct_train += (src_predicted == src_labels).sum().item()

    src_val_accuracy = 100 * src_correct_train / float(src_total_train)
    with open("P3/svhn.txt", "a") as file:
        sys.stdout = file
        print(
            f"Epoch:{epoch} | src val acc:{src_val_accuracy:3f}%, target acc: {target_val_accuracy:3f}%"
        )
    sys.stdout = sys.__stdout__
    print(f"target: {target_correct_train}/{target_total_train}")
    print(f"[src] acc: {src_val_accuracy:3f}%, loss: {loss:3f}")
    print(f"[target] acc: {target_val_accuracy:3f}%, loss: {loss:3f}")

    if epoch % 5 == 0:
        torch.save(dann.state_dict(), f"P3/svhn_{epoch}.pt")
        # best_acc = target_val_accuracy
