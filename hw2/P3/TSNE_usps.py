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
        # transforms.Grayscale(1),
        transforms.Resize(28),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5], std=[0.5]),
    ]
)

mnist_dir = "hw2_data/digits/mnistm/"
svhn_dir = "hw2_data/digits/svhn/"
usps_dir = "hw2_data/digits/usps/"

src_val = ImageDataset(file_path=mnist_dir, transform=transform, datatype="val")
target_val = ImageDataset(file_path=usps_dir, transform=transform, datatype="val")

batch_size = 256
src_val_loader = DataLoader(
    dataset=src_val,
    batch_size=batch_size,
    shuffle=False,
    num_workers=4,
)

target_val_loader = DataLoader(
    dataset=target_val,
    batch_size=batch_size,
    shuffle=False,
    num_workers=4,
)

pt_files = sorted([x for x in os.listdir("P3/model_usps") if x.endswith(".pt")])

for pt_file in pt_files:
    dann = DANN(FeatureExtractor(), DomainRegressor(), Classifier()).to(device)
    pt_usps_dir = "P3/usps_84.946.pt"
    pt_svhn_dir = "P3/svhn_44.313.pt"
    pt_base_dir = "P3/basenet_95.732.pt"
    svhn_dir_pt_path = "P3/model_usps/" + pt_file
    print(f"Load {pt_file} to plot...")
    dann.load_state_dict(torch.load(svhn_dir_pt_path))

    # 1. extract_features from conv3
    def extract_features(model, dataloader, domain):
        hook_list = []

        def hook_fn(module, input, output):
            hook_list.append(output)

        target_layer = dann.feature_extractor.conv3
        hook = target_layer.register_forward_hook(hook_fn)
        labels_list = []
        Domain_list = []
        with torch.no_grad():
            for images, labels in dataloader:
                _ = model(images.to(device))
                labels_list.extend(labels.tolist())
                Domain_list.extend([domain] * len(labels))
        hook.remove()
        flattened_outputs = torch.cat(hook_list, dim=0)
        size = torch.prod(torch.tensor(flattened_outputs.size()[1:]))
        flattened_outputs = flattened_outputs.reshape(-1, size)

        return (flattened_outputs, labels_list, Domain_list)

    # 2. using tsne to tans to 2ds
    def perform_tsne(features):
        tsne = TSNE(n_components=2, init="pca", learning_rate="auto", n_iter=5000)
        features = tsne.fit_transform(features.cpu())

        return features

    def plot_tsne(tsne_result, labels, domains):
        print("drawing...")
        # color for labels
        plt.figure(figsize=(10, 5))
        # 第一個 subplot
        plt.subplot(1, 2, 1)
        scatter_class = plt.scatter(
            tsne_result[:, 0], tsne_result[:, 1], c=labels, marker=".", s=1
        )
        plt.colorbar(scatter_class, label="Class Colorbar")
        plt.title("t-SNE Colored by Digit Class")

        # 第二個 subplot
        plt.subplot(1, 2, 2)
        scatter_domain = plt.scatter(
            tsne_result[:, 0], tsne_result[:, 1], c=domains, marker=".", s=1
        )
        plt.colorbar(scatter_domain, label="Domain Colorbar")
        plt.title("t-SNE Colored by Domain")

        plt.tight_layout()  # 自動調整 subplot 之間的間距以避免重疊
        plt.savefig(f"P3/usps_pic/{pt_file}_tsne_class.png")

    src_features, src_labels, src_domains = extract_features(dann, src_val_loader, 0)
    target_features, target_labels, target_domains = extract_features(
        dann, target_val_loader, 1
    )

    all_features = torch.cat((src_features, target_features), dim=0).cpu()
    all_labels = src_labels + target_labels
    all_domains = src_domains + target_domains

    all_tsne_result = perform_tsne(all_features)
    src_tsne_result = perform_tsne(src_features)
    target_tsne_result = perform_tsne(target_features)

    plot_tsne(all_tsne_result, all_labels, all_domains)
