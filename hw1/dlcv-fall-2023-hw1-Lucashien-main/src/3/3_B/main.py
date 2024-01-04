from LoadData import ImageDataset
import numpy as np
import torch
import torch.nn as nn
from torchsummary import summary
import os
from train import fit_model
from torch.optim import Adam, SGD
import matplotlib.pyplot as plt
import torchvision.transforms as trns
from mean_iou_evaluate import read_masks, mean_iou_score
from viz_mask import viz_data
from DeepLabv3 import Deeplabv3_Resnet50_Model
import random

# parameter setting

p_lr = 0.001
p_batch_size = 8
num_epochs = 200
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
img_dir_train = "DLCV_hw1/hw1_data/p3_data/train"
img_dir_val = "DLCV_hw1/hw1_data/p3_data/validation"

###


def same_seeds(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchark = False
    torch.backends.cudnn.deterministic = True


same_seeds(2023)

transform_train = trns.Compose(
    [
        trns.Resize([512, 512]),
        trns.ToTensor(),
        trns.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ]
)

transform_val = trns.Compose(
    [
        trns.Resize([512, 512]),
        trns.ToTensor(),
        trns.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ]
)

train_data = ImageDataset(
    img_dir_train,
    img_transform=transform_train,
)

val_data = ImageDataset(img_dir_val, img_transform=transform_val)

train_loader = torch.utils.data.DataLoader(
    dataset=train_data, batch_size=p_batch_size, shuffle=True, drop_last=True
)

val_loader = torch.utils.data.DataLoader(
    dataset=val_data, batch_size=4, shuffle=True, drop_last=True
)

model = Deeplabv3_Resnet50_Model()
model = model.to(device)

optimizer = torch.optim.SGD(
    model.parameters(), lr=0.001, momentum=0.9, weight_decay=0.0001
)
loss_func = nn.CrossEntropyLoss(ignore_index=6)

# 訓練模型
fit_model(
    model,
    optimizer,
    num_epochs,
    train_loader,
    val_loader,
)
