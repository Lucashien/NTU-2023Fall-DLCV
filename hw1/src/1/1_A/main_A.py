from LoadData import ImageDataset
import numpy as np
import torch
import torch.nn as nn
from torchsummary import summary
from CNN_model_A import CNN
import os
from train import fit_model
from torch.optim import Adam
import matplotlib.pyplot as plt
from torchvision.models import resnet50, ResNet50_Weights


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

img_dir_train = "DLCV_hw1/hw1_data/p1_data/train_50"
img_dir_val = "DLCV_hw1/hw1_data/p1_data/val_50"

train_dataset = [
    imgfile for imgfile in os.listdir(img_dir_train) if imgfile.endswith(".png")
]

val_dataset = [
    imgfile for imgfile in os.listdir(img_dir_val) if imgfile.endswith(".png")
]

train_data = ImageDataset(img_dir_train, train_dataset)
val_data = ImageDataset(img_dir_val, val_dataset)

train_loader = torch.utils.data.DataLoader(
    dataset=train_data, batch_size=32, shuffle=True
)

val_loader = torch.utils.data.DataLoader(dataset=val_data, batch_size=32, shuffle=True)

model = CNN().to(device)

optimizer = torch.optim.Adam(model.parameters(), lr=0.0003)
loss_func = nn.CrossEntropyLoss()
num_epochs = 80

# 訓練模型
training_loss, training_accuracy, validation_loss, validation_accuracy = fit_model(
    model, loss_func, optimizer, num_epochs, train_loader, val_loader
)
