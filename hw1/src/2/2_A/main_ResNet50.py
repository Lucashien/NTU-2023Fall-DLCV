from LoadData import ImageDataset
import numpy as np
import torch
import torch.nn as nn
from torchsummary import summary
import os
from train import fit_model
from torch.optim import Adam, SGD
import matplotlib.pyplot as plt
from torchvision.models import resnet50, ResNet50_Weights
import torchvision.transforms as trns

# parameter setting
p_lr = 0.0005
p_batch_size = 128
num_epochs = 50
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
img_dir_train = "DLCV_hw1/hw1_data/p2_data/office/train"
img_dir_val = "DLCV_hw1/hw1_data/p2_data/office/val"

basic_transform = [
    # additional data argument
    trns.RandomResizedCrop(224, scale=(0.08, 1.0), ratio=(3.0 / 4.0, 4.0 / 3.0)),
    trns.RandomHorizontalFlip(),  # 水平翻轉
    trns.ColorJitter(brightness=0.5, contrast=0.5, hue=0.5),
]

transform_train = trns.Compose(
    [
        trns.Resize((232, 232), interpolation=trns.InterpolationMode.BICUBIC),
        trns.CenterCrop((224, 224)),
        trns.RandomChoice(basic_transform),
        trns.ToTensor(),
        trns.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ]
)

transform_val = trns.Compose(
    [
        trns.Resize((232, 232), interpolation=trns.InterpolationMode.BICUBIC),
        trns.CenterCrop((224, 224)),
        trns.ToTensor(),
        trns.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ]
)

train_dataset = [
    imgfile for imgfile in os.listdir(img_dir_train) if imgfile.endswith(".jpg")
]

val_dataset = [
    imgfile for imgfile in os.listdir(img_dir_val) if imgfile.endswith(".jpg")
]


train_data = ImageDataset(img_dir_train, train_dataset, transform=transform_train)

val_data = ImageDataset(img_dir_val, val_dataset, transform=transform_val)

train_loader = torch.utils.data.DataLoader(
    dataset=train_data, batch_size=p_batch_size, shuffle=True
)

val_loader = torch.utils.data.DataLoader(
    dataset=val_data, batch_size=p_batch_size, shuffle=True
)

model = resnet50()

model.classifier = nn.Sequential(
    nn.Flatten(), nn.Linear(2048, 500), nn.ReLU(), nn.Linear(500, 65)
)

# for i, (name, param) in enumerate(model.named_parameters()):
#     param.requires_grad = False
#     if "classifier" in name or "fc" in name:
#         param.requires_grad = True


for name, param in model.named_parameters():
    print("name: ", name)
    print("requires_grad: ", param.requires_grad)

model = model.to(device)
summary(model, (3, 232, 232))

parameters = filter(lambda p: p.requires_grad, model.parameters())

optimizer = torch.optim.Adam(parameters, lr=p_lr)
loss_func = nn.CrossEntropyLoss()

# 訓練模型
training_loss, training_accuracy, validation_loss, validation_accuracy = fit_model(
    model,
    loss_func,
    optimizer,
    num_epochs,
    train_loader,
    val_loader,
)
# Loss
plt.plot(range(num_epochs), training_loss, "b-", label="Training_loss")
plt.plot(range(num_epochs), validation_loss, "g-", label="validation_loss")
plt.title("Training & Validation loss")
plt.xlabel("Number of epochs")
plt.ylabel("Loss")
# 添加其他文本
plt.legend()
plt.savefig("loss_cb.png")
plt.show()

# Accuracy
plt.plot(range(num_epochs), training_accuracy, "b-", label="Training_accuracy")
plt.plot(range(num_epochs), validation_accuracy, "g-", label="Validation_accuracy")
plt.title("Training & Validation accuracy")
plt.xlabel("Number of epochs")
plt.ylabel("Accuracy")
plt.legend()
plt.savefig("acc_cb.png")
plt.show()
