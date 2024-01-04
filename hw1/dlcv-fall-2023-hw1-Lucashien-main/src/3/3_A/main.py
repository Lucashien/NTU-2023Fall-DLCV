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
from FCN32_model import FCN32VGG16
from mean_iou_evaluate import read_masks, mean_iou_score
from combine_seg.viz_mask import viz_data


# parameter setting

p_lr = 0.0005
p_batch_size = 16
num_epochs = 100
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
img_dir_train = "DLCV_hw1/hw1_data/p3_data/train"
img_dir_val = "DLCV_hw1/hw1_data/p3_data/validation"

###

transform_train = trns.Compose(
    [
        trns.Resize((512, 512), interpolation=trns.InterpolationMode.BICUBIC),
        trns.CenterCrop((512, 512)),
        trns.ColorJitter(brightness=0.5, contrast=0.5, hue=0.5),
        trns.ToTensor(),
        trns.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ]
)

transform_val = trns.Compose(
    [
        trns.Resize((512, 512), interpolation=trns.InterpolationMode.BICUBIC),
        trns.CenterCrop((512, 512)),
        trns.ToTensor(),
        trns.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ]
)

train_dataset = [
    imgfile for imgfile in os.listdir(img_dir_train) if imgfile.endswith(".jpg")
]
train_dataset.sort()


val_dataset = [
    imgfile for imgfile in os.listdir(img_dir_val) if imgfile.endswith(".jpg")
]
val_dataset.sort()


train_data = ImageDataset(
    img_dir_train,
    train_dataset,
    read_masks(img_dir_train),
    img_transform=transform_train,
)

val_data = ImageDataset(
    img_dir_val, val_dataset, read_masks(img_dir_val), img_transform=transform_val
)

train_loader = torch.utils.data.DataLoader(
    dataset=train_data, batch_size=p_batch_size, shuffle=True
)

val_loader = torch.utils.data.DataLoader(
    dataset=val_data, batch_size=p_batch_size, shuffle=True
)

model = FCN32VGG16(7)

model = model.to(device)
summary(model, (3, 512, 512))

parameters = filter(lambda p: p.requires_grad, model.parameters())

optimizer = torch.optim.Adam(parameters, lr=p_lr)
loss_func = nn.CrossEntropyLoss()

# 訓練模型
training_loss, training_iou, validation_loss, validation_iou = fit_model(
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
plt.savefig("loss_3.png")
plt.show()

# Accuracy
plt.plot(range(num_epochs), training_iou, "b-", label="Training_iou")
plt.plot(range(num_epochs), validation_iou, "g-", label="Validation_iou")
plt.title("Training & Validation accuracy")
plt.xlabel("Number of epochs")
plt.ylabel("Accuracy")
plt.legend()
plt.savefig("acc_3.png")
plt.show()
