import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
import random
import torch.nn as nn
import os
import imageio.v2 as imageio
from tqdm import tqdm
import torchvision
import torchvision.transforms as trns
from PIL import Image
from LoadData import MiniDataset, ImageDataset
from byol_pytorch import BYOL
import time


def same_seeds(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True


basic_trans = trns.Compose(
    [
        trns.Resize([128, 128]),
        trns.RandomHorizontalFlip(),
        trns.TrivialAugmentWide(),
        trns.ToTensor(),
        trns.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ]
)


def fit_model(learner, model, optimizer, SSL_num_epochs, train_loader):
    print(f"\n--Pre-train by BYOL--\n")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    optimizer = torch.optim.Adam(learner.parameters(), lr=0.01)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
        optimizer, SSL_num_epochs, T_mult=2
    )

    for epoch in range(SSL_num_epochs):
        # ---------- Training ----------
        model.train()
        train_loss = []

        for i, (imgs, _) in enumerate(train_loader):
            print(f"Batch {i+1}/{len(train_loader)} processed", end="\r")
            time.sleep(0.00000001)
            imgs = imgs.float()
            loss = learner(imgs.to(device))
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            learner.update_moving_average()
            train_loss.append(loss.item())

        train_loss = sum(train_loss) / len(train_loss)
        scheduler.step()

        print(
            "Train Epoch: {}/{} Traing_Loss: {:.3f}".format(
                epoch + 1, SSL_num_epochs, train_loss
            )
        )

    torch.save(model.state_dict(), "DLCV_hw1/2/B/PretrainBYOL.ckpt")
