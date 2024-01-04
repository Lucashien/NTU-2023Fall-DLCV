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
from PretrainSSL import fit_model
import time
from torchvision.models import resnet50


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


if __name__ == "__main__":
    SSL_num_epochs = 10
    num_epochs = 600
    best_acc = 0
    same_seeds(607)
    batch_size = 256
    SSL = True
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = resnet50(weights=None).to(device)

    if SSL:
        img_dir_train = "DLCV_hw1/hw1_data/p2_data/mini/train"
        train_set = ImageDataset(img_dir_train, transform=basic_trans)
        train_loader = DataLoader(
            train_set,
            batch_size=batch_size,
            shuffle=True,
            num_workers=2,
            pin_memory=True,
        )

        learner = BYOL(model, image_size=128, hidden_layer="avgpool")
        optimizer = torch.optim.Adam(learner.parameters(), lr=0.01)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
            optimizer, SSL_num_epochs, T_mult=2
        )

        fit_model(learner, model, optimizer, SSL_num_epochs, train_loader)

    # %% office dataset is below
    print(f"\n--Fine Tuning Part--\n")

    img_dir_train = "DLCV_hw1/hw1_data/p2_data/office/train"
    img_dir_val = "DLCV_hw1/hw1_data/p2_data/office/val"

    train_set = ImageDataset(img_dir_train, transform=basic_trans)
    valid_set = ImageDataset(img_dir_val, transform=None)

    train_loader = DataLoader(
        train_set, batch_size=batch_size, shuffle=True, num_workers=4
    )
    valid_loader = DataLoader(valid_set, batch_size=256, shuffle=True, num_workers=4)

    Resnet50 = resnet50(weights=None).to(device)
    Pretrain_path = "DLCV_hw1/2/C/PretrainBYOL.pt"
    Resnet50.load_state_dict(torch.load(Pretrain_path))
    Resnet50.fc = nn.Sequential(
        nn.Linear(2048, 65),
    )
    Resnet50 = Resnet50.to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(Resnet50.parameters(), lr=0.001)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=num_epochs * len(train_loader), eta_min=1e-6
    )

    for epoch in range(num_epochs):
        # ---------- Training ----------
        # Make sure the model is in train mode before training.
        Resnet50.train()
        train_loss = []
        train_accs = []

        for i, (imgs, labels) in enumerate(train_loader):
            print(f"Batch {i+1}/{len(train_loader)} processed", end="\r")
            time.sleep(0.00000001)
            imgs = imgs.float().to(device)
            output = Resnet50(imgs)
            loss = criterion(output, labels.to(device))
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            acc = (output.argmax(dim=-1) == labels.to(device)).float()
            acc = acc.mean()
            train_loss.append(loss.item())
            train_accs.append(acc)

        train_loss = sum(train_loss) / len(train_loss)
        train_acc = sum(train_accs) / len(train_accs)
        scheduler.step()

        # ---------- Validation ----------
        Resnet50.eval()
        valid_loss = []
        valid_accs = []

        for i, (imgs, labels) in enumerate(valid_loader):
            print(f"Batch (valid){i+1}/{len(train_loader)} processed", end="\r")
            time.sleep(0.00000001)
            imgs = imgs.float().to(device)
            with torch.no_grad():
                output = Resnet50(imgs)

            loss = criterion(output, labels.to(device))
            acc = (output.argmax(dim=-1) == labels.to(device)).float()
            acc = acc.mean()
            valid_loss.append(loss.item())
            valid_accs.append(acc)

        valid_loss = sum(valid_loss) / len(valid_loss)
        valid_acc = sum(valid_accs) / len(valid_accs)

        print(
            "Train Epoch: {}/{} Traing_Loss: {:.3f} Traing_acc: {:.3f}% ,Test_Loss: {:.3f},Test_acc:{:.3f}%".format(
                epoch + 1,
                num_epochs,
                train_loss,
                train_acc * 100,
                valid_loss,
                valid_acc * 100,
            )
        )

        # save models
        if valid_acc > best_acc:
            torch.save(
                Resnet50.state_dict(), f"DLCV_hw1/2/C/2C_best_acc{valid_acc:.3f}.pt"
            )
            best_acc = valid_acc
