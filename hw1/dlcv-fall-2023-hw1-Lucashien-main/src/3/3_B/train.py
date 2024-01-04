import torch
from torch.optim import lr_scheduler
from torch import optim
import time
import os
from mean_iou_evaluate import mean_iou_score
import numpy as np
import torch.nn as nn


class FocalLoss(nn.Module):
    def __init__(self, alpha=0.25, gamma=2.0):
        super(FocalLoss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.loss = nn.CrossEntropyLoss(ignore_index=6)

    def forward(self, inputs, targets):
        ce_loss = self.loss(inputs, targets)
        exp_loss = torch.exp(-ce_loss)
        loss = self.alpha * (1 - exp_loss) ** self.gamma * ce_loss
        return loss


def fit_model(
    model,
    optimizer,
    num_epochs,
    train_loader,
    val_loader,
):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    Focalloss = FocalLoss()
    print("The model will be running on", device, "device")

    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=num_epochs * len(train_loader), eta_min=1e-6
    )
    best_acc = 0

    for epoch in range(num_epochs):
        start_time = time.time()
        # ---------------------------
        # Training Stage
        # ---------------------------
        model.train()
        train_loss = []

        for i, (imgs, masks) in enumerate(train_loader):
            print(f"Batch {i+1}/{len(train_loader)} processed", end="\r")
            time.sleep(0.00000001)
            imgs = imgs.float().to(device)
            masks_t = torch.Tensor(masks).long().to(device)
            output = model(imgs)
            loss = Focalloss(output, masks_t)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            train_loss.append(loss.item())
            # del logits, labels

        train_loss = sum(train_loss) / len(train_loss)
        scheduler.step()

        # ---------------------------
        # Valid Stage
        # ---------------------------
        model.eval()

        val_loss, val_pred_list, val_mask_list = [], [], []

        for i, (imgs, masks) in enumerate(val_loader):
            print(f"Batch (val){i+1}/{len(val_loader)} processed", end="\r")
            imgs = imgs.float().to(device)
            masks_t = torch.Tensor(masks).long().to(device)
            with torch.no_grad():
                output = model(imgs)
            loss = Focalloss(output, torch.Tensor(masks_t).long().to(device))
            val_pred_list.append(output.cpu().argmax(dim=1))
            val_mask_list.append(masks)
            val_loss.append(loss.item())

        val_loss = sum(val_loss) / len(val_loss)

        # 把矩陣垂直拼起來
        val_pred_list = np.concatenate(val_pred_list, axis=0)
        val_mask_list = np.concatenate(val_mask_list, axis=0)
        val_iou = mean_iou_score(val_pred_list, val_mask_list)
        end_time = time.time()

        if val_iou > best_acc:
            torch.save(
                model.state_dict(),
                f"DLCV_hw1/3/3_B_101/Best_3B_model_{val_iou*100:.3f}.pt",
            )
            best_acc = val_iou

        if epoch == 1 or epoch == num_epochs / 2 or epoch == num_epochs:
            torch.save(
                model.state_dict(),
                f"DLCV_hw1/3/3_B_101/3B_model_EPOCH_{epoch}.pt",
            )

        print(
            "Train Epoch: {}/{} Traing_Loss: {:.3f}, Test_Loss: {:.3f},Test_iou:{:.3f}%, Spend_time: {:.3f}".format(
                epoch + 1,
                num_epochs,
                train_loss,
                val_loss,
                val_iou * 100,
                end_time - start_time,
            )
        )

    return "OK"
