import torch
from torch.optim import lr_scheduler
from torch import optim
import time
import os
from mean_iou_evaluate import mean_iou_score
import numpy as np


def fit_model(
    model,
    loss_func,
    optimizer,
    num_epochs,
    train_loader,
    val_loader,
):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("The model will be running on", device, "device")
    scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=0.9)
    training_loss, training_iou_list = [], []
    test_loss, test_iou_list = [], []
    best_acc = 0
    for epoch in range(num_epochs):
        start_time = time.time()
        # ---------------------------
        # Training Stage
        # ---------------------------
        train_prediction_list, train_masks_list = [], []
        train_iou, val_iou = 0, 0
        for i, (images, masks) in enumerate(train_loader):
            print(f"Batch {i+1}/{len(train_loader)} processed", end="\r")
            time.sleep(0.00000001)

            images = images.to(torch.float32).to(device)
            masks = masks.to(torch.float32).to(device)

            # 反向傳播演算法
            optimizer.zero_grad()  # 清空 gradients
            outputs = model(images)  # 將imgs輸入至模型進行訓練 (Forward propagation)
            train_loss = loss_func(outputs, masks.long())  # 計算 loss
            train_loss.backward()  # 將 loss 反向傳播
            optimizer.step()  # 根據計算出的gradients更新參數

            # 計算訓練資料的準確度 (correct_train / total_train)
            prediction = torch.max(outputs, 1)[1]
            train_prediction_list.extend(prediction.cpu().numpy())
            train_masks_list.extend(masks.cpu().numpy())

        train_iou = mean_iou_score(
            np.array(train_prediction_list), np.array(train_masks_list)
        )
        training_loss.append(train_loss.data.cpu())
        training_iou_list.append(train_iou)

        torch.cuda.empty_cache()
        # ---------------------------
        # Valid Stage
        # ---------------------------
        val_prediction_list, val_masks_list = [], []
        for i, (images, masks) in enumerate(val_loader):
            print(f"Batch (val){i+1}/{len(val_loader)} processed", end="\r")
            test_images = images.to(torch.float32).to(device)
            test_masks = masks.to(torch.int64).to(device)
            outputs = model(test_images)  # 將imgs輸入至模型進行訓練 (Forward propagation)
            val_loss = loss_func(outputs, test_masks)  # 計算 loss

            prediction = torch.max(outputs, 1)[1]  # 取出output(預測)的maximum idx
            val_prediction_list.extend(prediction.cpu().numpy().flatten())
            val_masks_list.extend(masks.cpu().numpy().flatten())

        val_iou = mean_iou_score(
            np.array(val_prediction_list), np.array(val_masks_list)
        )
        test_loss.append(val_loss.data.cpu())
        test_iou_list.append(val_iou)

        end_time = time.time()
        spend_time = end_time - start_time

        if val_iou > best_acc:
            torch.save(
                model.state_dict(),
                f"DLCV_hw1/3/3_A/Best_3A_model_{val_iou*100:.3f}.pt",
            )
            best_acc = val_iou

        if epoch == 1 or epoch == num_epochs / 2 or epoch == num_epochs:
            torch.save(
                model.state_dict(),
                f"DLCV_hw1/3/3_A/3A_model_EPOCH_{epoch}.pt",
            )

        print(
            "Train Epoch: {}/{} Traing_Loss: {:.3f} Traing_iou {:.3f}% ,Test_Loss: {:.3f},Test_iou:{:.3f}%, Spend_time: {:.3f}".format(
                epoch + 1,
                num_epochs,
                train_loss.data,
                train_iou * 100,
                val_loss.data,
                val_iou * 100,
                spend_time,
            )
        )
        scheduler.step()

    return (training_loss, training_iou_list, test_loss, test_iou_list)
