import torch
from torch.optim import lr_scheduler
from torch import optim
import time
from PIL import Image
import os
from torchvision.transforms import ToPILImage
import matplotlib.pyplot as plt


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
    training_loss, training_accuracy = [], []
    test_loss, test_accuracy = [], []
    best_acc = 0

    for epoch in range(num_epochs):
        start_time = time.time()
        # ---------------------------
        # Training Stage
        # ---------------------------
        correct_train, total_train = 0, 0
        for i, (images, labels) in enumerate(train_loader):
            print(f"Batch {i+1}/{len(train_loader)} processed", end="\r")
            time.sleep(0.00000001)

            images = images.to(torch.float32).to(device)
            labels = labels.to(torch.int64).to(device)

            # 反向傳播演算法
            optimizer.zero_grad()
            outputs = model(images)
            train_loss = loss_func(outputs, labels)
            train_loss.backward()
            optimizer.step()

            predicted = torch.max(outputs, 1)[1]
            total_train += len(labels)
            correct_train += (predicted == labels).float().sum()

        train_accuracy = 100 * correct_train / float(total_train)
        training_accuracy.append(train_accuracy.cpu())
        training_loss.append(train_loss.data.cpu())
        torch.cuda.empty_cache()
        # ---------------------------
        # Valid Stage
        # ---------------------------
        correct_test, total_test = 0, 0
        for i, (images, labels) in enumerate(val_loader):
            test_images = images.to(torch.float32).to(device)
            test_labels = labels.to(torch.int64).to(device)
            outputs = model(test_images)
            val_loss = loss_func(outputs, test_labels)

            predicted = torch.max(outputs, 1)[1]
            total_test += len(test_labels)
            correct_test += (predicted == test_labels).float().sum()

        val_accuracy = (100 * correct_test / float(total_test)).cpu()
        test_accuracy.append(val_accuracy)
        test_loss.append(val_loss.data.cpu())

        end_time = time.time()
        spend_time = end_time - start_time
        print(
            "Train Epoch: {}/{} Traing_Loss: {:.3f} Traing_acc: {:.3f}% ,Test_Loss: {:.3f},Test_acc:{:.3f}%, Spend_time: {:.3f}".format(
                epoch + 1,
                num_epochs,
                train_loss.data,
                train_accuracy,
                val_loss.data,
                val_accuracy,
                spend_time,
            )
        )
        scheduler.step()
        # save models
        if val_accuracy > best_acc:
            torch.save(
                model.state_dict(), f"DLCV_hw1/2/A/2A_best_acc{val_accuracy:.3f}.pt"
            )
            best_acc = val_accuracy

    return (training_loss, training_accuracy, test_loss, test_accuracy)
