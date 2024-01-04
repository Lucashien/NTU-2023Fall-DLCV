import torch
from torch.optim import lr_scheduler
from torch import optim

# 訓練 function
# model = CNN model
# loss_func = nn.CrossEntropyLoss()
# input_shape = 32*32
# num_epochs
# train_loader = imgs, labels
# val_loader = imgs, labels


def fit_model(model, loss_func, optimizer, num_epochs, train_loader, val_loader):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("The model will be running on", device, "device")
    scheduler = optim.lr_scheduler.ExponentialLR(optimizer, gamma=0.5)

    # Traning the Model
    # 儲存訓練資訊的 List
    training_loss, training_accuracy = [], []
    test_loss, test_accuracy = [], []
    best_acc = 0
    for epoch in range(num_epochs):
        # ---------------------------
        # Training Stage
        # ---------------------------
        correct_train, total_train = 0, 0

        # enumerate 會吐回 idx 跟 (img, label) from __getitem__
        for i, (images, labels) in enumerate(train_loader):
            # get the inputs(轉 device 的型態)
            images = images.to(torch.float32).to(device)
            labels = labels.to(torch.int64).to(device)
            # 反向傳播演算法
            optimizer.zero_grad()  # 清空 gradients
            outputs = model(images)  # 將imgs輸入至模型進行訓練 (Forward propagation)
            train_loss = loss_func(outputs, labels)  # 計算 loss
            train_loss.backward()  # 將 loss 反向傳播
            optimizer.step()  # 根據計算出的gradients更新參數

            # 計算訓練資料的準確度 (correct_train / total_train)
            # torch.max(a,1) -> value = 每個row最大值 indices = 最大值的idx
            predicted = torch.max(outputs, 1)[1]  # 取出output(預測)的maximum idx
            total_train += len(labels)  # 全部的 label 數 (Total number of labels)
            correct_train += (predicted == labels).float().sum()

        train_accuracy = 100 * correct_train / float(total_train)
        training_accuracy.append(train_accuracy.cpu())
        training_loss.append(train_loss.data.cpu())  # training loss (To cpu())

        # ---------------------------
        # Valid Stage
        # ---------------------------
        correct_test, total_test = 0, 0
        for i, (images, labels) in enumerate(val_loader):
            test_images = images.to(torch.float32).to(device)
            test_labels = labels.to(torch.int64).to(device)
            outputs = model(test_images)  # 將imgs輸入至模型進行訓練 (Forward propagation)
            val_loss = loss_func(outputs, test_labels)  # 計算 loss

            predicted = torch.max(outputs, 1)[1]  # 取出output(預測)的maximum idx
            total_test += len(test_labels)  # 全部的 label 數 (Total number of labels)
            correct_test += (predicted == test_labels).float().sum()

        val_accuracy = (100 * correct_test / float(total_test)).cpu()
        test_accuracy.append(val_accuracy)
        test_loss.append(val_loss.data.cpu())

        if val_accuracy > best_acc:
            torch.save(
                model.state_dict(), f"DLCV_hw1/1/1_A/1_A_acc_{val_accuracy:.3f}.pt"
            )
            best_acc = val_accuracy

        if epoch == 1 or epoch == num_epochs / 2 or epoch == num_epochs:
            torch.save(model.state_dict(), f"DLCV_hw1/1/1_A/EPOCH{epoch}_1_A.pt")

        print(
            "Train Epoch: {}/{} Traing_Loss: {:.3f} Traing_acc: {:.3f}% ,Test_Loss: {:.3f},Test_acc:{:.3f}%".format(
                epoch + 1,
                num_epochs,
                train_loss.data,
                train_accuracy,
                val_loss.data,
                val_accuracy,
            )
        )
        scheduler.step()

    return (training_loss, training_accuracy, test_loss, test_accuracy)
