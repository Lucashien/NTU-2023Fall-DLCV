import torch.nn as nn
from torchvision.models import convnext_base, ConvNeXt_Base_Weights


# 定義簡單的CNN模型
class CNN(nn.Module):
    def __init__(self):
        super(CNN, self).__init__()  # 可調用nn.Moudule的函數

        # 第一個卷積層
        self.conv1 = nn.Conv2d(in_channels=3, out_channels=32, kernel_size=3, padding=1)
        self.relu1 = nn.ReLU()
        self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2)

        # 第二個卷積層
        self.conv2 = nn.Conv2d(
            in_channels=32, out_channels=64, kernel_size=3, padding=1
        )
        self.relu2 = nn.ReLU()
        self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2)

        # 全連接層
        self.fc1 = nn.Linear(in_features=29696, out_features=128)
        self.relu3 = nn.ReLU()
        self.fc2 = nn.Linear(in_features=128, out_features=50)  # 50類

        self.drop25 = nn.Dropout2d(0.25)

    def forward(self, x):
        x = self.conv1(x)
        x = self.relu1(x)
        x = self.pool1(x)

        x = self.conv2(x)
        x = self.relu2(x)
        x = self.pool2(x)

        x = x.view(x.size(0), -1)  # 展平特徵圖
        x = self.fc1(x)
        return x
