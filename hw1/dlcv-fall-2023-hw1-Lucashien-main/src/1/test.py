import os
import torch
import torchvision.io as tvio
import torchvision.transforms.functional as fn
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as transforms
from torchvision.transforms import ToPILImage
import matplotlib.pyplot as plt
import time
from PIL import Image
from sklearn.manifold import TSNE
import torch.nn as nn
from torchvision.models import convnext_base, ConvNeXt_Base_Weights
import torch.nn.functional as F


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


class ImageDataset(Dataset):
    ## 初始化
    def __init__(self, img_dir, dataset_list, transform=None, target_transform=None):
        self.dataset_list = dataset_list
        self.img_dir = img_dir
        self.transform = transform
        self.target_transform = target_transform

    ### dataset 長度
    def __len__(self):
        self.datasetlength = len(self.dataset_list)
        return self.datasetlength

    def __getitem__(self, idx):
        img_name = self.dataset_list[idx]
        img_path = os.path.join(self.img_dir, img_name)
        img = tvio.read_image(img_path)
        img = F.interpolate(img, 232)
        label = img_name.split("_")[0]
        label = torch.tensor(int(label), dtype=torch.float32)
        return img, label


import numpy as np
import torch
import torch.nn as nn
from torchsummary import summary
import os
from torch.optim import Adam, SGD
import matplotlib.pyplot as plt
from torchvision.models import convnext_base, ConvNeXt_Base_Weights
import torchvision.transforms as trns
from sklearn.decomposition import PCA


# parameter setting
p_lr = 0.0005
p_batch_size = 32
num_epochs = 1
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
img_dir_train = "DLCV_hw1/hw1_data/p1_data/train_50"
img_dir_val = "DLCV_hw1/hw1_data/p1_data/val_50"

transform_val = trns.Compose(
    [
        trns.Resize((232, 232), interpolation=trns.InterpolationMode.BICUBIC),
        trns.CenterCrop((224, 224)),
        trns.ToTensor(),
        trns.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ]
)
val_dataset = [
    imgfile for imgfile in os.listdir(img_dir_val) if imgfile.endswith(".png")
]

val_data = ImageDataset(img_dir_val, val_dataset)
val_loader = torch.utils.data.DataLoader(dataset=val_data, batch_size=32, shuffle=True)

model = convnext_base()
model.classifier = nn.Sequential(
    nn.Flatten(), nn.Linear(1024, 500), nn.ReLU(), nn.Linear(500, 50)
)
# model = CNN().to(device)
model.load_state_dict(torch.load("DLCV_hw1/1/Best_1_B_acc_90.pt"))
model = model.to(device)
# print(model)

hook_list = []


def hook_fn(module, input, output):
    hook_list.append(output)
    # print("hi")


print(model)
target_layer = model.features[7][2].block[0]
hook = target_layer.register_forward_hook(hook_fn)

# %% validation
loss_func = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=p_lr)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("The model will be running on", device, "device")
scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=0.9)
test_loss, test_accuracy = [], []
color_label = 0
for epoch in range(num_epochs):
    # ---------------------------
    # Valid Stage
    # ---------------------------
    correct_test, total_test = 0, 0
    for i, (images, labels) in enumerate(val_loader):
        print(f"Batch {i+1}/{len(val_loader)} processed", end="\r")
        time.sleep(0.00000001)
        test_images = images.to(torch.float32).to(device)
        test_labels = labels.to(torch.int64).to(device)
        with torch.no_grad():
            outputs = model(test_images)
        val_loss = loss_func(outputs, test_labels)  # 計算 loss

        predicted = torch.max(outputs, 1)[1]  # 取出output(預測)的maximum idx

        torch.cuda.empty_cache()

    print(
        "Train Epoch: {}/{}".format(
            epoch + 1,
            num_epochs,
        )
    )
hook.remove()
# %%-------------------------------------------------------------------------------
# get last second layer weight
flattened_outputs = torch.cat(hook_list, dim=0)
print(flattened_outputs.shape)
flattened_outputs = flattened_outputs.reshape(-1, 1024 * 1 * 7)
numpy_outputs = flattened_outputs.cpu().detach().numpy()

pca = PCA(n_components=2)  # 你可以根据需要选择要保留的主成分数量
pca.fit(numpy_outputs)  # 在数据上拟合 PCA 模型
transformed_outputs = pca.transform(numpy_outputs)  # 对数据进行降维


labels = []
for images, batch_labels in val_loader:
    labels.extend(batch_labels.tolist())  # 将标签添加到 labels 列表中

plt.scatter(transformed_outputs[:, 0], transformed_outputs[:, 1], c=labels)
plt.xlabel("Principal Component 1")
plt.ylabel("Principal Component 2")
plt.title("PCA of Layer Outputs")

plt.savefig("DLCV_hw1/1/conv_pca_result.png")


tsne = TSNE(n_components=2, perplexity=30, n_iter=300)
tsne_results = tsne.fit_transform(flattened_outputs.cpu())

# t-SNE
plt.figure(figsize=(8, 6))
plt.scatter(tsne_results[:, 0], tsne_results[:, 1], c=labels, cmap="viridis")
plt.title(f"t-SNE Visualization (EPOCH 80)")
plt.colorbar()
plt.savefig(f"DLCV_hw1/1/tsne_conv.png")
plt.show()
