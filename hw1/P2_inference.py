import os
import sys
import csv
import torch
import numpy as np
import torch.nn as nn
import torchvision.io as tvio
import matplotlib.pyplot as plt
import torchvision.transforms as trns
import torchvision.transforms.functional as fn
from PIL import Image
from torch.optim import Adam, SGD
from torchvision.models import resnet50
from torchvision.transforms import ToPILImage
from torch.utils.data import Dataset, DataLoader


class ImageDataset(Dataset):
    def __init__(self, file_path, transform=None):
        self.path = file_path
        self.transform = transform
        if transform:
            self.transform = transform
        else:
            self.transform = trns.Compose(
                [
                    trns.Resize([128, 128]),
                    trns.ToTensor(),
                    trns.Normalize(
                        mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
                    ),
                ]
            )

        self.files = sorted([x for x in os.listdir(self.path) if x.endswith(".jpg")])
        self.data = []
        for file in self.files:
            self.data.append(Image.open(os.path.join(self.path, file)).copy())

    def __getitem__(self, idx):
        data = Image.open(os.path.join(self.path, self.files[idx]))
        data = self.transform(data)
        imgname = self.files[idx]

        return data, imgname

    def __len__(self):
        return len(self.files)


# parameter setting
print("Running P2_inference.py...")
test_csv = sys.argv[1]
img_dir_val = sys.argv[2]
test_pred_csv = sys.argv[3]

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

val_data = ImageDataset(img_dir_val)
val_loader = torch.utils.data.DataLoader(dataset=val_data, batch_size=32, shuffle=False)

# Define Model
model = resnet50(weights=None).to(device)
model.fc = nn.Sequential(
    nn.Linear(2048, 65),
)
model.load_state_dict(torch.load("inference_model/Model_2C.pt"))
model = model.to(device)
model.eval()

# ---------------------------
# Valid Stage
# ---------------------------
predict_list = []
img_name_list = []
correct_test, total_test = 0, 0

with torch.no_grad():
    for i, (images, img_name) in enumerate(val_loader):
        test_images = images.to(torch.float32).to(device)
        outputs = model(test_images)
        predict = torch.max(outputs, 1)[1]
        predict_list.extend(predict.flatten().detach().tolist())
        img_name_list.extend(img_name)

np_img_name = np.array(img_name_list, dtype=str)
np_predict = np.array(predict_list, dtype=np.uint8)

test_img_name = []
with open(test_csv, "r", newline="") as file:
    reader = csv.reader(file)
    # 跳過首列（如果有列標題的話）
    next(reader)
    # 遍歷每一行
    for row in reader:
        if len(row) > 1:  # 確保行中有足夠的資料
            test_img_name.append(row[1])

with open(test_pred_csv, "w", newline="") as file:
    writer = csv.writer(file)
    writer.writerow(("id", "filename", "label"))
    id = 0
    for img_name, predict in zip(np_img_name, np_predict):
        if img_name in test_img_name:
            writer.writerow([id, img_name, predict])
            id += 1

print("Program all done.")
