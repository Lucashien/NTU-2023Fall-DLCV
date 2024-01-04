import os
import csv
import sys
import torch
import numpy as np
import torch.nn as nn
import torchvision.io as tvio
import matplotlib.pyplot as plt
import torchvision.transforms as trns
import torchvision.transforms.functional as fn
from torch.optim import Adam, SGD
from torchvision.models import convnext_base
from torchvision.transforms import ToPILImage
from torch.utils.data import Dataset, DataLoader


class ImageDataset(Dataset):
    ## 初始化
    def __init__(self, img_dir):
        self.dataset_list = [
            imgfile for imgfile in os.listdir(img_dir_val) if imgfile.endswith(".png")
        ]
        self.dataset_list.sort()
        self.img_dir = img_dir
        self.transform = trns.Compose(
            [
                trns.Resize((232, 232), interpolation=trns.InterpolationMode.BICUBIC),
                trns.CenterCrop((224, 224)),
                trns.ToTensor(),
                trns.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            ]
        )

    ### dataset 長度
    def __len__(self):
        self.datasetlength = len(self.dataset_list)
        return self.datasetlength

    def __getitem__(self, idx):
        img_name = self.dataset_list[idx]
        self.img_name = img_name
        img_path = os.path.join(self.img_dir, img_name)
        img = tvio.read_image(img_path)

        if self.transform:
            to_pil = ToPILImage()
            img_pil = to_pil(img)
            img = self.transform(img_pil)

        return img, img_name


# parameter setting
print("Running P1_inference.py...")
img_dir_val = sys.argv[1]
out_csv = sys.argv[2]
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

val_data = ImageDataset(img_dir_val)
val_loader = torch.utils.data.DataLoader(dataset=val_data, batch_size=32, shuffle=False)

model = convnext_base()
model.classifier = nn.Sequential(
    nn.Flatten(), nn.Linear(1024, 500), nn.ReLU(), nn.Linear(500, 50)
)
model.load_state_dict(torch.load("inference_model/Model_1.pt"))
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

try:
    os.makedirs(os.path.dirname(out_csv), exist_ok=True)
except:
    pass

with open(out_csv, "w", newline="") as file:
    writer = csv.writer(file)
    writer.writerow(("filename", "label"))
    for data in zip(np_img_name, np_predict):
        writer.writerow(data)

print("Program all done.")
