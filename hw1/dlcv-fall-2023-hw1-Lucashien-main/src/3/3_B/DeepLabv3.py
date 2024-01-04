from torchvision.models.segmentation import (
    deeplabv3_resnet50,
    deeplabv3_resnet101,
    DeepLabV3_ResNet50_Weights,
    DeepLabV3_ResNet101_Weights,
)
from torchvision.models import ResNet50_Weights, ResNet101_Weights
import torch.nn as nn


class Deeplabv3_Resnet50_Model(nn.Module):
    def __init__(self):
        super(Deeplabv3_Resnet50_Model, self).__init__()
        self.model = deeplabv3_resnet50(
            weights=DeepLabV3_ResNet50_Weights.DEFAULT,
            weights_backbone=ResNet50_Weights.DEFAULT,
        )

        self.model.classifier[4] = nn.Sequential(
            nn.Conv2d(256, 7, 1, 1),
        )
        print(self.model)

    def forward(self, x):
        output = self.model(x)
        return output["out"]
