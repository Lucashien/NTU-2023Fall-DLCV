import torch.nn as nn
from torchvision import models
from torchvision.models import VGG16_Weights
import torch.nn.functional as F
from torchsummary import summary


class FCN32VGG16(nn.Module):
    def __init__(self, num_classes):
        super().__init__()
        self.features = models.vgg16(weights=VGG16_Weights.DEFAULT).features

        self.classifier = nn.Sequential(
            nn.Conv2d(512, 4096, 7),
            nn.ReLU(),
            nn.Conv2d(4096, 4096, 1),
            nn.ReLU(),
        )
        self.upsample = nn.ConvTranspose2d(
            in_channels=4096,
            out_channels=num_classes,
            kernel_size=44,
            stride=52,
            padding=0,
        )

    def forward(self, x):
        features = self.features(x)
        # print(features.shape)
        classifier = self.classifier(features)
        upsample = self.upsample(classifier)
        # print(upsample.shape)
        return upsample
