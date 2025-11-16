import torch
import torch.nn as nn
from torchvision import models
import torch.nn.functional as F

class FreshMultiTaskResNet50(nn.Module):
    def __init__(self):
        super().__init__()
        base = models.resnet50(weights=None)
        base.conv1 = nn.Conv2d(2, 64, kernel_size=7, stride=2, padding=3, bias=False)
        self.conv1 = base.conv1
        self.bn1 = base.bn1
        self.relu = base.relu
        self.maxpool = base.maxpool
        self.layer1 = base.layer1
        self.layer2 = base.layer2
        self.layer3 = base.layer3
        self.layer4 = base.layer4
        self.avgpool = base.avgpool

        self.head = nn.Sequential(nn.Dropout(0.4), nn.Linear(2048,4))
        self.height_head = nn.Sequential(
            nn.ConvTranspose2d(2048,512,4,2,1), nn.BatchNorm2d(512), nn.ReLU(True),
            nn.ConvTranspose2d(512,256,4,2,1), nn.BatchNorm2d(256), nn.ReLU(True),
            nn.ConvTranspose2d(256,128,4,2,1), nn.BatchNorm2d(128), nn.ReLU(True),
            nn.ConvTranspose2d(128,64,4,2,1), nn.BatchNorm2d(64), nn.ReLU(True),
            nn.ConvTranspose2d(64,1,4,2,1)
        )

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        spatial = self.layer4(x)
        pooled = self.avgpool(spatial)
        pooled = torch.flatten(pooled,1)
        scores = self.head(pooled)
        height = self.height_head(spatial)
        return scores, height
