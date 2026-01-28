import torch
import torch.nn as nn
from torchvision import models


class HeadPoseNet(nn.Module):
    """
    CNN for head-pose regression (yaw, pitch) in DEGREES
    """
    def __init__(self):
        super().__init__()

        backbone = models.resnet18(weights=None)
        backbone.fc = nn.Identity()
        self.backbone = backbone

        self.fc1 = nn.Linear(512, 128)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(128, 2)  # [yaw, pitch]

    def forward(self, x):
        f = self.backbone(x)
        f = self.relu(self.fc1(f))
        out = self.fc2(f)
        return out[:, 0], out[:, 1]
