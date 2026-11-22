"""Model definition."""
import torch.nn as nn
from torchvision import models


class SimpleCNN(nn.Module):
    """ResNet18 backbone with its final layer replaced for our class count."""

    def __init__(self, num_classes: int):
        super().__init__()
        self.model = models.resnet18(weights=models.ResNet18_Weights.IMAGENET1K_V1)
        self.model.fc = nn.Linear(self.model.fc.in_features, num_classes)

    def forward(self, x):
        return self.model(x)
