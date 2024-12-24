import torch
from torchvision import models
import torch.nn as nn

class MobileNetV3Berdiri(nn.Module):
    def __init__(self, pretrained=True):
        super(MobileNetV3Berdiri, self).__init__()

        # MobileNetV3 Backbone
        self.mobilenet = models.mobilenet_v3_large(pretrained=pretrained)
        num_ftrs_mobilenet = self.mobilenet.classifier[0].in_features
        self.mobilenet.classifier = nn.Identity()  # Remove the classifier

        # Classifier
        self.classifier = nn.Sequential(
            nn.Linear(num_ftrs_mobilenet, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(512, 2)  # Output layer for binary classification
        )

    def forward(self, x):
        mobilenet_features = self.mobilenet(x)
        output = self.classifier(mobilenet_features)
        return output