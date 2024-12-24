# Import Library
import torch
import torch.nn as nn
from torchvision import models
from torchvision.models import mobilenet_v3_large

class CBAM(nn.Module):
    def __init__(self, channels, reduction=16):
        super(CBAM, self).__init__()

        # Channel Attention Module
        self.channel_attention = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(channels, channels // reduction, kernel_size=1),
            nn.ReLU(),
            nn.Conv2d(channels // reduction, channels, kernel_size=1),
            nn.Sigmoid()
        )
        
        # Spatial Attention Module
        self.spatial_attention = nn.Sequential(
            nn.Conv2d(2, 1, kernel_size=7, padding=3),
            nn.Sigmoid()
        )
        # Sigmoid untuk menekankan pada binary klasifikasi

    def forward(self, x):
        # Channel Attention
        ca = self.channel_attention(x)
        x = x * ca

        # Spatial Attention
        avg_out = torch.mean(x, dim=1, keepdim=True)
        max_out, _ = torch.max(x, dim=1, keepdim=True)
        sa_input = torch.cat([avg_out, max_out], dim=1)
        sa = self.spatial_attention(sa_input)
        x = x * sa

        return x

# Define the modified MobileNetV3Model with residual connection and CBAM
class MobileNetV3TidakBerdiri(nn.Module):
    def __init__(self, pretrained=True):
        super(MobileNetV3TidakBerdiri, self).__init__()

        # MobileNetV3 Backbone
        self.mobilenet = models.mobilenet_v3_large(pretrained=pretrained)
        num_ftrs_mobilenet = self.mobilenet.classifier[0].in_features
        self.mobilenet.classifier = nn.Identity()  # Remove the classifier

        # CBAM Layer
        self.cbam = CBAM(channels=num_ftrs_mobilenet)

        # Residual connection
        self.residual_conv = nn.Conv2d(num_ftrs_mobilenet, num_ftrs_mobilenet, kernel_size=1)

        # Classifier
        self.classifier = nn.Sequential(
            nn.Linear(num_ftrs_mobilenet, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(512, 2)  # Output layer for binary classification
        )

    def forward(self, x):
        # Extract features from MobileNetV3 backbone
        features = self.mobilenet.features(x)

        # Apply CBAM
        attention_features = self.cbam(features)

        # Residual connection
        if features.size() == attention_features.size():
            features = features + attention_features
        else:
            features = self.residual_conv(features) + attention_features

        # Global Average Pooling
        pooled_features = nn.AdaptiveAvgPool2d(1)(features)
        pooled_features = torch.flatten(pooled_features, 1)

        # Classifier
        output = self.classifier(pooled_features)
        return output
