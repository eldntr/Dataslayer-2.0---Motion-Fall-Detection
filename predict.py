import os
import torch
import csv
import numpy as np
from glob import glob
from PIL import Image
from tqdm import tqdm
from torchvision import transforms
import cv2
from collections import Counter
import os
import random
from collections import Counter, defaultdict, deque
from glob import glob

import cv2
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import albumentations as A
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split
from torchvision import models, transforms
from albumentations.pytorch import ToTensorV2
from scipy.ndimage import gaussian_filter1d
import torch.optim as optim
from torchvision import models
from torchvision import transforms

import os
from torchvision import datasets, transforms, models
from torch.utils.data import DataLoader, Dataset
from PIL import Image
import torch
import torch.nn as nn
from tqdm import tqdm

import warnings
warnings.filterwarnings("ignore", category=FutureWarning)

# Directories
test_directory = 'test'
BOUNDARY_RATIO = 0.5  # Define boundary ratio
device = 'cpu'

# Load YOLO model
print("Loading YOLO model...")
yolo_model = torch.hub.load('ultralytics/yolov5', 'yolov5s', device=device, force_reload=False)

# Initialize transforms
data_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

# Dataset for testing
class TestingDataset(Dataset):
    def __init__(self, image_dir, transform=None):
        self.image_paths = glob(os.path.join(image_dir, '*'))
        self.transform = transform

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        img_path = self.image_paths[idx]
        img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
        if img is None:
            raise ValueError(f"Image not found or unable to read: {img_path}")

        # Convert grayscale to RGB
        img = np.expand_dims(img, axis=2)
        img = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)
        img = Image.fromarray(img)

        if self.transform:
            img = self.transform(img)

        return img, img_path

# Load testing dataset
testing_dataset = TestingDataset(image_dir=test_directory, transform=data_transform)
testing_loader = DataLoader(testing_dataset, batch_size=1, shuffle=False)

# Load pretrained models
from model_tidak_berdiri import MobileNetV3TidakBerdiri
from model_berdiri import MobileNetV3Berdiri

model_berdiri = MobileNetV3Berdiri(pretrained=False).to(device)
model_tidak_berdiri = MobileNetV3TidakBerdiri(pretrained=False).to(device)

# Load saved weights
model_berdiri.load_state_dict(torch.load("mobilenetv3_berdiri_model.pth", map_location=device))
model_tidak_berdiri.load_state_dict(torch.load("mobilenetv3_tidak_berdiri_model.pth", map_location=device))

# Set models to evaluation mode
model_berdiri.eval()
model_tidak_berdiri.eval()

# Pipeline
print("Starting pipeline...")
results = []

HEAD_REGION_FRACTION = 0.2

for images, img_paths in tqdm(testing_loader, desc="Processing images"):
    images = images.to(device)

    # Run YOLO detection
    img_path = img_paths[0]
    with torch.no_grad():
        yolo_results = yolo_model(img_path)
    detections = yolo_results.xyxy[0].cpu().numpy()

    # Determine if "berdiri"
    img = Image.open(img_path)
    frame_width, frame_height = img.size
    boundary_y = int(BOUNDARY_RATIO * frame_height)
    is_berdiri = any(
        int(cls) == 0
        and (box[1] + HEAD_REGION_FRACTION * (box[3] - box[1])) <= boundary_y  # Top 20% region
        for *box, conf, cls in detections
    )

    # Use the appropriate model
    if is_berdiri:
        with torch.no_grad():
            outputs = model_berdiri(images)
            _, predicted = torch.max(outputs, 1)
            label = 1 if predicted.item() == 1 else 0  # 1 for 'fall', 0 for 'non_fall'
    else:
        with torch.no_grad():
            outputs = model_tidak_berdiri(images)
            _, predicted = torch.max(outputs, 1)
            label = 1 if predicted.item() == 1 else 0  # 1 for 'fall', 0 for 'non_fall'

    # Append result to the list
    results.append({"id": os.path.basename(img_path), "label": label})

# Save results to CSV
csv_file = "results.csv"
with open(csv_file, mode="w", newline="") as file:
    writer = csv.DictWriter(file, fieldnames=["id", "label"])
    writer.writeheader()
    writer.writerows(results)

# Display distribution of labels
labels = [result["label"] for result in results]
label_counts = Counter(labels)

print("\n=== Label Distribution ===")
for label, count in label_counts.items():
    label_name = "fall" if label == 1 else "non_fall"
    print(f"{label_name}: {count}")

print(f"\nPipeline complete. Results saved to '{csv_file}'.")