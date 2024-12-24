import os
import shutil
from glob import glob
import torch
from PIL import Image
import logging
import warnings

# Suppress warnings and logging
warnings.filterwarnings("ignore", category=FutureWarning)
logging.getLogger("torch").setLevel(logging.ERROR)
logging.getLogger("PIL").setLevel(logging.ERROR)
logging.getLogger("ultralytics").setLevel(logging.ERROR)

# Directories
test_directory = 'train'
output_directory_berdiri = 'output_berdiri'
output_directory_tidak_berdiri = 'output_tidak_berdiri'

# Remove and recreate directories
def remove_and_create_dir(directory):
    if os.path.exists(directory):
        shutil.rmtree(directory)
    os.makedirs(directory, exist_ok=True)

remove_and_create_dir(output_directory_berdiri)
remove_and_create_dir(output_directory_tidak_berdiri)
remove_and_create_dir(os.path.join(output_directory_berdiri, 'fall'))
remove_and_create_dir(os.path.join(output_directory_berdiri, 'non_fall'))
remove_and_create_dir(os.path.join(output_directory_tidak_berdiri, 'fall'))
remove_and_create_dir(os.path.join(output_directory_tidak_berdiri, 'non_fall'))


# Load YOLO model once
print("Loading YOLO model...")
device = 'cuda' if torch.cuda.is_available() else 'cpu'
print(f"Using device: {device}")
model = torch.hub.load('ultralytics/yolov5', 'yolov5s', device=device, force_reload=False)

# Variables
total_berdiri = 0
total_tidak_berdiri = 0
BOUNDARY_RATIO = 0.60

# Load image paths
image_extensions = ['*.jpg', '*.jpeg', '*.png', '*.bmp', '*.tiff']
image_paths = []
for ext in image_extensions:
    image_paths.extend(glob(os.path.join(test_directory, '**', ext), recursive=True))

# Process images in batches
batch_size = 8
print(f"Total images to process: {len(image_paths)}")
for i in range(0, len(image_paths), batch_size):
    batch_paths = image_paths[i:i+batch_size]
    print(f"Processing batch {i//batch_size + 1}/{len(image_paths)//batch_size + 1}...")
    for img_path in batch_paths:
        try:
            # Extract information from path
            path_parts = img_path.split(os.sep)
            if len(path_parts) < 4:
                print(f"Invalid path structure: {img_path}")
                continue

            subject = path_parts[-4]  # Example: "subject-1"
            action_type = path_parts[-3]  # Example: "fall" or "non_fall"
            action = path_parts[-2]  # Example: "1_jumping"
            frame_name = os.path.basename(img_path)  # Example: "frame000.jpg"

            print(f"Processing {img_path}: subject={subject}, action_type={action_type}, action={action}")

            # Open and get image dimensions
            with Image.open(img_path) as img:
                frame_width, frame_height = img.size
                img = img.resize((320, 240))  # Reduce resolution for YOLO
            boundary_y = int(BOUNDARY_RATIO * frame_height)

            # Run YOLO detection
            with torch.no_grad():
                results = model(img_path)

            detections = results.xyxy[0]
            print(f"Detections for {img_path}: {detections}")

            has_berdiri = any(
                int(cls) == 0 and box[1].item() <= boundary_y for *box, conf, cls in detections
            )
            print(f"Has berdiri for {img_path}: {has_berdiri}")

            # Determine label (fall or non_fall)
            label = 'fall' if action_type.lower() == 'fall' else 'non_fall'
            print(f"Label determined for {img_path}: {label}")

            # Set destination folder
            if has_berdiri:
                dest_dir = os.path.join(output_directory_berdiri, label)
                total_berdiri += 1
            else:
                dest_dir = os.path.join(output_directory_tidak_berdiri, label)
                total_tidak_berdiri += 1

            # Rename and copy file
            new_name = f"{subject}_{action}_{frame_name}"
            dest_path = os.path.join(dest_dir, new_name)
            shutil.copy2(img_path, dest_path)
            print(f"File copied to {dest_path}")

        except Exception as e:
            # Log errors to file
            with open('error_log.txt', 'a') as f:
                f.write(f"Error processing {img_path}: {e}\n")
            print(f"Error processing {img_path}: {e}")
            continue

# Print summary
print("\n=== Summary ===")
print(f"Total 'Berdiri': {total_berdiri}")
print(f"Total 'Tidak Berdiri': {total_tidak_berdiri}")
