import os
import shutil
import random
from pathlib import Path
import cv2
import numpy as np

# Define paths
DATASET_ROOT = Path('DatasetV12')
SOURCE_DIRS = {
    'Crack': 0,
    'Pothole': 1,
    'Surface Erosion': 2
}

# Create directories if they don't exist
for split in ['train', 'val', 'test']:
    (DATASET_ROOT / 'images' / split).mkdir(parents=True, exist_ok=True)
    (DATASET_ROOT / 'labels' / split).mkdir(parents=True, exist_ok=True)

# Split ratios
TRAIN_RATIO = 0.7
VAL_RATIO = 0.2
TEST_RATIO = 0.1

def create_yolo_label(img_path, class_id):
    """Create a YOLO format label file for an image."""
    # Read image to get dimensions
    img = cv2.imread(str(img_path))
    if img is None:
        print(f"Could not read image: {img_path}")
        return None
    
    height, width = img.shape[:2]
    
    # For classification, we'll create a label that covers the whole image
    # Format: class_id x_center y_center width height
    x_center = 0.5  # center of image
    y_center = 0.5  # center of image
    w = 1.0        # full width
    h = 1.0        # full height
    
    return f"{class_id} {x_center} {y_center} {w} {h}"

def process_dataset():
    # Process each class
    for class_name, class_id in SOURCE_DIRS.items():
        source_dir = DATASET_ROOT / class_name
        if not source_dir.exists():
            print(f"Warning: {source_dir} does not exist")
            continue
            
        # Get all images
        images = list(source_dir.glob('*.jpg')) + list(source_dir.glob('*.png'))
        random.shuffle(images)
        
        # Calculate split sizes
        n_images = len(images)
        n_train = int(n_images * TRAIN_RATIO)
        n_val = int(n_images * VAL_RATIO)
        
        # Split images
        train_images = images[:n_train]
        val_images = images[n_train:n_train + n_val]
        test_images = images[n_train + n_val:]
        
        # Process each split
        for split, split_images in [('train', train_images), ('val', val_images), ('test', test_images)]:
            for img_path in split_images:
                # Copy image
                dest_img_path = DATASET_ROOT / 'images' / split / img_path.name
                shutil.copy2(img_path, dest_img_path)
                
                # Create and save label
                label = create_yolo_label(img_path, class_id)
                if label:
                    label_path = DATASET_ROOT / 'labels' / split / f"{img_path.stem}.txt"
                    with open(label_path, 'w') as f:
                        f.write(label)

if __name__ == "__main__":
    # Set random seed for reproducibility
    random.seed(42)
    
    # Process the dataset
    process_dataset()
    
    # Create data.yaml file
    yaml_content = f"""path: {DATASET_ROOT.absolute()}  # dataset root dir
train: images/train  # train images (relative to 'path')
val: images/val  # val images (relative to 'path')
test: images/test  # test images (relative to 'path')

# Classes
names:
  0: Crack
  1: Pothole
  2: Surface Erosion
"""
    
    with open(DATASET_ROOT / 'data.yaml', 'w') as f:
        f.write(yaml_content)
    
    print("Dataset preparation completed!") 