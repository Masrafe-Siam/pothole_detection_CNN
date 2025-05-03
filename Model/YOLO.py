import torch
from ultralytics import YOLO
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path
import os
import sys

# Get the current directory
current_dir = Path(os.path.dirname(os.path.abspath(__file__)))
parent_dir = current_dir.parent

# Change to the parent directory
os.chdir(parent_dir)

# Initialize YOLO model
model = YOLO('yolov8n.pt')  # Load the smallest YOLOv8 model

# Define dataset path - point to the data.yaml file
dataset_path = Path('Dataset/data.yaml')

# Verify the data.yaml file exists
if not dataset_path.exists():
    print(f"Error: data.yaml not found at {dataset_path}")
    print("Please run prepare_dataset.py first")
    sys.exit(1)

# Verify dataset structure
required_dirs = [
    'Dataset/images/train',
    'Dataset/images/val',
    'Dataset/images/test',
    'Dataset/labels/train',
    'Dataset/labels/val',
    'Dataset/labels/test'
]

for dir_path in required_dirs:
    if not Path(dir_path).exists():
        print(f"Error: Required directory {dir_path} does not exist")
        print("Please run prepare_dataset.py first")
        sys.exit(1)

# Train the model
try:
    print("Starting training...")
    print(f"Using dataset at: {dataset_path.absolute()}")
    
    results = model.train(
        data=str(dataset_path),  # path to data.yaml file
        epochs=50,              # number of epochs
        imgsz=640,             # image size
        batch=16,              # batch size
        name='road_damage_detection',  # experiment name
        device='cpu'           # use CPU for training
    )
    print("Training completed successfully!")
    
except Exception as e:
    print(f"Error during training: {str(e)}")
    print("\nTroubleshooting steps:")
    print("1. Make sure you have run prepare_dataset.py first")
    print("2. Check if the Dataset directory structure is correct")
    print("3. Verify you have read/write permissions in the Dataset directory")
    print("4. Try running the script as administrator")
    sys.exit(1)

# Function to predict on new images
def predict_image(model, image_path):
    results = model(image_path)
    return results[0]  # Return first result

# Visualize predictions
def plot_predictions(image_path, results):
    plt.figure(figsize=(10, 10))
    plt.imshow(results.plot())
    plt.axis('off')
    plt.show()

# Example: Test on a single image
test_image_path = Path('Dataset/images/test/your_test_image.jpg')  # Replace with actual test image path

if test_image_path.exists():
    results = predict_image(model, test_image_path)
    plot_predictions(test_image_path, results)
    
    # Print detection results
    for r in results.boxes.data.tolist():
        x1, y1, x2, y2, score, class_id = r
        print(f"Detected class {int(class_id)} with confidence {score:.2f}")
else:
    print(f"Test image not found at {test_image_path}") 