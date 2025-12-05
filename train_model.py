#!/usr/bin/env python3
"""
Traffic Light Model Training Script

This script shows how to fine-tune YOLO models for better traffic light detection.
You can use this to train on your own traffic light dataset.
"""

from ultralytics import YOLO
import os

def train_traffic_light_model():
    """
    Train a custom YOLO model for traffic light detection.
    """
    
    print("🚦 Training Custom Traffic Light Detection Model")
    print("=" * 50)
    
    # Start with a pre-trained model (transfer learning)
    model = YOLO('yolov8s.pt')  # or yolov8n.pt for faster training
    
    # Train the model on your custom dataset
    # You'll need to create a dataset in YOLO format first
    results = model.train(
        data='traffic_lights.yaml',  # Path to your dataset config
        epochs=100,                  # Number of training epochs
        imgsz=640,                   # Image size
        batch=16,                    # Batch size (adjust based on your GPU memory)
        name='traffic-light-model',  # Name for this training run
        device='0',                  # GPU device (use 'cpu' if no GPU)
        
        # Optimization settings
        lr0=0.01,                   # Initial learning rate
        weight_decay=0.0005,        # Weight decay for regularization
        mosaic=1.0,                 # Mosaic augmentation probability
        mixup=0.1,                  # MixUp augmentation probability
        
        # Early stopping and saving
        patience=50,                # Early stopping patience
        save_period=10,             # Save checkpoint every N epochs
        
        # Validation settings
        val=True,                   # Validate during training
        plots=True,                 # Generate training plots
        verbose=True                # Verbose output
    )
    
    # Export the trained model
    model.export(format='onnx')  # Optional: export to ONNX for deployment
    
    print("✅ Training completed!")
    print(f"📊 Results: {results}")
    print(f"💾 Model saved as: runs/detect/traffic-light-model/weights/best.pt")

def create_dataset_config():
    """
    Create a dataset configuration file for training.
    """
    
    dataset_config = """
# Traffic Light Dataset Configuration
# Save this as 'traffic_lights.yaml'

# Dataset paths
path: ./datasets/traffic_lights  # Root directory
train: images/train              # Training images (relative to path)
val: images/val                  # Validation images (relative to path)
test: images/test               # Test images (optional)

# Classes (customize based on your needs)
names:
  0: traffic_light_red
  1: traffic_light_yellow  
  2: traffic_light_green
  3: traffic_light_unknown

# Number of classes
nc: 4
"""
    
    with open('traffic_lights.yaml', 'w') as f:
        f.write(dataset_config)
    
    print("📝 Created dataset configuration: traffic_lights.yaml")
    print("📁 You need to organize your data like this:")
    print("""
    datasets/
    └── traffic_lights/
        ├── images/
        │   ├── train/          # Training images (.jpg, .png)
        │   ├── val/            # Validation images
        │   └── test/           # Test images (optional)
        └── labels/
            ├── train/          # Training labels (.txt files)
            ├── val/            # Validation labels
            └── test/           # Test labels (optional)
    """)

if __name__ == "__main__":
    # Create dataset configuration template
    create_dataset_config()
    
    # Uncomment to start training (after you prepare your dataset)
    # train_traffic_light_model()