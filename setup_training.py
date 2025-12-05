#!/usr/bin/env python3
"""
Setup and Training Workflow Script

This script provides a complete workflow for setting up and training
custom traffic light detection models.
"""

import os
import sys
from pathlib import Path
import subprocess
import argparse

def setup_training_environment():
    """Set up the complete training environment."""
    
    print("🚦 Traffic Light Model Training Setup")
    print("=" * 50)
    
    # Check dependencies
    try:
        import ultralytics
        import torch
        print("✅ Dependencies found: ultralytics, torch")
    except ImportError as e:
        print(f"❌ Missing dependency: {e}")
        print("📦 Installing required packages...")
        subprocess.check_call([sys.executable, "-m", "pip", "install", "ultralytics", "torch", "torchvision"])
        print("✅ Dependencies installed!")
    
    # Create directories (already done, but just to show)
    print("\n📁 Training directories ready!")
    
    # Create helper scripts
    create_data_collection_script()
    create_training_config()
    create_quick_start_script()
    
    print("\n🎯 Setup Complete! Next Steps:")
    print("1. 📹 Collect training data: python collect_data.py")
    print("2. 🏷️  Annotate images (use LabelImg or Roboflow)")
    print("3. 🚀 Start training: python quick_train.py")

def create_data_collection_script():
    """Create a script to collect training data from your existing videos."""
    
    script_content = '''#!/usr/bin/env python3
"""
Data Collection Script
Extract frames from videos for training data.
"""

import cv2
import os
from pathlib import Path
import argparse

def extract_frames_from_video(video_path, output_dir, interval=30, max_frames=1000):
    """Extract frames from video for annotation."""
    
    print(f"🎬 Extracting frames from: {video_path}")
    
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"❌ Could not open video: {video_path}")
        return
    
    frame_count = 0
    saved_count = 0
    
    while saved_count < max_frames:
        ret, frame = cap.read()
        if not ret:
            break
        
        if frame_count % interval == 0:
            # Save frame
            filename = f"frame_{saved_count:06d}.jpg"
            filepath = os.path.join(output_dir, filename)
            cv2.imwrite(filepath, frame)
            saved_count += 1
            
            if saved_count % 100 == 0:
                print(f"  📸 Extracted {saved_count} frames...")
        
        frame_count += 1
    
    cap.release()
    print(f"✅ Extracted {saved_count} frames to {output_dir}")

def main():
    parser = argparse.ArgumentParser(description="Extract training frames from videos")
    parser.add_argument("--video", required=True, help="Path to input video")
    parser.add_argument("--output", default="training_data/extracted_frames", help="Output directory")
    parser.add_argument("--interval", type=int, default=30, help="Extract every N frames")
    parser.add_argument("--max-frames", type=int, default=1000, help="Maximum frames to extract")
    
    args = parser.parse_args()
    
    extract_frames_from_video(args.video, args.output, args.interval, args.max_frames)

if __name__ == "__main__":
    main()
'''
    
    with open('collect_data.py', 'w') as f:
        f.write(script_content)
    
    print("📝 Created: collect_data.py")

def create_training_config():
    """Create an optimized training configuration."""
    
    config_content = '''# Optimized Training Configuration for Traffic Lights
# This config is tuned for traffic light detection

# Model settings
model: yolov8s.pt  # Start with small model for good accuracy/speed balance

# Training parameters  
epochs: 100
batch: 16          # Adjust based on your GPU memory
imgsz: 640        # Image size for training
device: 0         # GPU device (use 'cpu' for CPU training)

# Learning rate settings
lr0: 0.01         # Initial learning rate
lrf: 0.1          # Final learning rate factor
momentum: 0.937   # SGD momentum
weight_decay: 0.0005

# Augmentation settings (good for traffic lights)
hsv_h: 0.015      # Hue augmentation
hsv_s: 0.7        # Saturation augmentation  
hsv_v: 0.4        # Value augmentation
degrees: 10.0     # Rotation degrees
translate: 0.1    # Translation
scale: 0.5        # Scale augmentation
shear: 2.0        # Shear degrees
perspective: 0.0  # Perspective augmentation
flipud: 0.0       # Vertical flip (don't flip traffic lights!)
fliplr: 0.5       # Horizontal flip
mosaic: 1.0       # Mosaic augmentation
mixup: 0.1        # MixUp augmentation

# Validation settings
val: true         # Validate during training
save_period: 10   # Save checkpoint every N epochs
patience: 50      # Early stopping patience

# Optimization
optimizer: SGD    # Optimizer (SGD, Adam, AdamW)
close_mosaic: 10  # Disable mosaic in final epochs

# Loss weights (tune for traffic lights)
box: 7.5          # Box loss weight
cls: 0.5          # Classification loss weight  
dfl: 1.5          # Distribution focal loss weight
'''
    
    with open('training_config.yaml', 'w') as f:
        f.write(config_content)
    
    print("📝 Created: training_config.yaml")

def create_quick_start_script():
    """Create a quick start training script."""
    
    script_content = '''#!/usr/bin/env python3
"""
Quick Start Training Script
Train a traffic light model with minimal setup.
"""

from ultralytics import YOLO
import os

def quick_train():
    """Quick training with sensible defaults."""
    
    print("🚀 Starting Quick Training...")
    
    # Check if dataset exists
    if not os.path.exists("datasets/traffic_lights/images/train"):
        print("❌ No training data found!")
        print("📋 Please run: python collect_data.py --video YOUR_VIDEO.mp4")
        print("🏷️  Then annotate your images and organize them in the dataset folder")
        return
    
    # Check if there are images in training folder
    train_images = list(Path("datasets/traffic_lights/images/train").glob("*.jpg")) + \\
                  list(Path("datasets/traffic_lights/images/train").glob("*.png"))
    
    if len(train_images) == 0:
        print("❌ No training images found in datasets/traffic_lights/images/train/")
        return
        
    print(f"📊 Found {len(train_images)} training images")
    
    # Load model
    model = YOLO('yolov8s.pt')  # Start with pre-trained model
    
    # Train
    results = model.train(
        data='traffic_lights.yaml',
        epochs=50,              # Start with fewer epochs for testing
        imgsz=640,
        batch=8,               # Conservative batch size
        name='traffic-light-quick',
        device='0',            # Use GPU if available
        patience=20,
        save_period=10,
        verbose=True
    )
    
    print("✅ Training completed!")
    print(f"💾 Model saved to: runs/detect/traffic-light-quick/weights/best.pt")
    
    # Test the trained model
    test_model = YOLO('runs/detect/traffic-light-quick/weights/best.pt')
    
    # Run validation
    val_results = test_model.val()
    print(f"📊 Validation mAP50: {val_results.box.map50}")
    
    return results

if __name__ == "__main__":
    from pathlib import Path
    quick_train()
'''
    
    with open('quick_train.py', 'w') as f:
        f.write(script_content)
    
    print("📝 Created: quick_train.py")

def create_annotation_guide():
    """Create a guide for annotating traffic lights."""
    
    guide_content = '''# Traffic Light Annotation Guide

## 🎯 What to Annotate

### Traffic Light Classes:
- **Red Light**: traffic_light_red (class 0)
- **Yellow Light**: traffic_light_yellow (class 1)  
- **Green Light**: traffic_light_green (class 2)
- **Unknown/Off**: traffic_light_unknown (class 3)

## 🏷️ Annotation Rules

### 1. Bounding Boxes
- Draw tight boxes around individual traffic light bulbs
- Include the colored lens but not the housing
- Each colored light gets its own annotation

### 2. What NOT to Annotate
- Traffic light housing/frame (only the colored lights)
- Lights that are clearly off/dark
- Reflections or glare
- Very small/distant lights (< 10 pixels)

### 3. Quality Guidelines
- Minimum box size: 15x15 pixels
- Clear state identification required
- No partial occlusions > 50%

## 🛠️ Recommended Tools

### LabelImg (Desktop)
```bash
pip install labelImg
labelimg datasets/traffic_lights/images/train
```

### Roboflow (Web)
1. Upload images to roboflow.com
2. Create project with traffic light classes
3. Annotate online
4. Export in YOLO format

### CVAT (Advanced)
- Best for large datasets
- Team collaboration features
- cvat.org

## 📊 Dataset Split Recommendations
- Training: 70-80% of annotated images
- Validation: 15-20% of annotated images  
- Test: 5-10% of annotated images (optional)

## 💡 Tips
- Annotate various conditions (day/night/weather)
- Include different distances and angles
- Ensure balanced class distribution
- Quality over quantity - better to have fewer high-quality annotations
'''
    
    with open('ANNOTATION_GUIDE.md', 'w') as f:
        f.write(guide_content)
    
    print("📝 Created: ANNOTATION_GUIDE.md")

if __name__ == "__main__":
    setup_training_environment()
    create_annotation_guide()