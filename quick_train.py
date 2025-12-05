#!/usr/bin/env python3
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
    train_images = list(Path("datasets/traffic_lights/images/train").glob("*.jpg")) + \
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
