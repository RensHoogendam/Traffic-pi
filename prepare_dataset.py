#!/usr/bin/env python3
"""
Dataset Collection and Preparation Tools

This script helps you prepare training data for traffic light detection.
"""

import cv2
import os
import json
from pathlib import Path

def collect_frames_from_video(video_path: str, output_dir: str, frame_interval: int = 30):
    """
    Extract frames from video for annotation.
    
    Args:
        video_path: Path to input video
        output_dir: Directory to save extracted frames
        frame_interval: Extract every Nth frame
    """
    
    print(f"🎬 Extracting frames from {video_path}")
    
    # Create output directory
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    
    # Open video
    cap = cv2.VideoCapture(video_path)
    frame_count = 0
    saved_count = 0
    
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        
        # Save every Nth frame
        if frame_count % frame_interval == 0:
            frame_filename = f"frame_{saved_count:06d}.jpg"
            frame_path = os.path.join(output_dir, frame_filename)
            cv2.imwrite(frame_path, frame)
            saved_count += 1
            
            if saved_count % 100 == 0:
                print(f"  Extracted {saved_count} frames...")
        
        frame_count += 1
    
    cap.release()
    print(f"✅ Extracted {saved_count} frames to {output_dir}")

def create_annotation_template(image_dir: str, output_file: str):
    """
    Create a template for manual annotation.
    """
    
    image_files = list(Path(image_dir).glob("*.jpg")) + list(Path(image_dir).glob("*.png"))
    
    annotation_template = {
        "info": {
            "description": "Traffic Light Dataset",
            "version": "1.0",
            "year": 2024
        },
        "categories": [
            {"id": 0, "name": "traffic_light_red"},
            {"id": 1, "name": "traffic_light_yellow"},
            {"id": 2, "name": "traffic_light_green"},
            {"id": 3, "name": "traffic_light_unknown"}
        ],
        "images": [],
        "annotations": []
    }
    
    for i, img_path in enumerate(image_files):
        img = cv2.imread(str(img_path))
        height, width = img.shape[:2]
        
        annotation_template["images"].append({
            "id": i,
            "file_name": img_path.name,
            "width": width,
            "height": height
        })
    
    with open(output_file, 'w') as f:
        json.dump(annotation_template, f, indent=2)
    
    print(f"📝 Created annotation template: {output_file}")
    print("💡 Use tools like LabelImg or Roboflow to annotate your images")

def convert_to_yolo_format(coco_json: str, output_dir: str):
    """
    Convert COCO format annotations to YOLO format.
    """
    
    print(f"🔄 Converting {coco_json} to YOLO format")
    
    with open(coco_json, 'r') as f:
        data = json.load(f)
    
    # Create output directory
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    
    # Create image_id to filename mapping
    images = {img['id']: img for img in data['images']}
    
    # Group annotations by image
    annotations_by_image = {}
    for ann in data['annotations']:
        img_id = ann['image_id']
        if img_id not in annotations_by_image:
            annotations_by_image[img_id] = []
        annotations_by_image[img_id].append(ann)
    
    # Convert each image's annotations
    for img_id, annotations in annotations_by_image.items():
        img_info = images[img_id]
        img_width = img_info['width']
        img_height = img_info['height']
        
        # Create YOLO format label file
        label_filename = Path(img_info['file_name']).stem + '.txt'
        label_path = os.path.join(output_dir, label_filename)
        
        with open(label_path, 'w') as f:
            for ann in annotations:
                # Convert COCO bbox to YOLO format
                x, y, width, height = ann['bbox']
                
                # Convert to center coordinates and normalize
                center_x = (x + width / 2) / img_width
                center_y = (y + height / 2) / img_height
                norm_width = width / img_width
                norm_height = height / img_height
                
                # Write YOLO format: class_id center_x center_y width height
                f.write(f"{ann['category_id']} {center_x:.6f} {center_y:.6f} {norm_width:.6f} {norm_height:.6f}\n")
    
    print(f"✅ Converted annotations to YOLO format in {output_dir}")

def prepare_training_data():
    """
    Complete workflow for preparing training data.
    """
    
    print("🚦 Traffic Light Dataset Preparation Workflow")
    print("=" * 50)
    
    print("\n📋 Steps to create your training dataset:")
    print("1. 🎬 Extract frames from videos using collect_frames_from_video()")
    print("2. 🏷️  Annotate images using tools like LabelImg or Roboflow")
    print("3. 🔄 Convert annotations to YOLO format")
    print("4. 📂 Organize into train/val/test splits")
    print("5. 🚀 Train your model using train_model.py")
    
    print("\n💡 Annotation Tools:")
    print("• LabelImg: https://github.com/HumanSignal/labelImg")
    print("• Roboflow: https://roboflow.com (web-based)")
    print("• CVAT: https://cvat.org (advanced)")
    
    print("\n📊 Dataset Size Recommendations:")
    print("• Minimum: 500 images per class")
    print("• Good: 1000-2000 images per class") 
    print("• Excellent: 5000+ images per class")
    
    print("\n🎯 Data Collection Tips:")
    print("• Include various lighting conditions (day/night/dawn/dusk)")
    print("• Different weather (sunny/cloudy/rainy/foggy)")
    print("• Multiple distances (close/medium/far)")
    print("• Various angles and orientations")
    print("• Different traffic light types and configurations")

if __name__ == "__main__":
    prepare_training_data()
    
    # Example usage (uncomment to use):
    # collect_frames_from_video("path/to/video.mp4", "extracted_frames", frame_interval=30)
    # create_annotation_template("extracted_frames", "annotations.json")
    # convert_to_yolo_format("annotated_data.json", "yolo_labels")