# Traffic Light Model Training Guide

## 🎯 Complete Training Workflow

Your training environment is now fully set up! Here's how to use it:

## 📂 Directory Structure Created
```
Traffic-pi/
├── datasets/traffic_lights/          # YOLO dataset format
│   ├── images/
│   │   ├── train/                   # Training images
│   │   ├── val/                     # Validation images  
│   │   └── test/                    # Test images
│   └── labels/
│       ├── train/                   # Training labels (.txt)
│       ├── val/                     # Validation labels
│       └── test/                    # Test labels
├── training_data/                   # Raw data collection
│   ├── raw_videos/                  # Store your videos here
│   ├── extracted_frames/            # Extracted frames for annotation
│   └── annotations/                 # Annotation files
├── models/
│   ├── custom/                      # Your trained models
│   └── experiments/                 # Training experiments
└── training_config.yaml             # Optimized training settings
```

## 🚀 Quick Start Commands

### Step 1: Collect Training Data
```bash
# Extract frames from your YouTube video or local video
python collect_data.py --video "path/to/video.mp4" --max-frames 500

# Or from your existing YouTube URL
# First download the video, then extract frames
```

### Step 2: Annotate Your Data
```bash
# Option A: Use LabelImg (recommended for beginners)
pip install labelImg
labelimg training_data/extracted_frames

# Option B: Use Roboflow (web-based)
# Go to roboflow.com, create project, upload images
```

### Step 3: Organize Dataset
```bash
# Move annotated images and labels to proper folders
# Images go to: datasets/traffic_lights/images/train/
# Labels go to: datasets/traffic_lights/labels/train/
```

### Step 4: Start Training
```bash
# Quick training with defaults
python quick_train.py

# Or full training with custom config
python train_model.py
```

## 📊 Training Tips

### Dataset Size Recommendations:
- **Minimum**: 200 images per class (800 total)
- **Good**: 500 images per class (2000 total)  
- **Excellent**: 1000+ images per class (4000+ total)

### Class Balance:
- Red lights: 25%
- Green lights: 25% 
- Yellow lights: 25%
- Unknown/Off: 25%

### Data Variety:
- Different times of day (dawn, day, dusk, night)
- Various weather conditions
- Multiple distances (close, medium, far)
- Different camera angles
- Various traffic light configurations

## 🎛️ Configuration Files

### `traffic_lights.yaml` - Dataset Config
- Points to your dataset folders
- Defines class names and IDs

### `training_config.yaml` - Training Settings  
- Optimized hyperparameters for traffic lights
- Augmentation settings
- Learning rate schedule

## 📈 Monitoring Training

Training will create:
- `runs/detect/traffic-light-quick/` - Training results
- Tensorboard logs for visualization
- Model checkpoints every 10 epochs
- Validation metrics and plots

## 🔧 Troubleshooting

### GPU Memory Issues
```yaml
# Reduce batch size in training_config.yaml
batch: 4  # Instead of 16
```

### Poor Results
1. Add more training data
2. Improve annotation quality
3. Increase training epochs
4. Adjust learning rate

### No Training Data Found
```bash
# Check if images are in correct folders
ls datasets/traffic_lights/images/train/
ls datasets/traffic_lights/labels/train/
```

## 🎯 Next Steps After Training

1. **Test Your Model**:
   ```bash
   # Update config to use your trained model
   # yolo_model: "runs/detect/traffic-light-quick/weights/best.pt"
   ```

2. **Evaluate Performance**:
   ```bash
   python -m src.main --image test_image.jpg
   ```

3. **Deploy to Production**:
   - Copy best.pt to your models/ folder
   - Update detection_config.yaml
   - Test with real traffic videos

## 📚 Additional Resources

- **LabelImg Tutorial**: https://github.com/HumanSignal/labelImg
- **Roboflow Guide**: https://docs.roboflow.com/
- **YOLO Training Docs**: https://docs.ultralytics.com/
- **Annotation Guidelines**: See ANNOTATION_GUIDE.md

Happy training! 🚦🤖