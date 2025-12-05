# Traffic Light Annotation Guide

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
