# 🏷️ Traffic Light Annotation Guide

## Quick Start - Multiple Options Available

### ❌ **Problem**: labelImg crashes with Python 3.14
```
TypeError: setValue(self, a0: int): argument 1 has unexpected type 'float'
```

### ✅ **Solutions Available**:

## 1. 🎯 Custom Annotation Tool (Recommended)
**Best for**: Simple workflow, Python 3.14 compatibility, YOLO format

```bash
python annotation_tool.py training_data/extracted_frames
```

**Controls**:
- **Left-click + drag**: Draw bounding box around traffic lights
- **1**: Red traffic light class
- **2**: Yellow traffic light class  
- **3**: Green traffic light class
- **4**: Unknown traffic light class
- **Space/Enter**: Save and move to next image
- **Backspace**: Go to previous image
- **U**: Undo last annotation
- **C**: Clear all annotations
- **S**: Save current annotations
- **Q**: Quit

## 2. 🔧 Patched LabelImg
**Best for**: Full labelImg features with compatibility fix

```bash
python labelimg_patch.py training_data/extracted_frames
```

## 3. 🤖 Auto-Detection Workflow
**Best for**: Automatic tool selection

```bash
python annotation_workflow.py training_data/extracted_frames
```

## 4. 🌐 Web-Based Options
**Best for**: Advanced features, team collaboration

```bash
python annotation_workflow.py --web
```

Opens browser with links to:
- [makesense.ai](https://www.makesense.ai/) - Free online tool
- [Roboflow](https://roboflow.com/) - Advanced features  
- [Labelbox](https://labelbox.com/) - Professional solution

---

## 📋 Annotation Guidelines

### Traffic Light Classes:
1. **Red (Class 0)**: Red traffic light (stop)
2. **Yellow (Class 1)**: Yellow/amber traffic light (caution)
3. **Green (Class 2)**: Green traffic light (go)  
4. **Unknown (Class 3)**: Unclear state or broken lights

### Annotation Tips:
- **Draw tight boxes**: Include just the light, not the entire fixture
- **Multiple lights**: Annotate each individual light separately
- **Horizontal spans**: Each light gets its own box
- **Vertical poles**: Each light gets its own box
- **Partial visibility**: Annotate if >50% of light is visible
- **Blurry lights**: Use "unknown" class if state is unclear

### Quality Checklist:
- ✅ Box tightly fits the light circle/shape
- ✅ Correct class assigned (red/yellow/green/unknown)
- ✅ No overlapping boxes for same light
- ✅ All visible lights annotated
- ✅ Consistent across similar scenes

---

## 🔄 Complete Workflow

### Step 1: Extract Frames
```bash
python collect_data.py --extract video_file.mp4
```

### Step 2: Annotate (Choose One Method)
```bash
# Option A: Custom tool
python annotation_tool.py training_data/extracted_frames

# Option B: Patched labelImg  
python labelimg_patch.py training_data/extracted_frames

# Option C: Auto-detection
python annotation_workflow.py training_data/extracted_frames
```

### Step 3: Verify Annotations
```bash
ls training_data/extracted_frames_labels/
# Should see .txt files matching each image
```

### Step 4: Train Model
```bash
python train_model.py
```

---

## 📊 File Formats

### YOLO Format (Used by all tools):
```
# Each line: class_id center_x center_y width height
0 0.5 0.3 0.1 0.15    # Red light at center-top
1 0.5 0.5 0.1 0.15    # Yellow light at center
2 0.5 0.7 0.1 0.15    # Green light at center-bottom
```

### Directory Structure:
```
training_data/
├── extracted_frames/           # Input images
│   ├── frame_000000.jpg
│   ├── frame_000001.jpg
│   └── ...
└── extracted_frames_labels/    # Output annotations
    ├── frame_000000.txt
    ├── frame_000001.txt
    └── ...
```

---

## 🛠️ Troubleshooting

### labelImg Issues:
- **Python 3.14 crash**: Use `labelimg_patch.py` or custom tool
- **Qt display errors**: Use `annotation_tool.py` (OpenCV-based)
- **Import errors**: Check virtual environment activation

### Custom Tool Issues:
- **No display**: Check OpenCV installation: `pip install opencv-python`
- **No images found**: Verify image directory path
- **Annotations not saved**: Check directory permissions

### Performance Tips:
- **Large datasets**: Annotate in batches of 50-100 images
- **Multiple people**: Use web-based tools for collaboration
- **Quality control**: Review annotations before training

---

## 🎯 Next Steps

After annotation:
1. **Verify quality**: `python visualizer.py --verify-annotations`
2. **Split dataset**: Annotations automatically split during training
3. **Train model**: `python train_model.py`
4. **Test results**: `python test_system.py`

## 📞 Support

If you encounter issues:
1. Check this guide first
2. Try alternative annotation methods
3. Verify file paths and permissions
4. Check virtual environment setup