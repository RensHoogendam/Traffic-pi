# Traffic-pi 🚦

A computer vision system for detecting and classifying traffic lights using Python and OpenCV. Designed to work on Raspberry Pi and regular computers for traffic monitoring applications.

## Features

- **🤖 AI-Powered Detection**: Uses YOLOv8 for accurate traffic light detection with color-based fallback
- **🚦 State Classification**: Determine if lights are red, yellow/amber, or green
- **📱 Multiple Input Sources**: Support for images, videos, YouTube URLs, camera feeds, and batch processing
- **🔧 Dual Detection Modes**: YOLO for accuracy, color-based for speed and legacy support
- **🥧 Raspberry Pi Compatible**: Optimized to run on Raspberry Pi for edge deployment
- **⚙️ Configurable**: Adjustable detection parameters via configuration files
- **💻 Easy to Use**: Simple command-line interface and Python API

## Quick Start

### Easy Installation (Recommended)

**One command setup:**

```bash
git clone https://github.com/RensHoogendam/Traffic-pi.git
cd Traffic-pi
bash setup.sh
```

Or using Make:

```bash
git clone https://github.com/RensHoogendam/Traffic-pi.git
cd Traffic-pi
make setup
```

That's it! The script automatically:

- Creates a virtual environment
- Installs all dependencies
- Sets up the project in development mode

### Manual Installation

If you prefer manual setup:

1. **Clone and create virtual environment:**

   ```bash
   git clone https://github.com/RensHoogendam/Traffic-pi.git
   cd Traffic-pi
   python3 -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

2. **Install:**
   ```bash
   pip install -e .
   ```

### Basic Usage

**First, activate the virtual environment:**

```bash
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

#### Quick Commands with Make:

```bash
make help          # Show all available commands
make run-test      # Run system test
make run-camera    # Start camera detection
make test          # Run test suite
```

#### CLI Commands:

**Detect traffic lights in a single image:**

```bash
traffic-pi --image path/to/image.jpg
```

**Process a video file:**

```bash
traffic-pi --video path/to/video.mp4 --output results/output_video.mp4
```

**Process a YouTube video:**

```bash
traffic-pi --video "https://www.youtube.com/watch?v=VIDEO_ID" --output results/youtube_result.mp4
```

**Use live camera feed:**

```bash
traffic-pi --camera
```

**Batch process multiple images:**

```bash
traffic-pi --batch path/to/images/ --output results/
```

**Alternative command:**

```bash
traffic-detect --image path/to/image.jpg  # Same as traffic-pi
```

**Detection Mode Options:**
```bash
# Use YOLO detection (default - most accurate)
traffic-pi --image path/to/image.jpg

# Use color-based detection (faster, works without GPU)
traffic-pi --no-yolo --image path/to/image.jpg
```

#### Test the Installation:

```bash
python test_system.py
```

## Project Structure

```
Traffic-pi/
├── src/
│   └── traffic_light_detector.py  # Main detection class
├── config/
│   └── detection_config.yaml      # Configuration parameters
├── data/                          # Input images/videos
├── models/                        # Trained models (if using ML approaches)
├── tests/                         # Unit tests
├── main.py                        # Command-line interface
├── requirements.txt               # Python dependencies
└── README.md                      # This file
```

## YouTube Video Support

Traffic-pi now supports processing videos directly from YouTube! Just provide a YouTube URL instead of a local file path:

```bash
# Process any YouTube video with traffic footage
traffic-pi --video "https://www.youtube.com/watch?v=VIDEO_ID"

# Save the processed result
traffic-pi --video "https://youtu.be/VIDEO_ID" --output results/youtube_analysis.mp4
```

**Supported YouTube URL formats:**

- `https://www.youtube.com/watch?v=VIDEO_ID`
- `https://youtu.be/VIDEO_ID`
- `https://m.youtube.com/watch?v=VIDEO_ID`

The system automatically:

1. Downloads the video in the best available quality
2. Processes it for traffic light detection
3. Cleans up temporary files after processing

## Detection Methods

Traffic-pi supports two detection approaches:

### 🤖 YOLO Detection (Default)
- **What it is**: Uses YOLOv8 neural network trained on millions of images
- **Accuracy**: Extremely high - can detect traffic lights in complex scenes
- **Speed**: Fast on GPU, moderate on CPU
- **Best for**: Production use, complex traffic scenes, varying lighting
- **Requirements**: Downloads ~6MB model on first use

### 🎨 Color-Based Detection (Fallback)
- **What it is**: HSV color space analysis with shape detection
- **Accuracy**: Good for clear, well-lit traffic lights
- **Speed**: Very fast, no GPU needed
- **Best for**: Simple scenes, resource-constrained devices, development
- **Requirements**: Only OpenCV

The system automatically falls back to color-based detection if YOLO fails to load.

## How It Works

### YOLO Detection Pipeline:
1. **Object Detection**: YOLOv8 identifies traffic light objects in the scene
2. **Region Extraction**: Extracts detected traffic light regions
3. **State Classification**: Analyzes color distribution within each region
4. **Confidence Scoring**: Combines YOLO confidence with color analysis

### Color-Based Detection Pipeline:
1. **Color Segmentation**: Converts images to HSV color space and creates masks for red, yellow, and green colors
2. **Shape Analysis**: Identifies circular/elliptical shapes that match traffic light characteristics  
3. **Contour Detection**: Finds and validates contours based on area, circularity, and other properties
4. **Clustering**: Groups nearby lights that form traffic light patterns
5. **Confidence Scoring**: Assigns confidence scores based on shape quality and color intensity### Detection Algorithm

- **HSV Color Ranges**:
  - Red: [0°-10°, 170°-180°] hue ranges
  - Yellow: [20°-30°] hue range
  - Green: [40°-80°] hue range
- **Shape Validation**: Circularity > 0.3, appropriate area range
- **Morphological Operations**: Opening and closing to clean up noise
- **Confidence Metrics**: Fill ratio and circularity combined

## Configuration

Create a `config/detection_config.yaml` file to customize detection parameters:

```yaml
min_contour_area: 100
max_contour_area: 10000
aspect_ratio_tolerance: 0.3
circularity_threshold: 0.3
brightness_threshold: 100
```

## API Usage

```python
from src.traffic_light_detector import TrafficLightDetector
import cv2

# Initialize detector
detector = TrafficLightDetector(config_path="config/detection_config.yaml")

# Load image
image = cv2.imread("path/to/image.jpg")

# Detect traffic lights
detections = detector.detect_traffic_lights(image)

# Process results
for detection in detections:
    state = detection['state']  # TrafficLightState enum
    confidence = detection['confidence']  # Float 0-1
    bbox = detection['bbox']  # (x, y, width, height)
    print(f"Found {state.value} light with confidence {confidence:.2f}")
```

## Raspberry Pi Setup

For Raspberry Pi deployment:

1. **Enable camera (if using Pi camera):**

   ```bash
   sudo raspi-config  # Enable camera interface
   ```

2. **Install Pi-specific packages:**

   ```bash
   # Uncomment RPi.GPIO and picamera2 lines in requirements.txt
   pip install RPi.GPIO picamera2
   ```

3. **Optimize for Pi:**
   - Use lower resolution images for faster processing
   - Consider using hardware acceleration if available
   - Adjust detection parameters for your specific use case

## Testing

Run the test suite:

```bash
cd tests
python -m pytest
```

## Performance Tips

- **Image Resolution**: Lower resolution = faster processing (try 640x480 for real-time)
- **ROI Selection**: Process only regions of interest to improve speed
- **Parameter Tuning**: Adjust HSV ranges and thresholds for your lighting conditions
- **Hardware**: Use GPU acceleration if available (OpenCV with CUDA)

## Contributing

1. Fork the repository
2. Create a feature branch: `git checkout -b feature-name`
3. Make your changes and add tests
4. Commit your changes: `git commit -am 'Add some feature'`
5. Push to the branch: `git push origin feature-name`
6. Submit a pull request

## Troubleshooting

### Common Issues

**No detections found:**

- Check lighting conditions (avoid very bright or dark images)
- Adjust HSV color ranges in the configuration
- Ensure traffic lights are clearly visible and not too small/large

**False positives:**

- Increase `circularity_threshold` parameter
- Adjust `min_contour_area` and `max_contour_area`
- Fine-tune color ranges to be more restrictive

**Poor performance:**

- Reduce image resolution
- Process only regions of interest
- Consider using a more powerful device

### Getting Help

- Check existing [issues](https://github.com/RensHoogendam/Traffic-pi/issues)
- Create a new issue with details about your problem
- Include sample images and error messages

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Acknowledgments

- OpenCV community for excellent computer vision tools
- Raspberry Pi Foundation for making edge AI accessible
- Traffic engineering community for domain knowledge

---

**Made with ❤️ for safer and smarter traffic systems**
