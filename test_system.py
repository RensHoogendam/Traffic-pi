#!/usr/bin/env python3
"""
Simple test script to create a synthetic traffic light image and test detection
"""

import cv2
import numpy as np
import os
import sys

# Add src to path
sys.path.append('src')

from traffic_light_detector import TrafficLightDetector


def create_synthetic_traffic_light():
    """Create a synthetic traffic light image for testing"""
    # Create a blank image (white background)
    img = np.ones((400, 600, 3), dtype=np.uint8) * 255
    
    # Traffic light housing (dark gray rectangle)
    cv2.rectangle(img, (250, 50), (350, 300), (50, 50, 50), -1)
    
    # Red light (top) - currently ON
    cv2.circle(img, (300, 100), 25, (0, 0, 255), -1)  # Red circle
    cv2.circle(img, (300, 100), 25, (200, 200, 200), 3)  # Light gray border
    
    # Yellow light (middle) - OFF
    cv2.circle(img, (300, 175), 25, (100, 100, 100), -1)  # Dark gray (off)
    cv2.circle(img, (300, 175), 25, (200, 200, 200), 3)  # Light gray border
    
    # Green light (bottom) - OFF  
    cv2.circle(img, (300, 250), 25, (100, 100, 100), -1)  # Dark gray (off)
    cv2.circle(img, (300, 250), 25, (200, 200, 200), 3)  # Light gray border
    
    return img


def test_detection():
    """Test the traffic light detection system"""
    print("🚦 Testing Traffic Light Detection System...")
    
    # Create synthetic image
    print("Creating synthetic traffic light image...")
    test_image = create_synthetic_traffic_light()
    
    # Save test image
    os.makedirs('data', exist_ok=True)
    test_image_path = 'data/synthetic_test.jpg'
    cv2.imwrite(test_image_path, test_image)
    print(f"Saved test image to: {test_image_path}")
    
    # Initialize detector
    print("Initializing traffic light detector...")
    try:
        detector = TrafficLightDetector('config/detection_config.yaml')
        print("✅ Detector initialized successfully")
    except Exception as e:
        print(f"❌ Error initializing detector: {e}")
        return False
    
    # Test detection
    print("Running detection...")
    try:
        detections = detector.detect_traffic_lights(test_image)
        print(f"✅ Detection completed. Found {len(detections)} traffic lights")
        
        # Print results
        for i, detection in enumerate(detections, 1):
            state = detection['state'].value
            confidence = detection['confidence']
            bbox = detection['bbox']
            print(f"  Light {i}: State={state}, Confidence={confidence:.3f}, BBox={bbox}")
        
        # Create result image
        result_image = detector.draw_detections(test_image, detections)
        result_path = 'data/detection_result.jpg'
        cv2.imwrite(result_path, result_image)
        print(f"✅ Result saved to: {result_path}")
        
        return True
        
    except Exception as e:
        print(f"❌ Error during detection: {e}")
        return False


if __name__ == "__main__":
    success = test_detection()
    if success:
        print("\n🎉 Test completed successfully!")
        print("You can now:")
        print("1. Check the images in the data/ folder")
        print("2. Try with real images: python main.py --image path/to/your/image.jpg")
        print("3. Use camera: python main.py --camera")
    else:
        print("\n❌ Test failed. Check the error messages above.")