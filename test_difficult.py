#!/usr/bin/env python3

"""
Test case for difficult detection scenarios where two-stage detection helps.
"""

import cv2
import numpy as np
from src.traffic_light_detector import TrafficLightDetector
import time

def create_difficult_scene():
    """Create a scene where traffic lights are hard to detect initially."""
    # Create a more complex scene with poor lighting
    img = np.zeros((600, 1000, 3), dtype=np.uint8)
    
    # Add heavy noise and poor lighting
    noise = np.random.randint(0, 100, (600, 1000, 3), dtype=np.uint8)
    img = cv2.addWeighted(img, 0.5, noise, 0.5, 0)
    
    # Add many distracting objects
    for _ in range(15):
        x, y = np.random.randint(50, 950), np.random.randint(50, 550)
        w, h = np.random.randint(20, 80), np.random.randint(20, 80)
        color = np.random.randint(0, 255, 3).tolist()
        cv2.rectangle(img, (x, y), (x + w, y + h), color, -1)
    
    # Add some pole-like structures
    cv2.rectangle(img, (200, 50), (220, 500), (60, 60, 60), -1)  # Pole 1
    cv2.rectangle(img, (700, 100), (720, 550), (70, 70, 70), -1)  # Pole 2
    
    # Add very small, dim traffic lights that are hard to detect
    # Light 1 - very small and dim
    light_x, light_y = 180, 150
    cv2.rectangle(img, (light_x, light_y), (light_x + 40, light_y + 100), (30, 30, 30), -1)
    # Very dim red light
    cv2.circle(img, (light_x + 20, light_y + 25), 12, (0, 0, 80), -1)
    cv2.circle(img, (light_x + 20, light_y + 25), 8, (50, 50, 120), -1)
    
    # Light 2 - also small and dim
    light_x2, light_y2 = 680, 200
    cv2.rectangle(img, (light_x2, light_y2), (light_x2 + 40, light_y2 + 100), (25, 25, 25), -1)
    # Very dim green light
    cv2.circle(img, (light_x2 + 20, light_y2 + 75), 12, (0, 60, 0), -1)
    cv2.circle(img, (light_x2 + 20, light_y2 + 75), 8, (50, 100, 50), -1)
    
    return img

def test_difficult_detection():
    """Test detection on a difficult scene."""
    print("🔍 Testing Difficult Detection Scenario")
    print("=" * 50)
    
    # Create difficult test image
    print("Creating difficult scene with small, dim traffic lights...")
    test_image = create_difficult_scene()
    cv2.imwrite("data/difficult_test.jpg", test_image)
    print("✅ Difficult test image saved to data/difficult_test.jpg")
    
    # Test with auto detection enabled
    print("\n🤖 Testing with smart auto-detection...")
    detector = TrafficLightDetector(config_path="config/detection_config.yaml", use_yolo=True)
    detector.debug = True
    
    start_time = time.time()
    detections = detector.detect_traffic_lights(test_image)
    elapsed = time.time() - start_time
    
    print(f"\n📊 Results:")
    print(f"Time: {elapsed:.3f}s")
    print(f"Detections: {len(detections)}")
    
    if detections:
        for i, det in enumerate(detections):
            print(f"  Light {i+1}: {det['state'].value} (conf: {det['confidence']:.3f})")
    else:
        print("  No detections found")
    
    # Draw results
    result_img = detector.draw_detections(test_image.copy(), detections)
    cv2.imwrite("data/difficult_result.jpg", result_img)
    print(f"✅ Result saved to data/difficult_result.jpg")
    
    return len(detections), elapsed

def main():
    detections, time_taken = test_difficult_detection()
    
    print(f"\n🎯 Summary:")
    print(f"The smart detection system {'succeeded' if detections > 0 else 'struggled'} with this difficult scene")
    print(f"Processing time: {time_taken:.3f}s")
    print(f"Detections found: {detections}")
    
    print(f"\n💡 The system automatically chooses the best detection strategy:")
    print(f"   • If single-stage works → use it (faster)")
    print(f"   • If no detections → try two-stage with lower confidence")
    print(f"   • If too many detections → use pole filtering")

if __name__ == "__main__":
    main()