#!/usr/bin/env python3

"""
Performance test for traffic light detection systems.
Compares single-stage vs two-stage detection performance.
"""

import time
import cv2
import numpy as np
from src.traffic_light_detector import TrafficLightDetector

def create_complex_test_image():
    """Create a more complex test image with multiple objects and a traffic light."""
    # Create a larger image with more complexity
    img = np.zeros((800, 1200, 3), dtype=np.uint8)
    
    # Add background noise/texture
    noise = np.random.randint(20, 60, (800, 1200, 3), dtype=np.uint8)
    img = cv2.addWeighted(img, 0.7, noise, 0.3, 0)
    
    # Draw some building-like structures (potential pole candidates)
    cv2.rectangle(img, (50, 100), (90, 700), (80, 80, 80), -1)  # Building edge
    cv2.rectangle(img, (200, 150), (230, 650), (100, 100, 100), -1)  # Pole-like structure
    cv2.rectangle(img, (400, 50), (450, 750), (90, 90, 90), -1)  # Another pole
    cv2.rectangle(img, (800, 100), (840, 600), (70, 70, 70), -1)  # Building
    
    # Add a traffic light on one of the poles
    light_x, light_y = 380, 200
    light_w, light_h = 80, 200
    
    # Traffic light housing (dark gray)
    cv2.rectangle(img, (light_x, light_y), (light_x + light_w, light_y + light_h), (40, 40, 40), -1)
    
    # Red light (active)
    cv2.circle(img, (light_x + light_w//2, light_y + 40), 25, (0, 0, 255), -1)
    cv2.circle(img, (light_x + light_w//2, light_y + 40), 20, (100, 100, 255), -1)  # Bright center
    
    # Yellow light (inactive)
    cv2.circle(img, (light_x + light_w//2, light_y + 100), 25, (0, 100, 200), -1)
    
    # Green light (inactive)  
    cv2.circle(img, (light_x + light_w//2, light_y + 160), 25, (0, 100, 0), -1)
    
    # Add some other objects that might confuse detection
    cv2.circle(img, (600, 300), 30, (0, 255, 255), -1)  # Yellow sign
    cv2.circle(img, (700, 400), 25, (0, 0, 200), -1)    # Red sign
    cv2.rectangle(img, (900, 200), (950, 250), (255, 255, 255), -1)  # White sign
    
    return img

def benchmark_detection_method(detector, image, method_name, iterations=5):
    """Benchmark a detection method."""
    print(f"\n=== Testing {method_name} ===")
    
    times = []
    results = []
    
    for i in range(iterations):
        start_time = time.time()
        detections = detector.detect_traffic_lights(image)
        end_time = time.time()
        
        elapsed = end_time - start_time
        times.append(elapsed)
        results.append(len(detections))
        
        print(f"Run {i+1}: {elapsed:.3f}s, {len(detections)} detections")
    
    avg_time = sum(times) / len(times)
    avg_detections = sum(results) / len(results)
    
    print(f"Average: {avg_time:.3f}s, {avg_detections:.1f} detections")
    return avg_time, avg_detections

def main():
    print("🚀 Performance Test: Single-Stage vs Two-Stage Detection")
    print("=" * 60)
    
    # Create test image
    print("Creating complex test image...")
    test_image = create_complex_test_image()
    cv2.imwrite("data/performance_test.jpg", test_image)
    print("✅ Test image saved to data/performance_test.jpg")
    
    # Test single-stage detection
    print("\n🔧 Setting up single-stage detector...")
    config_path = "config/detection_config.yaml"
    
    # Load config and modify for single-stage
    import yaml
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    
    config['enable_pole_detection'] = False
    config['enable_multiscale'] = False
    config['enable_preprocessing'] = False
    
    # Create temporary config
    with open("config/single_stage_config.yaml", 'w') as f:
        yaml.dump(config, f)
    
    detector_single = TrafficLightDetector(config_path="config/single_stage_config.yaml", use_yolo=True)
    detector_single.debug = True
    
    single_time, single_detections = benchmark_detection_method(
        detector_single, test_image, "Single-Stage Detection", 3
    )
    
    # Test two-stage detection
    print("\n🔧 Setting up two-stage detector...")
    config['enable_pole_detection'] = True
    
    with open("config/two_stage_config.yaml", 'w') as f:
        yaml.dump(config, f)
    
    detector_two_stage = TrafficLightDetector(config_path="config/two_stage_config.yaml", use_yolo=True)
    detector_two_stage.debug = True
    
    two_stage_time, two_stage_detections = benchmark_detection_method(
        detector_two_stage, test_image, "Two-Stage Detection", 3
    )
    
    # Results summary
    print("\n" + "=" * 60)
    print("📊 PERFORMANCE COMPARISON")
    print("=" * 60)
    print(f"Single-Stage: {single_time:.3f}s avg, {single_detections:.1f} detections")
    print(f"Two-Stage:    {two_stage_time:.3f}s avg, {two_stage_detections:.1f} detections")
    
    speedup = single_time / two_stage_time if two_stage_time > 0 else float('inf')
    print(f"Speedup:      {speedup:.2f}x {'faster' if speedup > 1 else 'slower'}")
    
    if two_stage_detections >= single_detections:
        print("✅ Two-stage maintains or improves detection accuracy")
    else:
        print("⚠️  Two-stage has fewer detections")
    
    # Cleanup
    import os
    try:
        os.remove("config/single_stage_config.yaml")
        os.remove("config/two_stage_config.yaml")
    except:
        pass
    
    print(f"\n🎯 Recommendation: {'Use two-stage detection' if speedup > 1 and two_stage_detections >= single_detections else 'Keep single-stage detection'}")

if __name__ == "__main__":
    main()