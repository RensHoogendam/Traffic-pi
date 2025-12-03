#!/usr/bin/env python3

"""
Demo script showing the enhanced traffic light detection visualization.
"""

import cv2
import time
from visualizer import TrafficLightVisualizer
from src.traffic_light_detector import TrafficLightDetector

def demo_visualization():
    """Run a demo showing the visualization features."""
    print("🚦 Traffic Light Detection Visualization Demo")
    print("=" * 60)
    
    # Initialize detector
    print("Initializing detector...")
    detector = TrafficLightDetector(config_path="config/detection_config.yaml", use_yolo=True)
    visualizer = TrafficLightVisualizer(detector)
    
    # Test images
    test_images = [
        ("data/synthetic_test.jpg", "Simple synthetic scene"),
        ("data/performance_test.jpg", "Complex synthetic scene"), 
        ("data/difficult_test.jpg", "Difficult detection scene")
    ]
    
    print(f"\nTesting {len(test_images)} scenarios...")
    print("Press any key to advance to next image, 'q' to quit")
    
    for i, (image_path, description) in enumerate(test_images):
        print(f"\n--- Demo {i+1}/{len(test_images)}: {description} ---")
        
        # Load and process image
        image = cv2.imread(image_path)
        if image is None:
            print(f"❌ Could not load {image_path}")
            continue
        
        # Process with visualization
        vis_image, detection_info = visualizer.visualize_detection_process(image, show_poles=True)
        
        # Add demo title
        cv2.putText(vis_image, f"Demo {i+1}: {description}", (10, 50), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
        
        # Create combined view with schematic panel
        combined_image = visualizer.create_combined_view(vis_image, detection_info)
        
        # Show results
        cv2.imshow("Traffic Light Detection Demo", combined_image)
        
        # Print info
        print(f"Method: {detection_info['method_used']}")
        print(f"Processing time: {detection_info['processing_time']*1000:.1f}ms")
        print(f"Poles found: {len(detection_info['pole_regions'])}")
        print(f"Traffic lights detected: {len(detection_info['detections'])}")
        
        # Wait for key press
        key = cv2.waitKey(0) & 0xFF
        if key == ord('q'):
            break
    
    cv2.destroyAllWindows()
    
    print("\n🎯 Demo Features Showcased:")
    print("✅ Smart single-stage vs two-stage detection")
    print("✅ Pole detection visualization (blue rectangles)")
    print("✅ Traffic light state detection (colored rectangles)")
    print("✅ Method indicators (1S = single-stage, 2S = two-stage)")
    print("✅ Real-time performance metrics")
    print("✅ Confidence visualization bars")
    print("✅ Automatic scene complexity adaptation")
    
    print(f"\n🚀 Try the interactive modes:")
    print(f"Camera: python visualizer.py --camera")
    print(f"Image:  python visualizer.py --image <path>")

if __name__ == "__main__":
    demo_visualization()