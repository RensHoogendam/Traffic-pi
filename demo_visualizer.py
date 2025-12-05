#!/usr/bin/env python3

"""
Crash-resistant visualizer for traffic light detection.
Handles Qt crashes and provides fallback options.
"""

import cv2
import sys
import os
import time
import traceback
from typing import List, Dict, Optional

def safe_import():
    """Safely import required modules with error handling."""
    try:
        from src.traffic_light_detector import TrafficLightDetector
        return TrafficLightDetector
    except ImportError as e:
        print(f"❌ Import error: {e}")
        print("💡 Make sure you're in the correct directory and dependencies are installed")
        return None

def safe_visualization(detector, image, description="Image"):
    """
    Safely visualize detection results with error handling.
    Falls back to basic OpenCV if advanced visualization fails.
    """
    
    try:
        # Try enhanced visualization first
        try:
            from visualizer import TrafficLightVisualizer
            visualizer = TrafficLightVisualizer(detector)
            
            # Process with enhanced visualization
            vis_image, detection_info = visualizer.visualize_detection_process(image, show_poles=True)
            
            # Add title safely
            cv2.putText(vis_image, description, (10, 50), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
            
            # Try combined view
            try:
                combined_image = visualizer.create_combined_view(vis_image, detection_info)
                return combined_image, detection_info, "enhanced"
            except Exception as e:
                print(f"⚠️ Combined view failed, using standard: {e}")
                return vis_image, detection_info, "standard"
                
        except Exception as e:
            print(f"⚠️ Enhanced visualization failed: {e}")
            print("🔄 Falling back to basic detection...")
            
            # Fallback to basic detection
            detections = detector.detect_traffic_lights(image)
            
            # Draw basic detections
            result_image = detector.draw_detections(image, detections, show_method=True)
            
            # Add title
            cv2.putText(result_image, description, (10, 50), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
            
            # Create basic detection info
            detection_info = {
                'detections': detections,
                'method_used': 'basic_fallback',
                'processing_time': 0,
                'pole_regions': []
            }
            
            return result_image, detection_info, "basic"
    
    except Exception as e:
        print(f"❌ Critical visualization error: {e}")
        traceback.print_exc()
        
        # Ultimate fallback - just return original image
        cv2.putText(image, f"ERROR: {description}", (10, 50), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)
        return image, {'detections': [], 'method_used': 'error', 'processing_time': 0, 'pole_regions': []}, "error"

def safe_display(image, window_name="Traffic Light Detection"):
    """
    Safely display image with error handling.
    """
    try:
        cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
        cv2.resizeWindow(window_name, min(1200, image.shape[1]), min(800, image.shape[0]))
        cv2.imshow(window_name, image)
        return True
    except Exception as e:
        print(f"⚠️ Display error: {e}")
        # Try to save image instead
        try:
            output_path = f"fallback_output_{int(time.time())}.jpg"
            cv2.imwrite(output_path, image)
            print(f"💾 Image saved to: {output_path}")
            return False
        except Exception as save_error:
            print(f"❌ Could not save image: {save_error}")
            return False

def demo_visualization():
    """Run a crash-resistant demo of the visualization features."""
    print("🚦 Traffic Light Detection - Crash-Resistant Demo")
    print("=" * 60)
    
    try:
        # Initialize detector safely
        TrafficLightDetector = safe_import()
        if TrafficLightDetector is None:
            return
        
        print("🔧 Initializing detector...")
        detector = TrafficLightDetector(config_path="config/detection_config.yaml", use_yolo=True)
        print("✅ Detector initialized successfully!")
        
        # Test images with error handling
        test_images = [
            ("data/synthetic_test.jpg", "Simple synthetic scene"),
            ("data/performance_test.jpg", "Complex synthetic scene"), 
            ("data/difficult_test.jpg", "Difficult detection scene")
        ]
        
        print(f"\n🧪 Testing {len(test_images)} scenarios...")
        print("💡 Controls: Any key = next image, 'q' = quit, 's' = save current image")
        
        successful_demos = 0
        
        for i, (image_path, description) in enumerate(test_images):
            print(f"\n--- Demo {i+1}/{len(test_images)}: {description} ---")
            
            try:
                # Load image with error handling
                if not os.path.exists(image_path):
                    print(f"⚠️ Test image not found: {image_path}")
                    print("🔧 Creating synthetic test image...")
                    
                    # Create a simple test image
                    test_image = create_synthetic_test_image(description)
                else:
                    test_image = cv2.imread(image_path)
                    
                if test_image is None:
                    print(f"❌ Could not load {image_path}, skipping...")
                    continue
                
                print(f"📐 Image size: {test_image.shape[1]}x{test_image.shape[0]}")
                
                # Process with safe visualization
                start_time = time.time()
                vis_image, detection_info, vis_mode = safe_visualization(detector, test_image, f"Demo {i+1}: {description}")
                process_time = time.time() - start_time
                
                # Display results safely
                display_success = safe_display(vis_image, f"Demo {i+1} - {vis_mode.title()} Mode")
                
                # Print results
                print(f"🎯 Results:")
                print(f"   Method: {detection_info['method_used']}")
                print(f"   Processing time: {process_time*1000:.1f}ms")
                print(f"   Visualization mode: {vis_mode}")
                print(f"   Poles found: {len(detection_info['pole_regions'])}")
                print(f"   Traffic lights: {len(detection_info['detections'])}")
                
                if detection_info['detections']:
                    for j, detection in enumerate(detection_info['detections'], 1):
                        state = detection['state'].value if hasattr(detection['state'], 'value') else str(detection['state'])
                        conf = detection['confidence']
                        print(f"     Light {j}: {state} (confidence: {conf:.3f})")
                
                if display_success:
                    # Wait for user input
                    print("⌨️  Press any key to continue, 'q' to quit, 's' to save...")
                    key = cv2.waitKey(0) & 0xFF
                    
                    if key == ord('q'):
                        print("🛑 Demo stopped by user")
                        break
                    elif key == ord('s'):
                        save_path = f"demo_result_{i+1}_{int(time.time())}.jpg"
                        cv2.imwrite(save_path, vis_image)
                        print(f"💾 Saved to: {save_path}")
                
                successful_demos += 1
                
            except Exception as e:
                print(f"❌ Error in demo {i+1}: {e}")
                print("🔄 Continuing to next demo...")
                continue
        
        # Cleanup
        try:
            cv2.destroyAllWindows()
        except:
            pass
        
        # Summary
        print(f"\n📊 Demo Summary:")
        print(f"✅ Completed demos: {successful_demos}/{len(test_images)}")
        print(f"🎯 Features tested:")
        print(f"   ✓ Crash-resistant visualization")
        print(f"   ✓ Fallback error handling")
        print(f"   ✓ Multiple detection modes")
        print(f"   ✓ Safe display with alternatives")
        
        if successful_demos > 0:
            print(f"\n🚀 Next steps:")
            print(f"   • Try interactive modes: python visualizer.py")
            print(f"   • Test with camera: python visualizer.py --camera")
            print(f"   • Process videos: python visualizer.py --video <path>")
        
    except Exception as e:
        print(f"❌ Critical demo error:")
        print(f"Error: {e}")
        traceback.print_exc()
        
        print(f"\n🛠️ Troubleshooting:")
        print(f"• Check if all dependencies are installed: pip install -r requirements.txt")
        print(f"• Verify you're in the correct directory")
        print(f"• Try the safe mode: python demo_visualizer.py")

def create_synthetic_test_image(description):
    """Create a simple synthetic test image when real test images are missing."""
    
    # Create a simple image with colored rectangles to simulate traffic lights
    image = cv2.rectangle(cv2.zeros((600, 800, 3), dtype=np.uint8), (0, 0), (800, 600), (50, 50, 50), -1)
    
    # Add some "traffic lights"
    if "simple" in description.lower():
        # Single traffic light
        cv2.rectangle(image, (350, 150), (450, 350), (100, 100, 100), -1)  # pole
        cv2.circle(image, (400, 200), 20, (0, 0, 255), -1)  # red light
        cv2.circle(image, (400, 250), 20, (0, 255, 255), -1)  # yellow light
        cv2.circle(image, (400, 300), 20, (0, 255, 0), -1)  # green light
    
    elif "complex" in description.lower():
        # Multiple traffic lights
        positions = [(300, 150), (500, 150), (650, 200)]
        colors = [(0, 0, 255), (0, 255, 255), (0, 255, 0)]
        
        for (x, y), color in zip(positions, colors):
            cv2.rectangle(image, (x-25, y), (x+25, y+150), (100, 100, 100), -1)
            cv2.circle(image, (x, y+50), 15, color, -1)
    
    # Add title
    cv2.putText(image, "SYNTHETIC TEST IMAGE", (10, 30), 
               cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
    
    return image

def interactive_mode():
    """Run interactive crash-resistant mode."""
    print("🎮 Interactive Mode - Crash Resistant")
    print("=" * 50)
    
    try:
        TrafficLightDetector = safe_import()
        if TrafficLightDetector is None:
            return
        
        detector = TrafficLightDetector(config_path="config/detection_config.yaml", use_yolo=True)
        
        while True:
            print(f"\n🎯 Options:")
            print(f"1. Process image file")
            print(f"2. Process video file")
            print(f"3. Test camera (if available)")
            print(f"4. Run demo scenarios")
            print(f"5. Exit")
            
            choice = input(f"\nEnter choice (1-5): ").strip()
            
            if choice == "1":
                path = input("Enter image path: ").strip()
                if path and os.path.exists(path):
                    image = cv2.imread(path)
                    if image is not None:
                        vis_image, info, mode = safe_visualization(detector, image, os.path.basename(path))
                        safe_display(vis_image, f"Image: {os.path.basename(path)}")
                        cv2.waitKey(0)
                        cv2.destroyAllWindows()
            
            elif choice == "2":
                path = input("Enter video path: ").strip()
                if path:
                    process_video_safe(detector, path)
            
            elif choice == "3":
                test_camera_safe(detector)
            
            elif choice == "4":
                demo_visualization()
            
            elif choice == "5":
                print("👋 Goodbye!")
                break
    
    except KeyboardInterrupt:
        print("\n👋 Goodbye!")
    except Exception as e:
        print(f"❌ Interactive mode error: {e}")

def process_video_safe(detector, video_path):
    """Process video with crash resistance."""
    try:
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            print(f"❌ Could not open video: {video_path}")
            return
        
        print(f"🎥 Processing video. Press 'q' to quit, 'p' to pause")
        
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            
            try:
                vis_frame, _, _ = safe_visualization(detector, frame, "Video Frame")
                if safe_display(vis_frame, "Video Processing"):
                    key = cv2.waitKey(1) & 0xFF
                    if key == ord('q'):
                        break
            except Exception as e:
                print(f"⚠️ Frame processing error: {e}")
                continue
        
        cap.release()
        cv2.destroyAllWindows()
        
    except Exception as e:
        print(f"❌ Video processing error: {e}")

def test_camera_safe(detector):
    """Test camera with crash resistance."""
    try:
        cap = cv2.VideoCapture(0)
        if not cap.isOpened():
            print("❌ Could not open camera")
            return
        
        print("📷 Camera test. Press 'q' to quit")
        
        while True:
            ret, frame = cap.read()
            if not ret:
                print("❌ Could not read from camera")
                break
            
            try:
                vis_frame, _, _ = safe_visualization(detector, frame, "Camera Feed")
                if safe_display(vis_frame, "Camera Test"):
                    if cv2.waitKey(1) & 0xFF == ord('q'):
                        break
            except Exception as e:
                print(f"⚠️ Camera frame error: {e}")
                continue
        
        cap.release()
        cv2.destroyAllWindows()
        
    except Exception as e:
        print(f"❌ Camera test error: {e}")

if __name__ == "__main__":
    import numpy as np
    
    if len(sys.argv) > 1 and sys.argv[1] == "--interactive":
        interactive_mode()
    else:
        demo_visualization()