#!/usr/bin/env python3

"""
Create a test image with multiple poles and traffic lights to showcase the schematic panel.
"""

import cv2
import numpy as np
from visualizer import TrafficLightVisualizer
from src.traffic_light_detector import TrafficLightDetector

def create_multi_pole_scene():
    """Create a scene with multiple poles and traffic lights."""
    # Create a larger image
    img = np.zeros((800, 1200, 3), dtype=np.uint8)
    
    # Add some background texture
    noise = np.random.randint(10, 40, (800, 1200, 3), dtype=np.uint8)
    img = cv2.addWeighted(img, 0.8, noise, 0.2, 0)
    
    # Create multiple pole structures
    poles_info = [
        (100, 50, 20, 600),   # Pole 1 - left side
        (400, 100, 25, 500),  # Pole 2 - center-left
        (700, 80, 20, 550),   # Pole 3 - center-right
        (1000, 70, 22, 580),  # Pole 4 - right side
    ]
    
    # Draw poles
    for px, py, pw, ph in poles_info:
        cv2.rectangle(img, (px, py), (px + pw, py + ph), (70, 70, 70), -1)
    
    # Add traffic lights on some poles
    traffic_lights = [
        # (pole_index, offset_x, offset_y, state)
        (0, -10, 150, 'red'),     # On pole 1
        (1, -15, 120, 'green'),   # On pole 2  
        (2, -12, 180, 'yellow'),  # On pole 3
        (3, -15, 140, 'red'),     # On pole 4
    ]
    
    for pole_idx, offset_x, offset_y, state in traffic_lights:
        px, py, pw, ph = poles_info[pole_idx]
        
        # Traffic light position
        light_x = px + pw + offset_x
        light_y = py + offset_y
        light_w = 50
        light_h = 120
        
        # Traffic light housing
        cv2.rectangle(img, (light_x, light_y), (light_x + light_w, light_y + light_h), (40, 40, 40), -1)
        cv2.rectangle(img, (light_x, light_y), (light_x + light_w, light_y + light_h), (80, 80, 80), 2)
        
        # Light positions
        center_x = light_x + light_w // 2
        red_y = light_y + 25
        yellow_y = light_y + 60
        green_y = light_y + 95
        
        # Draw all lights (dim)
        cv2.circle(img, (center_x, red_y), 15, (50, 0, 0), -1)
        cv2.circle(img, (center_x, yellow_y), 15, (50, 50, 0), -1)
        cv2.circle(img, (center_x, green_y), 15, (0, 50, 0), -1)
        
        # Activate the appropriate light        if state == 'red':\n            cv2.circle(img, (center_x, red_y), 15, (0, 0, 255), -1)\n            cv2.circle(img, (center_x, red_y), 10, (100, 100, 255), -1)  # Bright center\n        elif state == 'yellow':\n            cv2.circle(img, (center_x, yellow_y), 15, (0, 255, 255), -1)\n            cv2.circle(img, (center_x, yellow_y), 10, (150, 255, 255), -1)\n        elif state == 'green':\n            cv2.circle(img, (center_x, green_y), 15, (0, 255, 0), -1)\n            cv2.circle(img, (center_x, green_y), 10, (100, 255, 100), -1)\n    \n    return img\n\ndef test_schematic_panel():\n    \"\"\"Test the schematic panel with a complex multi-pole scene.\"\"\"\n    print(\"\ud83c\udfa8 Testing Schematic Panel with Multi-Pole Scene\")\n    print(\"=\" * 60)\n    \n    # Create test scene\n    print(\"Creating multi-pole test scene...\")\n    test_image = create_multi_pole_scene()\n    cv2.imwrite(\"data/multi_pole_test.jpg\", test_image)\n    print(\"\u2705 Multi-pole test image saved to data/multi_pole_test.jpg\")\n    \n    # Initialize detector and visualizer\n    print(\"\\nInitializing detector...\")\n    detector = TrafficLightDetector(config_path=\"config/detection_config.yaml\", use_yolo=True)\n    visualizer = TrafficLightVisualizer(detector)\n    \n    # Process with visualization\n    print(\"Processing with enhanced visualization...\")\n    vis_image, detection_info = visualizer.visualize_detection_process(test_image, show_poles=True)\n    \n    # Create combined view with schematic\n    combined_image = visualizer.create_combined_view(vis_image, detection_info)\n    \n    # Add title\n    cv2.putText(combined_image, \"SCHEMATIC PANEL DEMO - Multi-Pole Scene\", (10, 30), \n               cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)\n    \n    # Save result\n    cv2.imwrite(\"data/multi_pole_schematic_demo.jpg\", combined_image)\n    print(\"\u2705 Schematic demo saved to data/multi_pole_schematic_demo.jpg\")\n    \n    # Display results\n    print(f\"\\n\ud83d\udcca Results:\")\n    print(f\"Poles detected: {len(detection_info['pole_regions'])}\")\n    print(f\"Traffic lights detected: {len(detection_info['detections'])}\")\n    print(f\"Method used: {detection_info['method_used']}\")\n    print(f\"Processing time: {detection_info['processing_time']*1000:.1f}ms\")\n    \n    for i, det in enumerate(detection_info['detections']):\n        print(f\"  TL{i+1}: {det['state'].value} (confidence: {det['confidence']:.3f})\")\n    \n    # Show interactive display\n    print(\"\\n\ud83d\udd0d Showing interactive display...\")\n    print(\"Top: Original image with overlay annotations\")\n    print(\"Bottom: Computer-generated schematic diagram\")\n    print(\"Press any key to close...\")\n    \n    cv2.imshow(\"Schematic Panel Demo\", combined_image)\n    cv2.waitKey(0)\n    cv2.destroyAllWindows()\n    \n    print(\"\\n\ud83c\udf86 Schematic Panel Features:\")\n    print(\"\u2705 Simplified pole representations (orange rectangles)\")\n    print(\"\u2705 Traffic light icons with state visualization\")\n    print(\"\u2705 Automatic scaling to fit all detected infrastructure\")\n    print(\"\u2705 Grid background for spatial reference\")\n    print(\"\u2705 Legend and labels for easy interpretation\")\n    print(\"\u2705 Real-time generation from detection data\")\n\nif __name__ == \"__main__\":\n    test_schematic_panel()"