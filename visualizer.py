#!/usr/bin/env python3

"""
Interactive Traffic Light Detection Visualizer

Real-time visualization showing:
- Pole detection regions (blue rectangles)
- Traffic light detections (colored rectangles with state)
- Detection method indicators
- Performance metrics
- Live camera feed or image processing
"""

import cv2
import numpy as np
import time
from typing import List, Dict, Tuple
import argparse
from src.traffic_light_detector import TrafficLightDetector, TrafficLightState

class TrafficLightVisualizer:
    """Enhanced visualizer for traffic light detection with pole visualization."""
    
    def __init__(self, detector: TrafficLightDetector):
        self.detector = detector
        self.detector.debug = True  # Enable debug output for pole detection
        self.font = cv2.FONT_HERSHEY_SIMPLEX
        self.font_scale = 0.6
        self.font_thickness = 2
        
        # Colors for different elements
        self.colors = {
            'pole': (255, 128, 0),      # Orange for poles
            'pole_search': (255, 200, 100),  # Light orange for search areas
            'red': (0, 0, 255),         # Red traffic light
            'yellow': (0, 255, 255),    # Yellow traffic light
            'green': (0, 255, 0),       # Green traffic light
            'unknown': (128, 128, 128), # Gray for unknown
            'text_bg': (0, 0, 0),       # Black background for text
            'text': (255, 255, 255),    # White text
            'confidence': (255, 255, 0), # Cyan for confidence bars
            'method_single': (0, 255, 255), # Cyan for single-stage
            'method_two_stage': (255, 0, 255), # Magenta for two-stage
        }
        
        # Performance tracking
        self.frame_times = []
        self.detection_history = []
        
    def visualize_detection_process(self, image: np.ndarray, show_poles: bool = True) -> Tuple[np.ndarray, Dict]:
        """
        Visualize the complete detection process including poles and lights.
        """
        start_time = time.time()
        
        # Create a copy for visualization
        vis_image = image.copy()
        
        # Store original detector debug state and enable it
        original_debug = getattr(self.detector, 'debug', False)
        self.detector.debug = True
        
        # Intercept the detection process to get pole information
        detection_info = {
            'pole_regions': [],
            'search_regions': [],
            'detections': [],
            'method_used': 'unknown',
            'processing_time': 0,
            'pole_detection_time': 0
        }
        
        # Hook into the detection process
        if show_poles:
            pole_start = time.time()
            
            # Get pole regions if two-stage detection might be used
            auto_pole = self.detector.config.get('auto_pole_detection', True)
            force_pole = self.detector.config.get('enable_pole_detection', False)
            
            if auto_pole or force_pole:
                # Try to get pole regions
                pole_regions = self.detector._detect_pole_structures(image)
                detection_info['pole_regions'] = pole_regions
                detection_info['pole_detection_time'] = time.time() - pole_start
                
                # Visualize pole regions
                self._draw_pole_regions(vis_image, pole_regions)
        
        # Perform actual detection
        detections = self.detector.detect_traffic_lights(image)
        
        # Restore original debug state
        self.detector.debug = original_debug
        
        # Calculate processing time
        processing_time = time.time() - start_time
        detection_info['processing_time'] = processing_time
        detection_info['detections'] = detections
        
        # Determine method used (approximate based on timing and results)
        if processing_time > 0.08:  # Likely two-stage
            detection_info['method_used'] = 'two-stage'
        else:
            detection_info['method_used'] = 'single-stage'
        
        # Draw detections
        self._draw_traffic_lights(vis_image, detections)
        
        # Draw UI overlay
        self._draw_ui_overlay(vis_image, detection_info)
        
        # Update performance tracking
        self.frame_times.append(processing_time)
        if len(self.frame_times) > 30:  # Keep last 30 frames
            self.frame_times.pop(0)
            
        self.detection_history.append(len(detections))
        if len(self.detection_history) > 30:
            self.detection_history.pop(0)
        
        return vis_image, detection_info
    
    def create_combined_view(self, vis_image: np.ndarray, detection_info: Dict) -> np.ndarray:
        """
        Create a combined view with the main image on top and schematic panel below.
        """        
        # Create schematic panel
        panel_width = vis_image.shape[1]
        panel_height = 300
        schematic_panel = self._create_schematic_panel(detection_info, panel_width, panel_height)
        
        # Combine vertically
        combined_height = vis_image.shape[0] + panel_height
        combined = np.zeros((combined_height, panel_width, 3), dtype=np.uint8)
        
        # Place main image on top
        combined[0:vis_image.shape[0], 0:panel_width] = vis_image
        
        # Place schematic panel below
        combined[vis_image.shape[0]:combined_height, 0:panel_width] = schematic_panel
        
        return combined
    
    def _create_schematic_panel(self, detection_info: Dict, panel_width: int = 400, panel_height: int = 300) -> np.ndarray:
        """
        Create a computer-generated schematic diagram of detected poles and traffic lights.
        """
        # Create black panel
        panel = np.zeros((panel_height, panel_width, 3), dtype=np.uint8)
        
        # Draw title
        cv2.putText(panel, "DETECTED INFRASTRUCTURE", (10, 25), 
                   self.font, 0.6, self.colors['text'], 2)
        
        # Draw grid background
        grid_color = (30, 30, 30)
        for x in range(0, panel_width, 20):
            cv2.line(panel, (x, 40), (x, panel_height), grid_color, 1)
        for y in range(40, panel_height, 20):
            cv2.line(panel, (0, y), (panel_width, y), grid_color, 1)
        
        pole_regions = detection_info['pole_regions']
        detections = detection_info['detections']
        
        if not pole_regions and not detections:
            # No data to display
            cv2.putText(panel, "No infrastructure detected", (50, panel_height//2), 
                       self.font, 0.5, (128, 128, 128), 1)
            return panel
        
        # Calculate scale to fit all objects in the panel
        all_objects = pole_regions + [(d['bbox'][0], d['bbox'][1], d['bbox'][2], d['bbox'][3]) for d in detections]
        
        if all_objects:
            # Find bounds
            min_x = min(obj[0] for obj in all_objects)
            max_x = max(obj[0] + obj[2] for obj in all_objects)
            min_y = min(obj[1] for obj in all_objects)
            max_y = max(obj[1] + obj[3] for obj in all_objects)
            
            # Calculate scale and offset
            scene_width = max_x - min_x
            scene_height = max_y - min_y
            
            if scene_width > 0 and scene_height > 0:
                scale_x = (panel_width - 60) / scene_width
                scale_y = (panel_height - 100) / scene_height
                scale = min(scale_x, scale_y) * 0.8  # Leave some margin
                
                offset_x = (panel_width - scene_width * scale) // 2
                offset_y = 60 + (panel_height - 100 - scene_height * scale) // 2
            else:
                scale, offset_x, offset_y = 1, 50, 80
        else:
            scale, offset_x, offset_y = 1, 50, 80
        
        # Draw poles as simplified structures
        for i, (x, y, w, h) in enumerate(pole_regions):
            # Scale coordinates
            px = int((x - (all_objects[0][0] if all_objects else 0)) * scale + offset_x)
            py = int((y - (all_objects[0][1] if all_objects else 0)) * scale + offset_y)
            pw = max(8, int(w * scale * 0.3))  # Make poles narrower in schematic
            ph = int(h * scale)
            
            # Draw pole as vertical rectangle
            cv2.rectangle(panel, (px, py), (px + pw, py + ph), self.colors['pole'], -1)
            
            # Add pole label
            cv2.putText(panel, f"P{i+1}", (px - 5, py - 5), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.4, self.colors['pole'], 1)
        
        # Draw traffic lights as simplified icons
        for i, detection in enumerate(detections):
            x, y, w, h = detection['bbox']
            state = detection['state']
            
            # Scale coordinates
            lx = int((x - (all_objects[0][0] if all_objects else 0)) * scale + offset_x)
            ly = int((y - (all_objects[0][1] if all_objects else 0)) * scale + offset_y)
            lw = max(12, int(w * scale * 0.5))
            lh = max(20, int(h * scale * 0.5))
            
            # Draw traffic light housing
            cv2.rectangle(panel, (lx, ly), (lx + lw, ly + lh), (60, 60, 60), -1)
            cv2.rectangle(panel, (lx, ly), (lx + lw, ly + lh), (100, 100, 100), 2)
            
            # Draw active light based on state
            light_size = min(lw, lh) // 4
            center_x = lx + lw // 2
            
            # Red light position
            red_y = ly + lh // 6
            # Yellow light position  
            yellow_y = ly + lh // 2
            # Green light position
            green_y = ly + lh - lh // 6
            
            # Draw all lights (dim)
            cv2.circle(panel, (center_x, red_y), light_size, (50, 0, 0), -1)
            cv2.circle(panel, (center_x, yellow_y), light_size, (50, 50, 0), -1)
            cv2.circle(panel, (center_x, green_y), light_size, (0, 50, 0), -1)
            
            # Highlight active light
            if state.value == 'red':
                cv2.circle(panel, (center_x, red_y), light_size, self.colors['red'], -1)
            elif state.value == 'yellow':
                cv2.circle(panel, (center_x, yellow_y), light_size, self.colors['yellow'], -1)
            elif state.value == 'green':
                cv2.circle(panel, (center_x, green_y), light_size, self.colors['green'], -1)
            
            # Add traffic light label
            cv2.putText(panel, f"TL{i+1}", (lx - 10, ly - 5), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.4, (200, 200, 200), 1)
        
        # Add legend at bottom
        legend_y = panel_height - 30
        cv2.putText(panel, "Legend:", (10, legend_y), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.4, self.colors['text'], 1)
        
        # Pole legend
        cv2.rectangle(panel, (70, legend_y - 10), (80, legend_y + 5), self.colors['pole'], -1)
        cv2.putText(panel, "Poles", (85, legend_y), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.4, self.colors['text'], 1)
        
        # Traffic light legend
        cv2.rectangle(panel, (140, legend_y - 8), (150, legend_y + 3), (100, 100, 100), 2)
        cv2.putText(panel, "Traffic Lights", (155, legend_y), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.4, self.colors['text'], 1)
        
        return panel
    
    def _draw_pole_regions(self, image: np.ndarray, pole_regions: List[Tuple[int, int, int, int]]):
        """Draw detected pole regions on the image."""
        for i, (x, y, w, h) in enumerate(pole_regions):
            # Draw pole rectangle
            cv2.rectangle(image, (x, y), (x + w, y + h), self.colors['pole'], 2)
            
            # Draw pole label
            label = f"Pole {i+1}"
            (label_w, label_h), _ = cv2.getTextSize(label, self.font, self.font_scale, self.font_thickness)
            cv2.rectangle(image, (x, y - label_h - 10), (x + label_w + 10, y), self.colors['pole'], -1)
            cv2.putText(image, label, (x + 5, y - 5), self.font, self.font_scale, 
                       self.colors['text'], self.font_thickness)
            
            # Draw search expansion area (lighter color)
            expansion = self.detector.config.get('pole_expansion_factor', 1.3)
            exp_w = int(w * expansion)
            exp_h = int(h * expansion)
            exp_x = max(0, x - (exp_w - w) // 2)
            exp_y = max(0, y - (exp_h - h) // 2)
            
            cv2.rectangle(image, (exp_x, exp_y), (exp_x + exp_w, exp_y + exp_h), 
                         self.colors['pole_search'], 1)
    
    def _draw_traffic_lights(self, image: np.ndarray, detections: List[Dict]):
        """Draw traffic light detections with enhanced visualization."""
        for i, detection in enumerate(detections):
            x, y, w, h = detection['bbox']
            state = detection['state']
            confidence = detection['confidence']
            method = detection.get('method', 'unknown')
            
            # Choose color based on state
            if state == TrafficLightState.RED:
                color = self.colors['red']
            elif state == TrafficLightState.YELLOW:
                color = self.colors['yellow']
            elif state == TrafficLightState.GREEN:
                color = self.colors['green']
            else:
                color = self.colors['unknown']
            
            # Draw main bounding box
            thickness = 3 if method == 'two-stage' else 2
            cv2.rectangle(image, (x, y), (x + w, y + h), color, thickness)
            
            # Draw confidence bar
            bar_width = w
            bar_height = 8
            bar_x = x
            bar_y = y - bar_height - 5
            
            # Background bar
            cv2.rectangle(image, (bar_x, bar_y), (bar_x + bar_width, bar_y + bar_height), 
                         (50, 50, 50), -1)
            # Confidence bar
            conf_width = int(bar_width * confidence)
            cv2.rectangle(image, (bar_x, bar_y), (bar_x + conf_width, bar_y + bar_height), 
                         self.colors['confidence'], -1)
            
            # Create label with state, confidence, and method
            label = f"{state.value.upper()} {confidence:.2f}"
            method_indicator = "2S" if method == 'two-stage' else "1S"
            
            # Draw method indicator
            method_color = self.colors['method_two_stage'] if method == 'two-stage' else self.colors['method_single']
            cv2.circle(image, (x + w - 15, y + 15), 8, method_color, -1)
            cv2.putText(image, method_indicator, (x + w - 20, y + 20), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.4, self.colors['text'], 1)
            
            # Draw main label
            (label_w, label_h), _ = cv2.getTextSize(label, self.font, self.font_scale, self.font_thickness)
            label_bg_y = y + h + 5
            cv2.rectangle(image, (x, label_bg_y), (x + label_w + 10, label_bg_y + label_h + 10), 
                         self.colors['text_bg'], -1)
            cv2.putText(image, label, (x + 5, label_bg_y + label_h + 5), 
                       self.font, self.font_scale, color, self.font_thickness)
    
    def _draw_ui_overlay(self, image: np.ndarray, detection_info: Dict):
        """Draw performance and status information overlay."""
        h, w = image.shape[:2]
        
        # Performance panel
        panel_width = 300
        panel_height = 200
        panel_x = w - panel_width - 10
        panel_y = 10
        
        # Semi-transparent background
        overlay = image.copy()
        cv2.rectangle(overlay, (panel_x, panel_y), (panel_x + panel_width, panel_y + panel_height), 
                     (0, 0, 0), -1)
        cv2.addWeighted(overlay, 0.7, image, 0.3, 0, image)
        
        # Panel content
        y_offset = panel_y + 25
        line_height = 25
        
        # Title
        cv2.putText(image, "Detection Info", (panel_x + 10, y_offset), 
                   self.font, 0.7, self.colors['text'], 2)
        y_offset += line_height + 5
        
        # Method used
        method_color = self.colors['method_two_stage'] if detection_info['method_used'] == 'two-stage' else self.colors['method_single']
        cv2.putText(image, f"Method: {detection_info['method_used']}", 
                   (panel_x + 10, y_offset), self.font, 0.5, method_color, 1)
        y_offset += line_height
        
        # Processing time
        time_ms = detection_info['processing_time'] * 1000
        time_color = self.colors['green'] if time_ms < 50 else self.colors['yellow'] if time_ms < 100 else self.colors['red']
        cv2.putText(image, f"Time: {time_ms:.1f}ms", 
                   (panel_x + 10, y_offset), self.font, 0.5, time_color, 1)
        y_offset += line_height
        
        # Detections count
        det_count = len(detection_info['detections'])
        cv2.putText(image, f"Lights: {det_count}", 
                   (panel_x + 10, y_offset), self.font, 0.5, self.colors['text'], 1)
        y_offset += line_height
        
        # Poles found
        pole_count = len(detection_info['pole_regions'])
        if pole_count > 0:
            cv2.putText(image, f"Poles: {pole_count}", 
                       (panel_x + 10, y_offset), self.font, 0.5, self.colors['pole'], 1)
            y_offset += line_height
        
        # Average FPS
        if len(self.frame_times) > 1:
            avg_time = sum(self.frame_times) / len(self.frame_times)
            fps = 1.0 / avg_time if avg_time > 0 else 0
            cv2.putText(image, f"Avg FPS: {fps:.1f}", 
                       (panel_x + 10, y_offset), self.font, 0.5, self.colors['text'], 1)
        
        # Legend
        legend_y = h - 100
        cv2.putText(image, "Legend:", (10, legend_y), self.font, 0.6, self.colors['text'], 2)
        legend_y += 20
        
        # Pole legend
        cv2.rectangle(image, (10, legend_y - 10), (25, legend_y + 5), self.colors['pole'], 2)
        cv2.putText(image, "Poles", (35, legend_y), self.font, 0.5, self.colors['text'], 1)
        legend_y += 20
        
        # Method legend
        cv2.circle(image, (17, legend_y - 3), 8, self.colors['method_single'], -1)
        cv2.putText(image, "1S", (12, legend_y + 2), cv2.FONT_HERSHEY_SIMPLEX, 0.4, self.colors['text'], 1)
        cv2.putText(image, "Single-stage", (35, legend_y), self.font, 0.5, self.colors['text'], 1)
        
        cv2.circle(image, (150, legend_y - 3), 8, self.colors['method_two_stage'], -1)
        cv2.putText(image, "2S", (145, legend_y + 2), cv2.FONT_HERSHEY_SIMPLEX, 0.4, self.colors['text'], 1)
        cv2.putText(image, "Two-stage", (165, legend_y), self.font, 0.5, self.colors['text'], 1)

def process_camera_with_visualization(detector: TrafficLightDetector, show_poles: bool = True):
    """Process camera feed with enhanced visualization."""
    print("🎥 Starting Enhanced Traffic Light Visualization")
    print("Controls:")
    print("  'q' - Quit")
    print("  'p' - Toggle pole visualization")
    print("  's' - Save current frame")
    print("=" * 60)
    
    visualizer = TrafficLightVisualizer(detector)
    
    # Try different camera indices
    cap = None
    for camera_idx in [0, 1, 2]:
        cap = cv2.VideoCapture(camera_idx)
        if cap.isOpened():
            print(f"✅ Using camera {camera_idx}")
            break
    
    if not cap or not cap.isOpened():
        print("❌ Error: Could not open camera")
        return
    
    frame_count = 0
    
    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                print("❌ Error: Could not read frame from camera")
                break
            
            frame_count += 1
            
            # Process with visualization  
            vis_frame, detection_info = visualizer.visualize_detection_process(frame, show_poles)
            
            # Add frame counter to main image area
            cv2.putText(vis_frame, f"Frame: {frame_count}", (10, 30), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
            
            # Create combined view with schematic panel
            combined_frame = visualizer.create_combined_view(vis_frame, detection_info)
            
            # Show combined frame
            cv2.imshow("Traffic Light Detection Visualizer", combined_frame)
            
            # Handle key presses
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                break
            elif key == ord('p'):
                show_poles = not show_poles
                print(f"Pole visualization: {'ON' if show_poles else 'OFF'}")
            elif key == ord('s'):
                filename = f"visualization_frame_{frame_count}.jpg"
                cv2.imwrite(filename, vis_frame)
                print(f"📸 Saved frame to {filename}")
    
    finally:
        cap.release()
        cv2.destroyAllWindows()

def process_video_with_visualization(detector: TrafficLightDetector, video_path: str, output_path: str = None, show_poles: bool = True):
    """Process a video file with enhanced visualization including schematic panel."""
    # Import here to avoid circular imports
    from src.main import is_youtube_url, download_youtube_video
    
    print(f"🎥 Processing video with enhanced visualization: {video_path}")
    
    # Track if we downloaded from YouTube for cleanup
    is_youtube = is_youtube_url(video_path)
    downloaded_path = None
    
    # Check if it's a YouTube URL
    if is_youtube:
        try:
            # Download the video
            downloaded_path = download_youtube_video(video_path)
            video_path = downloaded_path
        except Exception as e:
            print(f"Failed to download YouTube video: {e}")
            return
    
    # Open video
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"Error: Could not open video {video_path}")
        return
    
    # Get video properties
    fps = int(cap.get(cv2.CAP_PROP_FPS))
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    
    print(f"Video info: {width}x{height}, {fps} FPS, {total_frames} frames")
    
    visualizer = TrafficLightVisualizer(detector)
    
    # Set up video writer if output path provided
    out = None
    if output_path:
        # Calculate new dimensions for combined view (image + schematic panel)
        panel_height = 300
        combined_height = height + panel_height
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out = cv2.VideoWriter(output_path, fourcc, fps, (width, combined_height))
    
    frame_count = 0
    skip_frames = detector.config.get('skip_frames', 1)
    
    print("Controls: 'q' to quit, 's' to save current frame, 'p' to toggle poles")
    
    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            
            frame_count += 1
            
            # Skip frames for performance if configured
            if frame_count % skip_frames == 0 or frame_count == 1:
                # Process with enhanced visualization
                vis_frame, detection_info = visualizer.visualize_detection_process(frame, show_poles)
                
                # Add frame counter
                cv2.putText(vis_frame, f"Frame: {frame_count}/{total_frames}", (10, 30), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
                
                # Create combined view with schematic panel
                combined_frame = visualizer.create_combined_view(vis_frame, detection_info)
                current_display = combined_frame
            else:
                # Use cached frame for skipped frames
                current_display = combined_frame if 'combined_frame' in locals() else frame
            
            print(f"Processing frame {frame_count}/{total_frames} (every {skip_frames})", end='\r')
            
            # Show frame
            cv2.imshow("Traffic Light Detection - Enhanced Video", current_display)
            
            # Save frame if output video specified
            if out and 'combined_frame' in locals():
                out.write(combined_frame)
            
            # Handle key presses
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                print("\nStopping video processing...")
                break
            elif key == ord('p'):
                show_poles = not show_poles
                print(f"\nPole visualization: {'ON' if show_poles else 'OFF'}")
            elif key == ord('s'):
                if 'combined_frame' in locals():
                    filename = f"enhanced_video_frame_{frame_count}.jpg"
                    cv2.imwrite(filename, combined_frame)
                    print(f"\n📸 Saved frame to {filename}")
    
    finally:
        cap.release()
        if out:
            out.release()
        cv2.destroyAllWindows()
        
        # Cleanup downloaded file
        if downloaded_path:
            try:
                import os
                os.remove(downloaded_path)
                print(f"Cleaned up temporary file: {downloaded_path}")
            except Exception as e:
                print(f"Note: Could not clean up temporary file: {e}")
    
    print(f"\n✅ Enhanced video processing completed!")
    if output_path:
        print(f"📹 Enhanced video saved to: {output_path}")

def process_image_with_visualization(detector: TrafficLightDetector, image_path: str, show_poles: bool = True):
    """Process a single image with enhanced visualization."""
    print(f"🖼️  Processing image: {image_path}")
    
    # Load image
    image = cv2.imread(image_path)
    if image is None:
        print(f"❌ Error: Could not load image {image_path}")
        return
    
    visualizer = TrafficLightVisualizer(detector)
    
    # Process with visualization
    vis_image, detection_info = visualizer.visualize_detection_process(image, show_poles)
    
    # Create combined view with schematic panel
    combined_image = visualizer.create_combined_view(vis_image, detection_info)
    
    # Display results
    print("\n📊 Detection Results:")
    print(f"Method: {detection_info['method_used']}")
    print(f"Processing time: {detection_info['processing_time']*1000:.1f}ms")
    print(f"Poles found: {len(detection_info['pole_regions'])}")
    print(f"Traffic lights: {len(detection_info['detections'])}")
    
    for i, det in enumerate(detection_info['detections']):
        print(f"  Light {i+1}: {det['state'].value} (confidence: {det['confidence']:.3f}, method: {det.get('method', 'unknown')})")
    
    # Save result
    output_path = image_path.replace('.', '_visualized.')
    cv2.imwrite(output_path, combined_image)
    print(f"✅ Visualization saved to: {output_path}")
    
    # Show combined image
    cv2.imshow("Traffic Light Detection Visualizer", combined_image)
    print("\nPress any key to close...")
    cv2.waitKey(0)
    cv2.destroyAllWindows()

def main():
    parser = argparse.ArgumentParser(description='Enhanced Traffic Light Detection Visualizer')
    parser.add_argument('--camera', action='store_true', help='Use camera input')
    parser.add_argument('--image', type=str, help='Path to input image')
    parser.add_argument('--config', type=str, default='config/detection_config.yaml', 
                       help='Path to detection config')
    parser.add_argument('--no-poles', action='store_true', help='Disable pole visualization')
    
    args = parser.parse_args()
    
    # Initialize detector
    print("🚦 Initializing Enhanced Traffic Light Detector...")
    detector = TrafficLightDetector(config_path=args.config, use_yolo=True)
    print("✅ Detector initialized")
    
    show_poles = not args.no_poles
    
    if args.camera:
        process_camera_with_visualization(detector, show_poles)
    elif args.image:
        process_image_with_visualization(detector, args.image, show_poles)
    else:
        print("Please specify either --camera or --image <path>")
        print("Example: python visualizer.py --camera")
        print("Example: python visualizer.py --image data/test.jpg")

if __name__ == "__main__":
    main()