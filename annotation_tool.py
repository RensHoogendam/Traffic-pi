#!/usr/bin/env python3
"""
Custom annotation tool for traffic light datasets.
Handles Python 3.14 compatibility and provides robust annotation workflow.
"""

import cv2
import json
import os
import sys
from pathlib import Path
import numpy as np
from typing import Dict, List, Tuple, Optional

class TrafficLightAnnotator:
    """Custom annotation tool for traffic lights."""
    
    def __init__(self, images_dir: str, labels_dir: str):
        """Initialize the annotator."""
        self.images_dir = Path(images_dir)
        self.labels_dir = Path(labels_dir)
        self.labels_dir.mkdir(exist_ok=True)
        
        # Get all image files
        self.image_extensions = ['.jpg', '.jpeg', '.png', '.bmp']
        self.image_files = []
        for ext in self.image_extensions:
            self.image_files.extend(self.images_dir.glob(f'*{ext}'))
            self.image_files.extend(self.images_dir.glob(f'*{ext.upper()}'))
        
        self.image_files = sorted(self.image_files)
        self.current_index = 0
        
        # Annotation state
        self.current_image = None
        self.current_annotations = []
        self.drawing = False
        self.start_point = None
        self.temp_rect = None
        
        # Colors for different traffic light states
        self.colors = {
            'red': (0, 0, 255),
            'yellow': (0, 255, 255),
            'green': (0, 255, 0),
            'unknown': (128, 128, 128)
        }
        
        self.current_class = 'red'
        
    def mouse_callback(self, event, x, y, flags, param):
        """Handle mouse events for drawing bounding boxes."""
        
        if event == cv2.EVENT_LBUTTONDOWN:
            self.drawing = True
            self.start_point = (x, y)
            self.temp_rect = None
            
        elif event == cv2.EVENT_MOUSEMOVE and self.drawing:
            # Update temporary rectangle
            self.temp_rect = (self.start_point[0], self.start_point[1], x, y)
            
        elif event == cv2.EVENT_LBUTTONUP:
            if self.drawing and self.start_point:
                # Finalize the annotation
                x1, y1 = self.start_point
                x2, y2 = x, y
                
                # Ensure proper rectangle (top-left to bottom-right)
                x1, x2 = min(x1, x2), max(x1, x2)
                y1, y2 = min(y1, y2), max(y1, y2)
                
                # Only add if rectangle has meaningful size
                if abs(x2 - x1) > 10 and abs(y2 - y1) > 10:
                    annotation = {
                        'bbox': [x1, y1, x2, y2],
                        'class': self.current_class,
                        'confidence': 1.0
                    }
                    self.current_annotations.append(annotation)
                    print(f"✅ Added {self.current_class} light: ({x1}, {y1}, {x2}, {y2})")
                
                self.drawing = False
                self.start_point = None
                self.temp_rect = None
    
    def draw_annotations(self, image: np.ndarray) -> np.ndarray:
        """Draw current annotations on the image."""
        
        display_image = image.copy()
        
        # Draw existing annotations
        for i, ann in enumerate(self.current_annotations):
            x1, y1, x2, y2 = ann['bbox']
            color = self.colors.get(ann['class'], (255, 255, 255))
            
            # Draw rectangle
            cv2.rectangle(display_image, (x1, y1), (x2, y2), color, 2)
            
            # Draw label
            label = f"{ann['class']} {i+1}"
            cv2.putText(display_image, label, (x1, y1-10), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
        
        # Draw temporary rectangle while drawing
        if self.temp_rect:
            x1, y1, x2, y2 = self.temp_rect
            color = self.colors.get(self.current_class, (255, 255, 255))
            cv2.rectangle(display_image, (x1, y1), (x2, y2), color, 2)
        
        return display_image
    
    def draw_ui(self, image: np.ndarray) -> np.ndarray:
        """Draw UI elements."""
        
        display_image = image.copy()
        h, w = display_image.shape[:2]
        
        # Draw status bar
        cv2.rectangle(display_image, (0, 0), (w, 80), (50, 50, 50), -1)
        
        # Current file info
        current_file = self.image_files[self.current_index].name if self.image_files else "No images"
        progress_text = f"Image {self.current_index + 1}/{len(self.image_files)}: {current_file}"
        cv2.putText(display_image, progress_text, (10, 25), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
        
        # Current class
        class_text = f"Current class: {self.current_class} (Press 1-4 to change)"
        cv2.putText(display_image, class_text, (10, 50), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, self.colors[self.current_class], 2)
        
        # Annotations count
        ann_text = f"Annotations: {len(self.current_annotations)} (Press 'u' to undo, 'c' to clear)"
        cv2.putText(display_image, ann_text, (10, 75), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (200, 200, 200), 2)
        
        # Controls at bottom
        controls = [
            "CONTROLS: Left-click drag = draw box | 1-4 = class | Space/Enter = next | Backspace = prev",
            "S = save | U = undo | C = clear all | Q = quit | D = delete mode"
        ]
        
        y_start = h - 50
        for i, control in enumerate(controls):
            cv2.putText(display_image, control, (10, y_start + i * 25), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (180, 180, 180), 1)
        
        return display_image
    
    def save_annotations(self):
        """Save current annotations to YOLO format."""
        
        if not self.image_files or self.current_index >= len(self.image_files):
            return
        
        image_file = self.image_files[self.current_index]
        label_file = self.labels_dir / f"{image_file.stem}.txt"
        
        # Get image dimensions for normalization
        h, w = self.current_image.shape[:2]
        
        # Class mapping
        class_map = {'red': 0, 'yellow': 1, 'green': 2, 'unknown': 3}
        
        with open(label_file, 'w') as f:
            for ann in self.current_annotations:
                x1, y1, x2, y2 = ann['bbox']
                class_id = class_map.get(ann['class'], 3)
                
                # Convert to YOLO format (center_x, center_y, width, height) normalized
                center_x = (x1 + x2) / 2 / w
                center_y = (y1 + y2) / 2 / h
                box_w = (x2 - x1) / w
                box_h = (y2 - y1) / h
                
                f.write(f"{class_id} {center_x:.6f} {center_y:.6f} {box_w:.6f} {box_h:.6f}\n")
        
        print(f"💾 Saved {len(self.current_annotations)} annotations to {label_file}")
    
    def load_annotations(self):
        """Load existing annotations if available."""
        
        if not self.image_files or self.current_index >= len(self.image_files):
            return
        
        image_file = self.image_files[self.current_index]
        label_file = self.labels_dir / f"{image_file.stem}.txt"
        
        self.current_annotations = []
        
        if label_file.exists():
            # Get image dimensions
            h, w = self.current_image.shape[:2]
            
            # Class mapping (reverse)
            class_map = {0: 'red', 1: 'yellow', 2: 'green', 3: 'unknown'}
            
            with open(label_file, 'r') as f:
                for line in f:
                    parts = line.strip().split()
                    if len(parts) == 5:
                        class_id = int(parts[0])
                        center_x, center_y, box_w, box_h = map(float, parts[1:])
                        
                        # Convert from YOLO format to pixel coordinates
                        x1 = int((center_x - box_w/2) * w)
                        y1 = int((center_y - box_h/2) * h)
                        x2 = int((center_x + box_w/2) * w)
                        y2 = int((center_y + box_h/2) * h)
                        
                        annotation = {
                            'bbox': [x1, y1, x2, y2],
                            'class': class_map.get(class_id, 'unknown'),
                            'confidence': 1.0
                        }
                        self.current_annotations.append(annotation)
            
            if self.current_annotations:
                print(f"📂 Loaded {len(self.current_annotations)} existing annotations")
    
    def load_current_image(self):
        """Load the current image and annotations."""
        
        if not self.image_files or self.current_index >= len(self.image_files):
            return False
        
        image_path = self.image_files[self.current_index]
        self.current_image = cv2.imread(str(image_path))
        
        if self.current_image is None:
            print(f"❌ Could not load image: {image_path}")
            return False
        
        # Load existing annotations
        self.load_annotations()
        
        print(f"📷 Loaded: {image_path.name} ({self.current_image.shape[1]}x{self.current_image.shape[0]})")
        return True
    
    def run(self):
        """Run the annotation tool."""
        
        if not self.image_files:
            print("❌ No images found in the directory!")
            return
        
        print(f"🏷️  Traffic Light Annotation Tool")
        print(f"📁 Images directory: {self.images_dir}")
        print(f"📁 Labels directory: {self.labels_dir}")
        print(f"📊 Found {len(self.image_files)} images")
        print("=" * 60)
        
        # Load first image
        if not self.load_current_image():
            return
        
        cv2.namedWindow("Traffic Light Annotator", cv2.WINDOW_NORMAL)
        cv2.setMouseCallback("Traffic Light Annotator", self.mouse_callback)
        
        delete_mode = False
        
        while True:
            if self.current_image is None:
                break
            
            # Create display image
            display_image = self.draw_annotations(self.current_image)
            display_image = self.draw_ui(display_image)
            
            # Add delete mode indicator
            if delete_mode:
                cv2.putText(display_image, "DELETE MODE - Click annotation to delete", 
                           (10, display_image.shape[0] - 80), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)
            
            cv2.imshow("Traffic Light Annotator", display_image)
            
            key = cv2.waitKey(1) & 0xFF
            
            if key == ord('q'):
                print("👋 Quitting annotator...")
                break
            
            elif key == ord('1'):
                self.current_class = 'red'
                print("🔴 Switched to RED class")
            
            elif key == ord('2'):
                self.current_class = 'yellow'
                print("🟡 Switched to YELLOW class")
            
            elif key == ord('3'):
                self.current_class = 'green'
                print("🟢 Switched to GREEN class")
            
            elif key == ord('4'):
                self.current_class = 'unknown'
                print("⚪ Switched to UNKNOWN class")
            
            elif key == ord('s'):
                self.save_annotations()
            
            elif key == ord('u') and self.current_annotations:
                removed = self.current_annotations.pop()
                print(f"↶ Undid {removed['class']} annotation")
            
            elif key == ord('c'):
                if self.current_annotations:
                    print(f"🗑️  Cleared {len(self.current_annotations)} annotations")
                    self.current_annotations = []
            
            elif key == ord('d'):
                delete_mode = not delete_mode
                print(f"🗑️  Delete mode: {'ON' if delete_mode else 'OFF'}")
            
            elif key in [ord(' '), 13]:  # Space or Enter
                # Save current and move to next
                self.save_annotations()
                self.current_index = min(self.current_index + 1, len(self.image_files) - 1)
                if not self.load_current_image():
                    break
            
            elif key == 8:  # Backspace
                # Save current and move to previous
                self.save_annotations()
                self.current_index = max(self.current_index - 1, 0)
                if not self.load_current_image():
                    break
            
            # Handle delete mode clicks
            if delete_mode and key == ord('x'):  # Click to delete
                # This would need mouse position tracking for click-to-delete
                pass
        
        # Save final annotations
        self.save_annotations()
        cv2.destroyAllWindows()
        
        print(f"✅ Annotation session completed!")
        print(f"📊 Processed {self.current_index + 1} images")

def main():
    """Main function with command line argument handling."""
    
    if len(sys.argv) < 2:
        print("Usage: python annotation_tool.py <images_directory> [labels_directory]")
        print("Example: python annotation_tool.py training_data/extracted_frames training_data/annotations")
        return
    
    images_dir = sys.argv[1]
    labels_dir = sys.argv[2] if len(sys.argv) > 2 else f"{images_dir}_labels"
    
    if not os.path.exists(images_dir):
        print(f"❌ Images directory not found: {images_dir}")
        return
    
    print(f"🚀 Starting Traffic Light Annotator...")
    annotator = TrafficLightAnnotator(images_dir, labels_dir)
    annotator.run()

if __name__ == "__main__":
    main()