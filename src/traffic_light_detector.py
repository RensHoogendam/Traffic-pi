"""
Traffic Light Detector

This module provides functionality to detect traffic lights in images using
computer vision techniques. It can identify traffic lights and determine their
current state (red, yellow/amber, green).

Author: Traffic-pi Project
"""

import cv2
import numpy as np
from typing import List, Tuple, Dict, Optional
from enum import Enum
import yaml
import os


class TrafficLightState(Enum):
    """Enumeration for traffic light states"""
    RED = "red"
    YELLOW = "yellow"
    GREEN = "green"
    UNKNOWN = "unknown"


class TrafficLightDetector:
    """
    A class for detecting traffic lights in images and determining their state.
    
    This detector uses color-based segmentation and shape analysis to identify
    traffic lights and classify their current state.
    """
    
    def __init__(self, config_path: Optional[str] = None):
        """
        Initialize the traffic light detector.
        
        Args:
            config_path: Path to configuration file. If None, uses default values.
        """
        self.config = self._load_config(config_path)
        
        # Color ranges in HSV for traffic light detection (more restrictive)
        self._color_ranges = {
            'red': [
                (np.array([0, 120, 70]), np.array([10, 255, 255])),    # Lower red range (higher saturation/value)
                (np.array([170, 120, 70]), np.array([180, 255, 255]))  # Upper red range (higher saturation/value)
            ],
            'yellow': [(np.array([15, 150, 150]), np.array([35, 255, 255]))],  # More restrictive yellow
            'green': [(np.array([45, 100, 100]), np.array([75, 255, 255]))]    # More restrictive green
        }
    
    def _load_config(self, config_path: Optional[str]) -> Dict:
        """Load configuration from file or use defaults"""
        default_config = {
            'min_contour_area': 100,
            'max_contour_area': 10000,
            'aspect_ratio_tolerance': 0.3,
            'circularity_threshold': 0.3,
            'brightness_threshold': 100
        }
        
        if config_path and os.path.exists(config_path):
            with open(config_path, 'r') as f:
                file_config = yaml.safe_load(f)
                default_config.update(file_config)
        
        return default_config
    
    def detect_traffic_lights(self, image: np.ndarray) -> List[Dict]:
        """
        Detect traffic lights in an image.
        
        Args:
            image: Input image as numpy array (BGR format)
            
        Returns:
            List of dictionaries containing detected traffic lights with their
            bounding boxes and states
        """
        # Convert BGR to HSV for better color detection
        hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
        
        detected_lights = []
        
        # Detect each color
        for color_name in ['red', 'yellow', 'green']:
            lights = self._detect_color_lights(hsv, image, color_name)
            detected_lights.extend(lights)
        
        # First filter for traffic light clusters
        cluster_filtered = self._detect_traffic_light_clusters(detected_lights)
        
        # Group nearby detections and determine final state
        grouped_lights = self._group_detections(cluster_filtered)
        
        return grouped_lights
    
    def _detect_color_lights(self, hsv_image: np.ndarray, original_image: np.ndarray, 
                           color_name: str) -> List[Dict]:
        """
        Detect lights of a specific color.
        
        Args:
            hsv_image: Image in HSV color space
            original_image: Original BGR image
            color_name: Name of the color to detect ('red', 'yellow', 'green')
            
        Returns:
            List of detected lights for this color
        """
        detections = []
        color_ranges = self._color_ranges[color_name]
        
        # Create mask for the color
        mask = np.zeros(hsv_image.shape[:2], dtype=np.uint8)
        
        for lower, upper in color_ranges:
            color_mask = cv2.inRange(hsv_image, lower, upper)
            mask = cv2.bitwise_or(mask, color_mask)
        
        # Apply morphological operations to clean up the mask
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
        
        # Additional noise reduction with Gaussian blur
        mask = cv2.GaussianBlur(mask, (3, 3), 0)
        
        # Find contours
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        for contour in contours:
            if self._is_valid_traffic_light_shape(contour):
                x, y, w, h = cv2.boundingRect(contour)
                
                # Calculate confidence based on shape and brightness
                confidence = self._calculate_confidence(mask[y:y+h, x:x+w], contour)
                
                if confidence > 0.5:  # Higher minimum confidence threshold
                    detections.append({
                        'bbox': (x, y, w, h),
                        'state': TrafficLightState[color_name.upper()],
                        'confidence': confidence,
                        'center': (x + w//2, y + h//2)
                    })
        
        return detections
    
    def _is_valid_traffic_light_shape(self, contour: np.ndarray) -> bool:
        """
        Check if a contour represents a valid traffic light shape.
        
        Args:
            contour: OpenCV contour
            
        Returns:
            True if the shape is likely a traffic light
        """
        area = cv2.contourArea(contour)
        
        # Check area constraints
        if area < self.config['min_contour_area'] or area > self.config['max_contour_area']:
            return False
        
        # Get bounding rectangle for aspect ratio check
        x, y, w, h = cv2.boundingRect(contour)
        aspect_ratio = w / h if h > 0 else 0
        
        # Traffic lights should be roughly circular (aspect ratio close to 1)
        if not (0.7 <= aspect_ratio <= 1.3):
            return False
        
        # Check if shape is roughly circular
        perimeter = cv2.arcLength(contour, True)
        if perimeter == 0:
            return False
            
        circularity = 4 * np.pi * area / (perimeter * perimeter)
        
        # More strict circularity for traffic lights
        if circularity < 0.5:
            return False
        
        # Additional check: traffic lights are usually quite bright
        # Check if the contour has sufficient brightness
        return self._check_brightness_uniformity(contour)
    
    def _calculate_confidence(self, mask_region: np.ndarray, contour: np.ndarray) -> float:
        """
        Calculate confidence score for a detection.
        
        Args:
            mask_region: Binary mask of the detected region
            contour: Contour of the detection
            
        Returns:
            Confidence score between 0 and 1
        """
        # Calculate fill ratio (how much of the bounding box is filled)
        fill_ratio = np.sum(mask_region > 0) / mask_region.size
        
        # Calculate circularity
        area = cv2.contourArea(contour)
        perimeter = cv2.arcLength(contour, True)
        if perimeter == 0:
            return 0.0
        circularity = 4 * np.pi * area / (perimeter * perimeter)
        
        # Combine metrics for confidence
        confidence = (fill_ratio * 0.6) + (circularity * 0.4)
        
        return min(confidence, 1.0)
    
    def _check_brightness_uniformity(self, contour: np.ndarray) -> bool:
        """
        Check if the contour area has uniform brightness (characteristic of traffic lights).
        
        Args:
            contour: OpenCV contour
            
        Returns:
            True if the area has uniform brightness
        """
        # For now, return True - this would need the original image
        # This is a placeholder for more sophisticated brightness analysis
        return True
    
    def _detect_traffic_light_clusters(self, detections: List[Dict]) -> List[Dict]:
        """
        Look for clusters of lights that form traffic light patterns.
        Traffic lights typically have 3 lights arranged vertically.
        
        Args:
            detections: Individual color detections
            
        Returns:
            Filtered detections that are likely part of traffic light clusters
        """
        if len(detections) < 1:
            return []
        
        # Group detections by proximity
        clusters = []
        used = set()
        
        for i, detection in enumerate(detections):
            if i in used:
                continue
                
            cluster = [detection]
            center_i = detection['center']
            
            # Find nearby detections
            for j, other_detection in enumerate(detections[i+1:], i+1):
                if j in used:
                    continue
                    
                center_j = other_detection['center']
                distance = np.sqrt(
                    (center_i[0] - center_j[0])**2 + 
                    (center_i[1] - center_j[1])**2
                )
                
                # If detections are close (within 150 pixels), they might be part of same traffic light
                if distance < 150:
                    cluster.append(other_detection)
                    used.add(j)
            
            used.add(i)
            clusters.append(cluster)
        
        # Filter clusters - prefer clusters with multiple lights or single very confident lights
        filtered_detections = []
        
        for cluster in clusters:
            if len(cluster) >= 2:  # Multiple lights suggest a traffic light
                # Add all lights in multi-light clusters
                filtered_detections.extend(cluster)
            elif len(cluster) == 1:
                # For single lights, require higher confidence
                detection = cluster[0]
                if detection['confidence'] > 0.7:  # Higher threshold for single lights
                    filtered_detections.append(detection)
        
        return filtered_detections
    
    def _group_detections(self, detections: List[Dict]) -> List[Dict]:
        """
        Group nearby detections and determine the final traffic light state.
        
        Args:
            detections: List of individual color detections
            
        Returns:
            List of grouped traffic light detections
        """
        if not detections:
            return []
        
        # Sort by confidence
        detections.sort(key=lambda x: x['confidence'], reverse=True)
        
        grouped = []
        used_detections = set()
        
        for i, detection in enumerate(detections):
            if i in used_detections:
                continue
            
            group = [detection]
            used_detections.add(i)
            
            # Find nearby detections
            for j, other_detection in enumerate(detections[i+1:], i+1):
                if j in used_detections:
                    continue
                
                # Check if detections are close enough to be the same traffic light
                distance = np.sqrt(
                    (detection['center'][0] - other_detection['center'][0])**2 +
                    (detection['center'][1] - other_detection['center'][1])**2
                )
                
                if distance < 100:  # Adjust this threshold as needed
                    group.append(other_detection)
                    used_detections.add(j)
            
            # Determine the best detection in the group (highest confidence)
            best_detection = max(group, key=lambda x: x['confidence'])
            grouped.append(best_detection)
        
        return grouped
    
    def draw_detections(self, image: np.ndarray, detections: List[Dict]) -> np.ndarray:
        """
        Draw detected traffic lights on the image.
        
        Args:
            image: Input image
            detections: List of detections from detect_traffic_lights
            
        Returns:
            Image with drawn detections
        """
        result = image.copy()
        
        # Color mapping for different states
        color_map = {
            TrafficLightState.RED: (0, 0, 255),
            TrafficLightState.YELLOW: (0, 255, 255),
            TrafficLightState.GREEN: (0, 255, 0),
            TrafficLightState.UNKNOWN: (128, 128, 128)
        }
        
        for detection in detections:
            x, y, w, h = detection['bbox']
            state = detection['state']
            confidence = detection['confidence']
            
            # Draw bounding box
            color = color_map.get(state, (128, 128, 128))
            cv2.rectangle(result, (x, y), (x + w, y + h), color, 2)
            
            # Draw label
            label = f"{state.value}: {confidence:.2f}"
            cv2.putText(result, label, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 
                       0.6, color, 2)
        
        return result


def main():
    """Example usage of the TrafficLightDetector"""
    detector = TrafficLightDetector()
    
    # Example with a test image (you'll need to provide your own image)
    test_image_path = "data/test_traffic_light.jpg"
    
    if os.path.exists(test_image_path):
        image = cv2.imread(test_image_path)
        detections = detector.detect_traffic_lights(image)
        
        print(f"Detected {len(detections)} traffic lights:")
        for i, detection in enumerate(detections):
            print(f"  {i+1}. State: {detection['state'].value}, "
                  f"Confidence: {detection['confidence']:.2f}, "
                  f"BBox: {detection['bbox']}")
        
        # Display result
        result_image = detector.draw_detections(image, detections)
        cv2.imshow("Traffic Light Detection", result_image)
        cv2.waitKey(0)
        cv2.destroyAllWindows()
    else:
        print(f"Test image not found at {test_image_path}")
        print("Please add a test image to the data/ directory")


if __name__ == "__main__":
    main()