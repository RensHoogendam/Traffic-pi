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
from ultralytics import YOLO
import torch


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
    
    def __init__(self, config_path: Optional[str] = None, use_yolo: bool = True):
        """
        Initialize the traffic light detector.
        
        Args:
            config_path: Path to configuration file. If None, uses default values.
            use_yolo: Whether to use YOLO model for detection. If False, falls back to color-based detection.
        """
        self.config = self._load_config(config_path)
        self.use_yolo = use_yolo
        self.debug = self.config.get('show_debug_info', False)
        
        # Check config for YOLO preference
        if use_yolo and self.config.get('use_yolo', True):
            self.use_yolo = True
        else:
            self.use_yolo = False
        
        # Initialize YOLO model if requested
        self.yolo_model = None
        if self.use_yolo:
            try:
                model_name = self.config.get('yolo_model', 'yolov8s.pt')  # Use small model by default for better accuracy
                self.yolo_model = YOLO(model_name)  # Will download automatically if not present
                print(f"✅ YOLO model ({model_name}) loaded successfully")
                
                # Warm up the model for consistent performance
                dummy_img = np.zeros((640, 640, 3), dtype=np.uint8)
                _ = self.yolo_model(dummy_img, verbose=False)
                print("✅ Model warmed up")
            except Exception as e:
                print(f"⚠️ Could not load YOLO model: {e}")
                print("Falling back to color-based detection")
                self.use_yolo = False
        
        # Color ranges in HSV for both YOLO and color-based detection
        # Very permissive for YOLO state classification
        self._color_ranges = {
            'red': [
                (np.array([0, 30, 50]), np.array([15, 255, 255])),    # Very wide lower red range 
                (np.array([165, 30, 50]), np.array([180, 255, 255]))  # Very wide upper red range
            ],
            'yellow': [(np.array([10, 30, 80]), np.array([50, 255, 255]))],   # Very wide yellow range
            'green': [(np.array([35, 30, 50]), np.array([90, 255, 255]))]      # Very wide green range
        }
        
        # More restrictive ranges for color-only detection (when not using YOLO)
        self._strict_color_ranges = {
            'red': [
                (np.array([0, 120, 70]), np.array([10, 255, 255])),    # Strict red
                (np.array([170, 120, 70]), np.array([180, 255, 255])) 
            ],
            'yellow': [(np.array([15, 150, 150]), np.array([35, 255, 255]))],  # Strict yellow
            'green': [(np.array([45, 100, 100]), np.array([75, 255, 255]))]    # Strict green
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
        if self.use_yolo and self.yolo_model is not None:
            return self._detect_with_yolo(image)
        else:
            return self._detect_with_color(image)
    
    def _detect_with_yolo(self, image: np.ndarray) -> List[Dict]:
        """
        Use YOLO model to detect traffic lights.
        Two-stage approach: detect poles first, then search for lights within pole regions.
        
        Args:
            image: Input image as numpy array (BGR format)
            
        Returns:
            List of detected traffic lights with enhanced state classification
        """
        # Smart detection strategy selection
        use_two_stage = self.config.get('enable_pole_detection', False)
        
        # Auto pole detection: try single-stage first, then decide
        if not use_two_stage and self.config.get('auto_pole_detection', True):
            if self.debug:
                print("Auto-detection: trying single-stage first")
            
            # Quick single-stage detection with normal confidence
            quick_results = self._single_stage_yolo_detection(image)
            
            # Decide based on results
            num_detections = len(quick_results)
            complexity_threshold = self.config.get('scene_complexity_threshold', 3)
            max_detections = self.config.get('max_initial_detections', 10)
            
            if num_detections == 0:
                # No detections - try two-stage with lower confidence
                if self.debug:
                    print(f"No detections found, trying two-stage detection")
                use_two_stage = True
            elif num_detections > max_detections:
                # Too many detections - use pole detection to filter
                if self.debug:
                    print(f"Too many detections ({num_detections}), using pole filtering")
                use_two_stage = True
            else:
                # Good number of detections - use single-stage results
                if self.debug:
                    print(f"Single-stage successful ({num_detections} detections)")
                return quick_results
        
        if use_two_stage:
            if self.debug:
                print("Using two-stage detection (poles first)")
            return self._two_stage_detection(image)
        else:
            if self.debug:
                print("Using single-stage detection")
            return self._single_stage_yolo_detection(image)
    
    def _single_stage_yolo_detection(self, image: np.ndarray) -> List[Dict]:
        """
        Original single-stage YOLO detection for traffic lights.
        """
        # Preprocess image for better distant detection
        processed_image = self._preprocess_for_distant_detection(image)
        
        # Run YOLO inference with parameters optimized for distant detection
        yolo_conf = self._get_adaptive_confidence(image)
        yolo_classes = self.config.get('yolo_classes', [9])
        yolo_imgsz = self.config.get('yolo_imgsz', 640)
        yolo_augment = self.config.get('yolo_augment', True)
        
        # Try multi-scale detection if enabled
        if self.config.get('enable_multiscale', False):
            results = self._multiscale_yolo_detection(image, yolo_conf, yolo_classes, yolo_imgsz, yolo_augment)
        else:
            # Conditionally preprocess image based on config
            if self.config.get('enable_preprocessing', False):
                processed_image = self._preprocess_for_distant_detection(image)
            else:
                processed_image = image
            results = self.yolo_model(processed_image, conf=yolo_conf, classes=yolo_classes, imgsz=yolo_imgsz, augment=yolo_augment)
        
        detected_lights = []
        
        for result in results:
            boxes = result.boxes
            if boxes is not None:
                for box in boxes:
                    # Get bounding box coordinates
                    x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
                    conf = box.conf[0].cpu().numpy()
                    
                    # Convert to our bbox format (x, y, width, height)
                    x, y, w, h = int(x1), int(y1), int(x2 - x1), int(y2 - y1)
                    
                    print(f"YOLO detected traffic light: bbox=({x}, {y}, {w}, {h}), conf={conf:.3f}")
                    
                    # Extract the traffic light region for color analysis
                    traffic_light_roi = image[y:y+h, x:x+w]
                    
                    if traffic_light_roi.size > 0:
                        # Save ROI for debugging (optional)
                        # cv2.imwrite(f"debug_roi_{len(detected_lights)}.jpg", traffic_light_roi)
                        
                        # Determine the state using color analysis on the detected traffic light
                        print(f"  Analyzing ROI of size {traffic_light_roi.shape}")
                        state = self._classify_traffic_light_state(traffic_light_roi)
                        
                        detected_lights.append({
                            'bbox': (x, y, w, h),
                            'state': state,
                            'confidence': float(conf),
                            'center': (x + w//2, y + h//2),
                            'method': 'yolo'
                        })
        
        return detected_lights
    
    def _two_stage_detection(self, image: np.ndarray) -> List[Dict]:
        """
        Two-stage detection: first find poles/structures, then search for traffic lights within those areas.
        """
        detected_lights = []
        
        # Stage 1: Detect pole-like structures and vertical objects
        pole_regions = self._detect_pole_structures(image)
        
        if self.debug:
            print(f"Found {len(pole_regions)} potential pole regions")
        
        if not pole_regions:
            # If no poles found, fall back to full image detection
            if self.debug:
                print("No poles found, falling back to full image detection")
            return self._single_stage_yolo_detection(image)
        
        # Stage 2: Search for traffic lights within pole regions
        for i, pole_region in enumerate(pole_regions):
            x, y, w, h = pole_region
            
            # Expand the pole region to catch traffic lights attached to or near the pole
            expansion = self.config.get('pole_expansion_factor', 1.5)
            expanded_w = int(w * expansion)
            expanded_h = int(h * expansion)
            expanded_x = max(0, x - (expanded_w - w) // 2)
            expanded_y = max(0, y - (expanded_h - h) // 2)
            
            # Ensure we don't go out of image bounds
            expanded_w = min(expanded_w, image.shape[1] - expanded_x)
            expanded_h = min(expanded_h, image.shape[0] - expanded_y)
            
            # Extract the region around the pole
            pole_roi = image[expanded_y:expanded_y+expanded_h, expanded_x:expanded_x+expanded_w]
            
            if pole_roi.size == 0:
                continue
                
            if self.debug:
                print(f"Searching pole region {i}: ({expanded_x}, {expanded_y}, {expanded_w}, {expanded_h})")
            
            # Detect traffic lights in this region with higher sensitivity
            region_lights = self._detect_lights_in_region(pole_roi, expanded_x, expanded_y)
            
            # Adjust coordinates back to full image
            for light in region_lights:
                light['bbox'] = (
                    light['bbox'][0] + expanded_x,
                    light['bbox'][1] + expanded_y,
                    light['bbox'][2],
                    light['bbox'][3]
                )
                light['center'] = (
                    light['center'][0] + expanded_x,
                    light['center'][1] + expanded_y
                )
                light['method'] = 'two-stage'
                
            detected_lights.extend(region_lights)
        
        return detected_lights
    
    def _detect_pole_structures(self, image: np.ndarray) -> List[Tuple[int, int, int, int]]:
        """
        Detect vertical pole-like structures that might support traffic lights.
        Uses both YOLO detection and contour analysis.
        """
        pole_regions = []
        
        # Method 1: Use YOLO to detect pole-like objects (stop signs, fire hydrants, etc.)
        try:
            # Look for pole-like objects with lower confidence to catch more candidates
            pole_classes = self.config.get('pole_classes', [10, 11])  # fire hydrant, stop sign
            results = self.yolo_model(image, conf=0.1, classes=pole_classes, verbose=False)
            
            for result in results:
                if result.boxes is not None:
                    for box in result.boxes:
                        x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
                        pole_regions.append((int(x1), int(y1), int(x2-x1), int(y2-y1)))
        except Exception as e:
            if self.debug:
                print(f"YOLO pole detection failed: {e}")
        
        # Method 2: Fast contour-based vertical structure detection (only if needed)
        if len(pole_regions) < 2:  # Only do contour analysis if YOLO didn't find enough
            # Resize image for faster processing
            height, width = image.shape[:2]
            scale = min(1.0, 400 / max(width, height))  # Max 400px for speed
            
            if scale < 1.0:
                small_img = cv2.resize(image, (int(width * scale), int(height * scale)))
            else:
                small_img = image
            
            gray = cv2.cvtColor(small_img, cv2.COLOR_BGR2GRAY)
            edges = cv2.Canny(gray, 50, 150)
            
            # Find contours
            contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            
            min_height = self.config.get('min_pole_height', 100) * scale
            max_width = self.config.get('max_pole_width', 100) * scale
            
            for contour in contours:
                x, y, w, h = cv2.boundingRect(contour)
                
                # Check if this looks like a vertical pole
                aspect_ratio = h / w if w > 0 else 0
                
                # Poles should be tall and narrow
                if (h >= min_height and w <= max_width and 
                    aspect_ratio >= 3 and h < small_img.shape[0] * 0.8):
                    # Scale back to original coordinates
                    if scale < 1.0:
                        x, y, w, h = int(x / scale), int(y / scale), int(w / scale), int(h / scale)
                    pole_regions.append((x, y, w, h))
        
        # Remove overlapping regions and limit count for performance
        pole_regions = self._remove_overlapping_regions(pole_regions)
        
        # Limit number of pole regions for performance
        max_regions = self.config.get('max_pole_regions', 3)
        if len(pole_regions) > max_regions:
            # Keep the tallest/most promising regions
            pole_regions = sorted(pole_regions, key=lambda r: r[3], reverse=True)[:max_regions]
            if self.debug:
                print(f"Limited to {max_regions} pole regions for performance")
        
        return pole_regions
    
    def _detect_lights_in_region(self, region: np.ndarray, offset_x: int, offset_y: int) -> List[Dict]:
        """
        Detect traffic lights within a specific region with higher sensitivity.
        """
        # Use lower confidence for regional detection since we're in a promising area
        region_conf = max(0.1, self.config.get('yolo_confidence', 0.25) * 0.7)
        
        results = self.yolo_model(region, 
                                 conf=region_conf,
                                 classes=[9],  # Traffic light only
                                 verbose=False)
        
        detected_lights = []
        
        for result in results:
            if result.boxes is not None:
                for box in result.boxes:
                    x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
                    conf = box.conf[0].cpu().numpy()
                    
                    x, y, w, h = int(x1), int(y1), int(x2 - x1), int(y2 - y1)
                    
                    # Extract ROI for color analysis (coordinates relative to region)
                    traffic_light_roi = region[y:y+h, x:x+w]
                    
                    if traffic_light_roi.size > 0:
                        state = self._classify_traffic_light_state(traffic_light_roi)
                        
                        detected_lights.append({
                            'bbox': (x, y, w, h),  # Will be adjusted by caller
                            'state': state,
                            'confidence': float(conf),
                            'center': (x + w//2, y + h//2),  # Will be adjusted by caller
                            'method': 'regional'
                        })
        
        return detected_lights
    
    def _remove_overlapping_regions(self, regions: List[Tuple[int, int, int, int]]) -> List[Tuple[int, int, int, int]]:
        """
        Remove overlapping bounding box regions, keeping the larger ones.
        """
        if not regions:
            return []
        
        # Sort by area (largest first)
        regions = sorted(regions, key=lambda r: r[2] * r[3], reverse=True)
        
        filtered = []
        for region in regions:
            x1, y1, w1, h1 = region
            
            # Check if this region overlaps significantly with any already kept region
            overlaps = False
            for kept in filtered:
                x2, y2, w2, h2 = kept
                
                # Calculate intersection
                ix1 = max(x1, x2)
                iy1 = max(y1, y2)
                ix2 = min(x1 + w1, x2 + w2)
                iy2 = min(y1 + h1, y2 + h2)
                
                if ix1 < ix2 and iy1 < iy2:
                    intersection_area = (ix2 - ix1) * (iy2 - iy1)
                    region_area = w1 * h1
                    
                    # If more than 30% overlap, consider it duplicate
                    if intersection_area / region_area > 0.3:
                        overlaps = True
                        break
            
            if not overlaps:
                filtered.append(region)
        
        return filtered
    
    def _get_adaptive_confidence(self, image: np.ndarray) -> float:
        """
        Calculate adaptive confidence threshold based on image characteristics.
        """
        if not self.config.get('adaptive_confidence', False):
            return self.config.get('yolo_confidence', 0.25)
        
        # Image size factor - larger images can use lower confidence
        height, width = image.shape[:2]
        image_size = width * height
        size_threshold = self.config.get('image_size_threshold', 800) ** 2
        
        min_conf = self.config.get('min_yolo_confidence', 0.1)
        max_conf = self.config.get('max_yolo_confidence', 0.3)
        
        if image_size > size_threshold:
            # Large image - use lower confidence for distant objects
            confidence = min_conf + (max_conf - min_conf) * 0.3
        else:
            # Small image - use higher confidence to reduce noise
            confidence = min_conf + (max_conf - min_conf) * 0.7
        
        return confidence
    
    def _detect_with_color(self, image: np.ndarray) -> List[Dict]:
        """
        Fallback color-based detection method.
        
        Args:
            image: Input image as numpy array (BGR format)
            
        Returns:
            List of detected traffic lights using color segmentation
        """
        # Convert BGR to HSV for better color detection
        hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
        
        detected_lights = []
        
        # Detect each color using strict ranges for color-only detection
        for color_name in ['red', 'yellow', 'green']:
            lights = self._detect_color_lights(hsv, image, color_name, use_strict=True)
            detected_lights.extend(lights)
        
        # First filter for traffic light clusters
        cluster_filtered = self._detect_traffic_light_clusters(detected_lights)
        
        # Group nearby detections and determine final state
        grouped_lights = self._group_detections(cluster_filtered)
        
        return grouped_lights
    
    def _classify_traffic_light_state(self, roi: np.ndarray) -> TrafficLightState:
        """
        Classify the state of a detected traffic light using color analysis.
        
        Args:
            roi: Region of interest containing the traffic light
            
        Returns:
            TrafficLightState enum value
        """
        if roi.size == 0:
            return TrafficLightState.UNKNOWN
        
        # Convert to HSV for better color analysis
        hsv_roi = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)
        
        # First, try analyzing the entire ROI for any bright colored regions
        full_red_score = self._analyze_section_for_color(hsv_roi, 'red')
        full_yellow_score = self._analyze_section_for_color(hsv_roi, 'yellow')
        full_green_score = self._analyze_section_for_color(hsv_roi, 'green')
        
        # Also try sectional analysis (top=red, middle=yellow, bottom=green)
        h, w = hsv_roi.shape[:2]
        if h >= 9:  # Only do sectional analysis if ROI is tall enough
            third_h = max(h // 3, 3)  # Ensure minimum section size
            
            # Analyze each section with some overlap to avoid missing lights at boundaries
            top_section = hsv_roi[0:third_h+2, :] if h > third_h+2 else hsv_roi[0:third_h, :]
            middle_section = hsv_roi[max(0, third_h-2):min(h, 2*third_h+2), :] 
            bottom_section = hsv_roi[max(0, 2*third_h-2):h, :]
            
            section_red_score = self._analyze_section_for_color(top_section, 'red')
            section_yellow_score = self._analyze_section_for_color(middle_section, 'yellow')
            section_green_score = self._analyze_section_for_color(bottom_section, 'green')
            
            # Use the higher of full ROI or sectional analysis
            red_score = max(full_red_score, section_red_score)
            yellow_score = max(full_yellow_score, section_yellow_score)
            green_score = max(full_green_score, section_green_score)
        else:
            # ROI too small for sectional analysis, use full ROI scores
            red_score = full_red_score
            yellow_score = full_yellow_score
            green_score = full_green_score
        
        # Determine the most likely state
        scores = {
            TrafficLightState.RED: red_score,
            TrafficLightState.YELLOW: yellow_score,
            TrafficLightState.GREEN: green_score
        }
        
        # Return the state with the highest score, or UNKNOWN if all scores are too low
        max_state = max(scores, key=scores.get)
        max_score = scores[max_state]
        
        # Try brightness-based detection if color analysis fails
        if max_score <= 0.1:
            print(f"  Trying brightness-based detection...")
            brightness_state = self._detect_by_brightness(hsv_roi)
            if brightness_state != TrafficLightState.UNKNOWN:
                print(f"  Brightness analysis: {brightness_state.value}")
                return brightness_state
        
        # Lower threshold and add debug info
        if max_score > 0.05:  # Even lower minimum threshold
            print(f"  Color analysis: Red={red_score:.3f}, Yellow={yellow_score:.3f}, Green={green_score:.3f} -> {max_state.value}")
            return max_state
        else:
            print(f"  Color analysis: All scores too low (R={red_score:.3f}, Y={yellow_score:.3f}, G={green_score:.3f})")
            return TrafficLightState.UNKNOWN
    
    def _detect_by_brightness(self, hsv_roi: np.ndarray) -> TrafficLightState:
        """
        Fallback detection based on brightest regions in the ROI.
        """
        # Find the brightest pixels
        v_channel = hsv_roi[:, :, 2]
        brightness_threshold = np.percentile(v_channel, 90)  # Top 10% brightest pixels
        
        if brightness_threshold < 100:  # If nothing is bright enough
            return TrafficLightState.UNKNOWN
        
        bright_mask = v_channel >= brightness_threshold
        bright_pixels = hsv_roi[bright_mask]
        
        if len(bright_pixels) < 10:  # Need at least some bright pixels
            return TrafficLightState.UNKNOWN
        
        # Analyze hue of bright pixels
        hues = bright_pixels[:, 0]
        
        # Count pixels in each color range
        red_count = np.sum((hues <= 15) | (hues >= 165))
        yellow_count = np.sum((hues >= 15) & (hues <= 45))  
        green_count = np.sum((hues >= 45) & (hues <= 90))
        
        total_bright = len(bright_pixels)
        red_ratio = red_count / total_bright
        yellow_ratio = yellow_count / total_bright
        green_ratio = green_count / total_bright
        
        print(f"    Brightness analysis: R={red_ratio:.3f}, Y={yellow_ratio:.3f}, G={green_ratio:.3f} (threshold={brightness_threshold})")
        
        # Return the color with highest ratio if above threshold
        if red_ratio > 0.3:
            return TrafficLightState.RED
        elif yellow_ratio > 0.3:
            return TrafficLightState.YELLOW
        elif green_ratio > 0.3:
            return TrafficLightState.GREEN
            
        return TrafficLightState.UNKNOWN
    
    def _preprocess_for_distant_detection(self, image: np.ndarray) -> np.ndarray:
        """
        Preprocess image to enhance detection of distant traffic lights.
        
        Args:
            image: Input image in BGR format
            
        Returns:
            Preprocessed image optimized for distant object detection
        """
        # Fast mode: just apply CLAHE without color space conversion
        if self.config.get('fast_mode', True):
            # Convert to grayscale, apply CLAHE, then convert back to BGR
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
            enhanced_gray = clahe.apply(gray)
            # Convert back to BGR by duplicating the enhanced channel
            enhanced = cv2.cvtColor(enhanced_gray, cv2.COLOR_GRAY2BGR)
            # Blend with original to preserve color information
            return cv2.addWeighted(image, 0.7, enhanced, 0.3, 0)
        
        # Full preprocessing (slower but better quality)
        # Convert to LAB color space for better illumination handling
        lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
        l, a, b = cv2.split(lab)
        
        # Apply CLAHE (Contrast Limited Adaptive Histogram Equalization) to L channel
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        l_enhanced = clahe.apply(l)
        
        # Merge channels back
        lab_enhanced = cv2.merge([l_enhanced, a, b])
        enhanced = cv2.cvtColor(lab_enhanced, cv2.COLOR_LAB2BGR)
        
        # Apply slight sharpening to enhance edges
        kernel = np.array([[-1,-1,-1],
                          [-1, 9,-1],
                          [-1,-1,-1]])
        sharpened = cv2.filter2D(enhanced, -1, kernel)
        
        # Blend original and sharpened (70% enhanced, 30% original to avoid over-sharpening)
        result = cv2.addWeighted(enhanced, 0.7, sharpened, 0.3, 0)
        
        return result

    def _get_adaptive_confidence(self, image: np.ndarray) -> float:
        """
        Calculate adaptive confidence threshold based on image characteristics.
        
        Args:
            image: Input image
            
        Returns:
            Adjusted confidence threshold
        """
        if not self.config.get('adaptive_confidence', False):
            return self.config.get('yolo_confidence', 0.15)
        
        # Get image dimensions
        height, width = image.shape[:2]
        image_area = height * width
        
        # Calculate image quality metrics
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        
        # Measure image sharpness using Laplacian variance
        laplacian_var = cv2.Laplacian(gray, cv2.CV_64F).var()
        
        # Measure brightness
        mean_brightness = np.mean(gray)
        
        # Base confidence from config
        base_conf = self.config.get('yolo_confidence', 0.15)
        min_conf = self.config.get('min_yolo_confidence', 0.1)
        max_conf = self.config.get('max_yolo_confidence', 0.3)
        size_threshold = self.config.get('image_size_threshold', 800)
        
        # Adjust confidence based on image size (larger images can use lower confidence)
        if width > size_threshold or height > size_threshold:
            # Large image - can use lower confidence for distant objects
            size_factor = 0.8
        else:
            # Small image - need higher confidence to reduce false positives
            size_factor = 1.2
        
        # Adjust for image quality
        # Sharp, bright images can use lower confidence
        if laplacian_var > 100 and mean_brightness > 100:
            quality_factor = 0.9
        elif laplacian_var < 50 or mean_brightness < 80:
            quality_factor = 1.3
        else:
            quality_factor = 1.0
        
        # Calculate final confidence
        adaptive_conf = base_conf * size_factor * quality_factor
        
        # Clamp to min/max bounds
        adaptive_conf = max(min_conf, min(max_conf, adaptive_conf))
        
        if self.debug:
            print(f"Adaptive confidence: {adaptive_conf:.3f} (base={base_conf:.3f}, size_factor={size_factor:.2f}, quality_factor={quality_factor:.2f})")
        
        return adaptive_conf

    def _multiscale_yolo_detection(self, image: np.ndarray, conf: float, classes: list, imgsz: int, augment: bool):
        """
        Run YOLO detection at multiple scales to catch distant traffic lights.
        Returns detection results in the same format as regular YOLO detection.
        """
        scale_factors = self.config.get('scale_factors', [0.8, 1.0, 1.2])
        all_detections = []
        
        original_h, original_w = image.shape[:2]
        
        for scale in scale_factors:
            if scale != 1.0:
                # Resize image
                new_w = int(original_w * scale)
                new_h = int(original_h * scale)
                scaled_image = cv2.resize(image, (new_w, new_h), interpolation=cv2.INTER_LINEAR)
            else:
                scaled_image = image
            
            # Conditionally apply preprocessing for distant detection
            if self.config.get('enable_preprocessing', False):
                processed_image = self._preprocess_for_distant_detection(scaled_image)
            else:
                processed_image = scaled_image
            
            # Run detection on processed image
            results = self.yolo_model(processed_image, conf=conf, classes=classes, imgsz=imgsz, augment=augment)
            
            # Extract detection data and scale back to original coordinates
            for result in results:
                if result.boxes is not None:
                    for box in result.boxes:
                        x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
                        
                        # Scale coordinates back to original image size
                        if scale != 1.0:
                            x1, x2 = x1 / scale, x2 / scale
                            y1, y2 = y1 / scale, y2 / scale
                        
                        confidence = box.conf[0].cpu().item()
                        class_id = int(box.cls[0])
                        
                        # Only keep traffic light detections (class 9 in COCO)
                        if class_id == 9:
                            all_detections.append({
                                'bbox': [x1, y1, x2, y2],
                                'confidence': confidence,
                                'scale': scale
                            })
        
        # Convert back to YOLO-like format for compatibility
        # Create a mock results object with the best detections
        if all_detections:
            # Sort by confidence and take the best ones
            all_detections.sort(key=lambda x: x['confidence'], reverse=True)
            
            # For compatibility with the existing detection processing,
            # we'll return the regular YOLO results from scale=1.0
            # The multiscale data will be used for improved detection
            for scale in scale_factors:
                if scale == 1.0:
                    processed_image = self._preprocess_for_distant_detection(image)
                    results = self.yolo_model(processed_image, conf=conf, classes=classes, imgsz=imgsz, augment=augment)
                    return results
        
        # If no detections found, return empty results
        return self.yolo_model(image, conf=conf, classes=classes, imgsz=imgsz, augment=augment)
    
    def _analyze_section_for_color(self, section: np.ndarray, color: str) -> float:
        """
        Analyze a section of the traffic light for a specific color.
        
        Args:
            section: HSV image section to analyze
            color: Color name to look for ('red', 'yellow', 'green')
            
        Returns:
            Score indicating how likely this section contains the specified color
        """
        if section.size == 0:
            return 0.0
        
        # Get color ranges for the specified color (use permissive ranges for YOLO analysis)
        color_ranges = self._color_ranges.get(color, [])
        
        total_mask = np.zeros(section.shape[:2], dtype=np.uint8)
        
        for lower, upper in color_ranges:
            mask = cv2.inRange(section, lower, upper)
            total_mask = cv2.bitwise_or(total_mask, mask)
        
        # Calculate the percentage of pixels matching the color
        color_pixels = np.sum(total_mask > 0)
        total_pixels = section.shape[0] * section.shape[1]
        
        if total_pixels == 0:
            return 0.0
        
        color_ratio = color_pixels / total_pixels
        
        # Also consider brightness - traffic lights are bright
        brightness = np.mean(section[:, :, 2])  # V channel in HSV
        brightness_factor = min(brightness / 150.0, 1.0)  # Lower brightness threshold
        
        # Debug output for color analysis
        if color_ratio > 0.01 or brightness > 100:  # Only show if there's some signal
            h_mean = np.mean(section[:, :, 0])
            s_mean = np.mean(section[:, :, 1])
            v_mean = np.mean(section[:, :, 2])
            print(f"    {color}: ratio={color_ratio:.3f}, brightness={brightness:.1f}, HSV=({h_mean:.1f},{s_mean:.1f},{v_mean:.1f})")
        
        return color_ratio * brightness_factor
    
    def _detect_color_lights(self, hsv_image: np.ndarray, original_image: np.ndarray, 
                           color_name: str, use_strict: bool = False) -> List[Dict]:
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
        # Use strict ranges for color-only detection, permissive for YOLO analysis
        if use_strict:
            color_ranges = self._strict_color_ranges[color_name]
        else:
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
    
    def draw_detections(self, image: np.ndarray, detections: List[Dict], show_method: bool = False) -> np.ndarray:
        """
        Draw detected traffic lights on the image.
        
        Args:
            image: Input image
            detections: List of detections from detect_traffic_lights
            show_method: Whether to show detection method indicators
            
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
            method = detection.get('method', 'unknown')
            
            # Draw bounding box with different thickness for different methods
            color = color_map.get(state, (128, 128, 128))
            thickness = 3 if method in ['two-stage', 'regional'] else 2
            cv2.rectangle(result, (x, y), (x + w, y + h), color, thickness)
            
            # Draw method indicator if requested
            if show_method:
                method_colors = {
                    'yolo': (0, 255, 255),      # Cyan
                    'two-stage': (255, 0, 255), # Magenta  
                    'regional': (255, 128, 0),   # Orange
                    'color': (255, 255, 0)       # Yellow
                }
                method_color = method_colors.get(method, (128, 128, 128))
                cv2.circle(result, (x + w - 15, y + 15), 8, method_color, -1)
                
                # Method abbreviations
                method_text = {
                    'yolo': '1S',
                    'two-stage': '2S', 
                    'regional': 'R',
                    'color': 'C'
                }.get(method, '?')
                
                cv2.putText(result, method_text, (x + w - 20, y + 20), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1)
            
            # Draw confidence bar
            bar_width = w
            bar_height = 6
            bar_x = x
            bar_y = y - bar_height - 5
            
            # Background bar
            cv2.rectangle(result, (bar_x, bar_y), (bar_x + bar_width, bar_y + bar_height), 
                         (50, 50, 50), -1)
            # Confidence bar
            conf_width = int(bar_width * confidence)
            cv2.rectangle(result, (bar_x, bar_y), (bar_x + conf_width, bar_y + bar_height), 
                         (0, 255, 255), -1)
            
            # Draw label with background
            label = f"{state.value}: {confidence:.2f}"
            (text_width, text_height), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)
            
            # Text background
            cv2.rectangle(result, (x, y - text_height - 20), 
                         (x + text_width + 10, y - 10), (0, 0, 0), -1)
            
            # Text
            cv2.putText(result, label, (x + 5, y - 15), cv2.FONT_HERSHEY_SIMPLEX, 
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