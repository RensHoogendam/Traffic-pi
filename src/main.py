#!/usr/bin/env python3
"""
Traffic Light Detection Main Application

This script demonstrates various ways to use the Traffic Light Detector:
- Single image processing
- Video file processing
- Real-time camera feed (for Raspberry Pi)
- Batch processing of multiple images

Usage:
    python main.py --image path/to/image.jpg
    python main.py --video path/to/video.mp4
    python main.py --camera
    python main.py --batch path/to/images/
"""

import argparse
import cv2
import os
import sys
import tempfile
import yt_dlp
from pathlib import Path
from urllib.parse import urlparse

from .traffic_light_detector import TrafficLightDetector


def process_image(detector: TrafficLightDetector, image_path: str, output_path: str = None):
    """
    Process a single image and detect traffic lights.
    
    Args:
        detector: TrafficLightDetector instance
        image_path: Path to input image
        output_path: Optional path to save result image
    """
    print(f"Processing image: {image_path}")
    
    # Read image
    image = cv2.imread(image_path)
    if image is None:
        print(f"Error: Could not read image {image_path}")
        return
    
    # Detect traffic lights
    detections = detector.detect_traffic_lights(image)
    
    # Print results
    print(f"Found {len(detections)} traffic lights:")
    for i, detection in enumerate(detections, 1):
        bbox = detection['bbox']
        state = detection['state'].value
        confidence = detection['confidence']
        print(f"  {i}. State: {state}, Confidence: {confidence:.3f}, "
              f"Position: ({bbox[0]}, {bbox[1]}, {bbox[2]}, {bbox[3]})")
    
    # Draw detections
    result_image = detector.draw_detections(image, detections)
    
    # Show result
    cv2.imshow("Traffic Light Detection", result_image)
    print("Press any key to continue...")
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    
    # Save result if output path provided
    if output_path:
        cv2.imwrite(output_path, result_image)
        print(f"Result saved to: {output_path}")


def is_youtube_url(url: str) -> bool:
    """Check if a given string is a YouTube URL"""
    youtube_domains = ['youtube.com', 'youtu.be', 'www.youtube.com', 'm.youtube.com']
    try:
        parsed = urlparse(url)
        return parsed.netloc.lower() in youtube_domains
    except:
        return False


def download_youtube_video(url: str) -> str:
    """
    Download a YouTube video and return the path to the downloaded file.
    
    Args:
        url: YouTube URL
        
    Returns:
        Path to the downloaded video file
    """
    print(f"Downloading YouTube video: {url}")
    
    # Create temporary directory for downloads
    temp_dir = tempfile.mkdtemp(prefix="traffic_pi_")
    
    # Configure yt-dlp options
    ydl_opts = {
        'format': 'best[ext=mp4]/best',  # Prefer mp4, fallback to best available
        'outtmpl': os.path.join(temp_dir, '%(title)s.%(ext)s'),
        'quiet': False,  # Set to True to suppress output
    }
    
    try:
        with yt_dlp.YoutubeDL(ydl_opts) as ydl:
            # Extract info to get the final filename
            info = ydl.extract_info(url, download=False)
            filename = ydl.prepare_filename(info)
            
            # Download the video
            ydl.download([url])
            
            print(f"Downloaded to: {filename}")
            return filename
    
    except Exception as e:
        print(f"Error downloading YouTube video: {e}")
        raise


def process_video(detector: TrafficLightDetector, video_path: str, output_path: str = None):
    """
    Process a video file and detect traffic lights in each frame.
    Supports local files and YouTube URLs.
    
    Args:
        detector: TrafficLightDetector instance
        video_path: Path to input video or YouTube URL
        output_path: Optional path to save result video
    """
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
    
    print(f"Processing video: {video_path}")
    
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
    
    # Set up video writer if output path provided
    out = None
    if output_path:
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
    
    frame_count = 0
    
    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            
            frame_count += 1
            print(f"Processing frame {frame_count}/{total_frames}", end='\r')
            
            # Detect traffic lights
            detections = detector.detect_traffic_lights(frame)
            
            # Draw detections
            result_frame = detector.draw_detections(frame, detections)
            
            # Show frame
            cv2.imshow("Traffic Light Detection - Video", result_frame)
            
            # Save frame if output video specified
            if out:
                out.write(result_frame)
            
            # Break on 'q' key
            if cv2.waitKey(1) & 0xFF == ord('q'):
                print("\nStopping video processing...")
                break
    
    finally:
        cap.release()
        if out:
            out.release()
        cv2.destroyAllWindows()
        
        if output_path:
            print(f"\nResult video saved to: {output_path}")
        
        # Clean up temporary YouTube download
        if is_youtube and downloaded_path and os.path.exists(downloaded_path):
            try:
                os.remove(downloaded_path)
                # Also try to remove the temporary directory if it's empty
                temp_dir = os.path.dirname(downloaded_path)
                if os.path.exists(temp_dir) and not os.listdir(temp_dir):
                    os.rmdir(temp_dir)
                print("Cleaned up temporary download")
            except Exception as e:
                print(f"Note: Could not clean up temporary file: {e}")


def process_camera(detector: TrafficLightDetector):
    """
    Process real-time camera feed for traffic light detection.
    
    Args:
        detector: TrafficLightDetector instance
    """
    print("Starting camera feed... Press 'q' to quit")
    
    # Try different camera indices (0 is usually the default camera)
    cap = None
    for camera_idx in [0, 1, 2]:
        cap = cv2.VideoCapture(camera_idx)
        if cap.isOpened():
            print(f"Using camera {camera_idx}")
            break
    
    if not cap or not cap.isOpened():
        print("Error: Could not open camera")
        return
    
    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                print("Error: Could not read frame from camera")
                break
            
            # Detect traffic lights
            detections = detector.detect_traffic_lights(frame)
            
            # Draw detections
            result_frame = detector.draw_detections(frame, detections)
            
            # Add instructions to the frame
            cv2.putText(result_frame, "Press 'q' to quit", (10, 30), 
                       cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
            
            # Show frame
            cv2.imshow("Traffic Light Detection - Live", result_frame)
            
            # Print detections (optional, might be too verbose)
            if detections:
                states = [d['state'].value for d in detections]
                print(f"Detected: {', '.join(states)}", end='\r')
            
            # Break on 'q' key
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
    
    finally:
        cap.release()
        cv2.destroyAllWindows()


def process_batch(detector: TrafficLightDetector, input_dir: str, output_dir: str = None):
    """
    Process multiple images in a directory.
    
    Args:
        detector: TrafficLightDetector instance
        input_dir: Directory containing input images
        output_dir: Optional directory to save results
    """
    print(f"Processing batch of images from: {input_dir}")
    
    # Create output directory if specified
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)
    
    # Supported image extensions
    image_extensions = {'.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.tif'}
    
    # Find all image files
    input_path = Path(input_dir)
    image_files = [f for f in input_path.iterdir() 
                   if f.suffix.lower() in image_extensions]
    
    if not image_files:
        print(f"No image files found in {input_dir}")
        return
    
    print(f"Found {len(image_files)} images to process")
    
    results_summary = []
    
    for i, image_file in enumerate(image_files, 1):
        print(f"Processing {i}/{len(image_files)}: {image_file.name}")
        
        # Read image
        image = cv2.imread(str(image_file))
        if image is None:
            print(f"  Warning: Could not read {image_file.name}")
            continue
        
        # Detect traffic lights
        detections = detector.detect_traffic_lights(image)
        
        # Store results
        results_summary.append({
            'filename': image_file.name,
            'detections': len(detections),
            'states': [d['state'].value for d in detections]
        })
        
        print(f"  Found {len(detections)} traffic lights")
        
        # Save result if output directory specified
        if output_dir:
            result_image = detector.draw_detections(image, detections)
            output_file = Path(output_dir) / f"result_{image_file.name}"
            cv2.imwrite(str(output_file), result_image)
    
    # Print summary
    print("\n=== BATCH PROCESSING SUMMARY ===")
    total_detections = sum(r['detections'] for r in results_summary)
    print(f"Total images processed: {len(results_summary)}")
    print(f"Total traffic lights detected: {total_detections}")
    
    for result in results_summary:
        if result['detections'] > 0:
            states_str = ', '.join(result['states'])
            print(f"  {result['filename']}: {result['detections']} lights ({states_str})")


def main():
    """Main function to parse arguments and run appropriate processing mode"""
    parser = argparse.ArgumentParser(description="Traffic Light Detection System")
    
    # Processing mode arguments (mutually exclusive)
    mode_group = parser.add_mutually_exclusive_group(required=True)
    mode_group.add_argument('--image', type=str, help='Path to input image')
    mode_group.add_argument('--video', type=str, help='Path to input video or YouTube URL')
    mode_group.add_argument('--camera', action='store_true', help='Use camera feed')
    mode_group.add_argument('--batch', type=str, help='Directory with multiple images')
    
    # Optional arguments
    parser.add_argument('--output', type=str, help='Output path for results')
    parser.add_argument('--config', type=str, help='Path to configuration file')
    parser.add_argument('--no-yolo', action='store_true', help='Use color-based detection instead of YOLO')
    
    args = parser.parse_args()
    
    # Initialize detector
    try:
        use_yolo = not args.no_yolo  # Use YOLO unless --no-yolo flag is set
        detector = TrafficLightDetector(config_path=args.config, use_yolo=use_yolo)
        detection_method = "YOLO" if detector.use_yolo else "Color-based"
        print(f"Traffic Light Detector initialized successfully using {detection_method} detection")
    except Exception as e:
        print(f"Error initializing detector: {e}")
        return 1
    
    # Run appropriate processing mode
    try:
        if args.image:
            if not os.path.exists(args.image):
                print(f"Error: Image file {args.image} not found")
                return 1
            process_image(detector, args.image, args.output)
        
        elif args.video:
            # Check if it's a YouTube URL or local file
            if not is_youtube_url(args.video) and not os.path.exists(args.video):
                print(f"Error: Video file {args.video} not found")
                return 1
            process_video(detector, args.video, args.output)
        
        elif args.camera:
            process_camera(detector)
        
        elif args.batch:
            if not os.path.exists(args.batch):
                print(f"Error: Directory {args.batch} not found")
                return 1
            process_batch(detector, args.batch, args.output)
    
    except KeyboardInterrupt:
        print("\nOperation cancelled by user")
        return 1
    except Exception as e:
        print(f"Error during processing: {e}")
        return 1
    
    print("Processing completed successfully!")
    return 0


if __name__ == "__main__":
    exit(main())