#!/usr/bin/env python3
"""
Data Collection Script
Extract frames from videos for training data.
"""

import cv2
import os
from pathlib import Path
import argparse

def extract_frames_from_video(video_path, output_dir, interval=30, max_frames=1000):
    """Extract frames from video for annotation."""
    
    print(f"🎬 Extracting frames from: {video_path}")
    
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"❌ Could not open video: {video_path}")
        return
    
    frame_count = 0
    saved_count = 0
    
    while saved_count < max_frames:
        ret, frame = cap.read()
        if not ret:
            break
        
        if frame_count % interval == 0:
            # Save frame
            filename = f"frame_{saved_count:06d}.jpg"
            filepath = os.path.join(output_dir, filename)
            cv2.imwrite(filepath, frame)
            saved_count += 1
            
            if saved_count % 100 == 0:
                print(f"  📸 Extracted {saved_count} frames...")
        
        frame_count += 1
    
    cap.release()
    print(f"✅ Extracted {saved_count} frames to {output_dir}")

def main():
    parser = argparse.ArgumentParser(description="Extract training frames from videos")
    parser.add_argument("--video", required=True, help="Path to input video")
    parser.add_argument("--output", default="training_data/extracted_frames", help="Output directory")
    parser.add_argument("--interval", type=int, default=30, help="Extract every N frames")
    parser.add_argument("--max-frames", type=int, default=1000, help="Maximum frames to extract")
    
    args = parser.parse_args()
    
    extract_frames_from_video(args.video, args.output, args.interval, args.max_frames)

if __name__ == "__main__":
    main()
