#!/usr/bin/env python3

import cv2
import numpy as np

def create_test_video():
    """Create a simple test video with a moving traffic light."""
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter('data/test_video.mp4', fourcc, 10.0, (640, 480))
    
    for i in range(50):  # 5 seconds at 10 FPS
        # Create frame
        frame = np.zeros((480, 640, 3), dtype=np.uint8)
        
        # Add some background
        cv2.rectangle(frame, (0, 0), (640, 480), (20, 30, 40), -1)
        
        # Add a pole
        cv2.rectangle(frame, (300 + i*2, 50), (320 + i*2, 400), (70, 70, 70), -1)
        
        # Add traffic light that changes state
        light_x = 280 + i*2
        light_y = 150
        cv2.rectangle(frame, (light_x, light_y), (light_x + 40, light_y + 100), (40, 40, 40), -1)
        
        # Cycle through states
        state = i // 15  # Change every 1.5 seconds
        if state % 3 == 0:  # Red
            cv2.circle(frame, (light_x + 20, light_y + 25), 12, (0, 0, 255), -1)
        elif state % 3 == 1:  # Yellow  
            cv2.circle(frame, (light_x + 20, light_y + 50), 12, (0, 255, 255), -1)
        else:  # Green
            cv2.circle(frame, (light_x + 20, light_y + 75), 12, (0, 255, 0), -1)
        
        out.write(frame)
    
    out.release()
    print('✅ Test video created: data/test_video.mp4')

if __name__ == "__main__":
    create_test_video()