#!/usr/bin/env python3
"""
Simple Video Frame Extractor
Extract every 15th frame from videos
"""

import cv2
from pathlib import Path

def extract_frames():
    video_dir = Path("Video")
    output_dir = Path("Photos")
    
    # Get all video files
    videos = list(video_dir.glob("*.mp4")) + list(video_dir.glob("*.avi"))
    
    if not videos:
        print("No videos found in Video/ folder")
        return
    
    frame_num = 0
    
    for video_path in videos:
        print(f"Processing {video_path.name}...")
        
        cap = cv2.VideoCapture(str(video_path))
        frame_count = 0
        
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            
            # Save every 15th frame
            if frame_count % 15 == 0:
                filename = f"frame{frame_num:04d}.jpg"
                cv2.imwrite(str(output_dir / filename), frame)
                print(f"Saved {filename}")
                frame_num += 1
            
            frame_count += 1
        
        cap.release()
    
    print(f"Done! Extracted {frame_num} frames total")

if __name__ == "__main__":
    extract_frames()