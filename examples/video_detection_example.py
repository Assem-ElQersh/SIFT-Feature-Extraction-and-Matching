"""
Example of object detection in videos using feature extraction and matching.
"""

import sys
import os
import argparse

# Add the parent directory to the path so we can import the src module
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.object_detection import detect_object_in_video_improved
from src.utils import download_test_images, download_test_video

def main():
    """Run the video detection example"""
    parser = argparse.ArgumentParser(description='Object detection in video example')
    parser.add_argument('--webcam', action='store_true', help='Use webcam instead of video file')
    args = parser.parse_args()
    
    print("=== Object Detection in Video Example ===")
    
    # Download test images and video if needed
    print("\nDownloading test images...")
    test_images = download_test_images()
    
    if not args.webcam:
        print("\nDownloading test video...")
        test_video = download_test_video()
        
        print("\nRunning object detection on video...")
        print(f"Query image: {test_images['query_image']}")
        print(f"Video: {test_video}")
        
        # Detect the object in the video
        detect_object_in_video_improved(test_images['query_image'], test_video)
    else:
        print("\nRunning object detection on webcam...")
        print(f"Query image: {test_images['query_image']}")
        
        # Detect the object using webcam
        detect_object_in_video_improved(test_images['query_image'])
    
    print("\nObject detection complete!")

if __name__ == "__main__":
    main()
