"""
Example of object detection in images using feature extraction and matching.
"""

import sys
import os

# Add the parent directory to the path so we can import the src module
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.object_detection import detect_object
from src.utils import download_test_images

def main():
    """Run the image detection example"""
    print("=== Object Detection in Images Example ===")
    
    # Download test images if needed
    print("\nDownloading test images...")
    test_images = download_test_images()
    
    print("\nRunning object detection...")
    print(f"Query image: {test_images['query_image']}")
    print(f"Target image: {test_images['target_image']}")
    
    # Detect the object
    detect_object(test_images['query_image'], test_images['target_image'])
    
    print("\nObject detection complete!")

if __name__ == "__main__":
    main()
