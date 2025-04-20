"""
Utility functions for feature extraction and matching.
"""

import os
from urllib.request import urlretrieve

def download_test_images():
    """Download sample images for testing if they don't exist"""
    # Create a directory for test images
    if not os.path.exists('data/images'):
        os.makedirs('data/images')
    
    # Sample images URLs
    image_urls = {
        'box.png': 'https://raw.githubusercontent.com/opencv/opencv/master/samples/data/box.png',
        'box_in_scene.png': 'https://raw.githubusercontent.com/opencv/opencv/master/samples/data/box_in_scene.png',
        'building.jpg': 'https://raw.githubusercontent.com/opencv/opencv/master/samples/data/building.jpg',
    }
    
    # Download images
    for img_name, url in image_urls.items():
        img_path = os.path.join('data/images', img_name)
        if not os.path.exists(img_path):
            print(f"Downloading {img_name}...")
            urlretrieve(url, img_path)
            print(f"Downloaded {img_name}")
        else:
            print(f"{img_name} already exists")
    
    return {
        'query_image': os.path.join('data/images', 'box.png'),
        'target_image': os.path.join('data/images', 'box_in_scene.png')
    }

def download_test_video():
    """Download a sample video for testing if it doesn't exist"""
    # Create a directory for test videos
    if not os.path.exists('data/videos'):
        os.makedirs('data/videos')
    
    # Sample video URL - using a Creative Commons video
    video_url = 'https://storage.googleapis.com/gtv-videos-bucket/sample/ForBiggerBlazes.mp4'
    video_path = os.path.join('data/videos', 'sample_video.mp4')
    
    if not os.path.exists(video_path):
        print(f"Downloading sample video...")
        urlretrieve(video_url, video_path)
        print(f"Downloaded sample video to {video_path}")
    else:
        print(f"Sample video already exists at {video_path}")
    
    return video_path

def ensure_output_directory():
    """Ensure the output directory exists"""
    if not os.path.exists('output'):
        os.makedirs('output')
    return 'output'
