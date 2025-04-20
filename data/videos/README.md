# Sample Videos

This directory contains sample videos for testing the feature extraction and matching algorithms.

## Automatically Downloaded Videos

The following videos will be automatically downloaded when running the video detection examples:

1. `sample_video.mp4`: A sample video for testing object detection

## Using Your Own Videos

You can add your own videos to this directory and use them for testing. To use your own videos, modify the code to use your video file paths:

```python
# Example using custom videos
from src.object_detection import detect_object_in_video_improved

# Replace with your own image and video paths
query_image = 'data/images/your_query_image.jpg'
video_path = 'data/videos/your_video.mp4'

detect_object_in_video_improved(query_image, video_path)
```

## Video Format Support

The code supports various video formats that OpenCV can read, including:
- MP4 (.mp4)
- AVI (.avi)
- MOV (.mov)
- WMV (.wmv)

## Using Webcam

You can also use your webcam for real-time object detection:

```python
from src.object_detection import detect_object_in_video_improved

query_image = 'data/images/your_query_image.jpg'
detect_object_in_video_improved(query_image)  # No video path means using webcam
```
