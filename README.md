# Feature Extraction and Matching

This repository contains code for feature extraction and matching using SIFT (Scale-Invariant Feature Transform) and other techniques for object detection in images and videos.

## Overview

The project implements object detection using feature extraction and matching algorithms. It can:

1. Detect objects in static images
2. Track objects in videos (bonus feature)
3. Work with various feature detectors (SIFT, ORB)

The implementation uses OpenCV, which provides efficient pre-implemented feature extractors and matchers.

## Installation

1. Clone this repository:
```bash
git clone https://github.com/Assem-ElQersh/feature-extraction-matching.git
cd feature-extraction-matching
```

2. Create a virtual environment (optional but recommended):
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

3. Install dependencies:
```bash
pip install -r requirements.txt
```

## Project Structure

```
SIFT-Feature-Extraction-and-Matching
├── LICENSE
├── README.md
├── requirements.txt
├── .gitignore
├── src/
│   ├── __init__.py
│   ├── feature_extraction.py  # Core feature extraction functionality
│   ├── object_detection.py    # Object detection implementation
│   └── utils.py               # Utility functions (image loading, etc.)
├── data/
│   ├── images/                # Sample images for testing
│   └── videos/                # Sample videos for testing
├── examples/
│   ├── image_detection_example.py   # Example of object detection in images
│   └── video_detection_example.py   # Example of object detection in videos
└── output/                    # Directory for output results
```

## Usage

### Object Detection in Images

```python
from src.object_detection import detect_object

# Detect an object in an image
detect_object('data/images/box.png', 'data/images/box_in_scene.png')
```

### Object Detection in Videos

```python
from src.object_detection import detect_object_in_video

# Use webcam for detection
detect_object_in_video('data/images/box.png')

# Use a video file for detection
detect_object_in_video('data/images/box.png', 'data/videos/sample_video.mp4')
```

## Examples

Run the examples directly:

```bash
# Object detection in images
python examples/image_detection_example.py

# Object detection in videos
python examples/video_detection_example.py
```

## Sample Data

The repository includes automatic download functionality for sample images and videos from OpenCV's sample data and other sources for testing purposes.

Sample images:
- box.png: A query image of a cereal box
- box_in_scene.png: A target image with the box in a cluttered scene

## License

This project is licensed under the MIT License - see the [LICENSE](https://github.com/Assem-ElQersh/SIFT-Feature-Extraction-and-Matching/blob/main/LICENSE) file for details.
