# Sample Images

This directory contains sample images for testing the feature extraction and matching algorithms.

## Automatically Downloaded Images

The following images will be automatically downloaded when running the examples:

1. `box.png`: A query image of a cereal box
2. `box_in_scene.png`: A target image with the box in a cluttered scene
3. `building.jpg`: An additional test image

## Using Your Own Images

You can add your own images to this directory and use them for testing. To use your own images, modify the code to use your image file paths:

```python
# Example using custom images
from src.object_detection import detect_object

# Replace with your own image paths
query_image = 'data/images/your_query_image.jpg'
target_image = 'data/images/your_target_image.jpg'

detect_object(query_image, target_image)
```

## Image Format Support

The code supports various image formats that OpenCV can read, including:
- PNG (.png)
- JPEG (.jpg, .jpeg)
- BMP (.bmp)
- TIFF (.tiff, .tif)
