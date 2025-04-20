"""
Object detection module for image and video processing.

This module contains functions to detect objects in images and videos
using feature extraction and matching techniques.
"""

import cv2
import numpy as np
import matplotlib.pyplot as plt
import os

from src.feature_extraction import extract_sift_features, create_feature_matcher, match_features
from src.utils import ensure_output_directory

def detect_object(query_image_path, target_image_path):
    """
    Detect an object from a query image in a target image.
    
    Args:
        query_image_path: Path to the query image (object to find)
        target_image_path: Path to the target image (scene where to find the object)
        
    Returns:
        target_with_keypoints: Target image with detected keypoints
        target_with_object: Target image with the object outlined
    """
    # Read the query image (object to find)
    query_img = cv2.imread(query_image_path)
    query_img_gray = cv2.cvtColor(query_img, cv2.COLOR_BGR2GRAY)
    
    # Read the target image (scene where to find the object)
    target_img = cv2.imread(target_image_path)
    target_img_gray = cv2.cvtColor(target_img, cv2.COLOR_BGR2GRAY)
    
    # Extract features
    keypoints_query, descriptors_query = extract_sift_features(query_img_gray)
    keypoints_target, descriptors_target = extract_sift_features(target_img_gray)
    
    # Create feature matcher
    matcher = create_feature_matcher('sift')
    
    # Match features
    good_matches = match_features(matcher, descriptors_query, descriptors_target)
    
    print(f"Number of keypoints in query image: {len(keypoints_query)}")
    print(f"Number of keypoints in target image: {len(keypoints_target)}")
    print(f"Number of good matches: {len(good_matches)}")
    
    # Draw the good matches
    if len(good_matches) >= 4:
        # Extract location of good matches
        src_pts = np.float32([keypoints_query[m.queryIdx].pt for m in good_matches]).reshape(-1, 1, 2)
        dst_pts = np.float32([keypoints_target[m.trainIdx].pt for m in good_matches]).reshape(-1, 1, 2)
        
        # Find homography
        M, mask = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, 5.0)
        matchesMask = mask.ravel().tolist()
        
        # Get the corners of the query image
        h, w = query_img_gray.shape
        pts = np.float32([[0, 0], [0, h-1], [w-1, h-1], [w-1, 0]]).reshape(-1, 1, 2)
        
        # Transform the corners to find the object in the target image
        dst = cv2.perspectiveTransform(pts, M)
        
        # Draw the outline of the detected object
        target_img_with_object = target_img.copy()
        cv2.polylines(target_img_with_object, [np.int32(dst)], True, (0, 255, 0), 3, cv2.LINE_AA)
        
        # Draw keypoints on the object in target image
        # Get inlier matches (mask == 1)
        inlier_matches = [good_matches[i] for i in range(len(good_matches)) if matchesMask[i]]
        
        # Draw inlier keypoints on target image
        target_with_keypoints = target_img.copy()
        for match in inlier_matches:
            # Get the keypoint coordinates in the target image
            x, y = map(int, keypoints_target[match.trainIdx].pt)
            cv2.circle(target_with_keypoints, (x, y), 5, (0, 255, 0), -1)
        
        # Prepare a match visualization image
        draw_params = dict(matchColor=(0, 255, 0),  # Green color for matches
                          singlePointColor=None,
                          matchesMask=matchesMask,  # Only draw inliers
                          flags=2)
        
        match_img = cv2.drawMatches(query_img, keypoints_query, 
                                  target_img, keypoints_target, 
                                  good_matches, None, **draw_params)
        
        # Convert images from BGR to RGB for matplotlib display
        match_img_rgb = cv2.cvtColor(match_img, cv2.COLOR_BGR2RGB)
        target_with_object_rgb = cv2.cvtColor(target_img_with_object, cv2.COLOR_BGR2RGB)
        target_with_keypoints_rgb = cv2.cvtColor(target_with_keypoints, cv2.COLOR_BGR2RGB)
        
        # Ensure output directory exists
        output_dir = ensure_output_directory()
        
        # Display results
        plt.figure(figsize=(15, 10))
        
        plt.subplot(2, 2, 1)
        plt.imshow(cv2.cvtColor(query_img, cv2.COLOR_BGR2RGB))
        plt.title('Query Image (Object to Find)')
        plt.axis('off')
        
        plt.subplot(2, 2, 2)
        plt.imshow(match_img_rgb)
        plt.title('Matching Features')
        plt.axis('off')
        
        plt.subplot(2, 2, 3)
        plt.imshow(target_with_object_rgb)
        plt.title('Object Outlined in Target Image')
        plt.axis('off')
        
        plt.subplot(2, 2, 4)
        plt.imshow(target_with_keypoints_rgb)
        plt.title('Object Keypoints in Target Image')
        plt.axis('off')
        
        plt.tight_layout()
        output_path = os.path.join(output_dir, 'detection_results.jpg')
        plt.savefig(output_path)
        print(f"Results saved to {output_path}")
        plt.show()
        
        return target_with_keypoints, target_img_with_object
    else:
        print("Not enough good matches to locate the object")
        return None, None

def detect_object_in_video_improved(query_image_path, video_path=None):
    """
    Improved version of object detection in video with better error handling and performance.
    
    Args:
        query_image_path: Path to the query image (object to find)
        video_path: Path to the video file (if None, use webcam)
    """
    # Read the query image
    query_img = cv2.imread(query_image_path)
    query_img_gray = cv2.cvtColor(query_img, cv2.COLOR_BGR2GRAY)
    
    # Initialize SIFT detector
    detector = cv2.SIFT_create()
    
    # Find keypoints and descriptors for query image
    keypoints_query, descriptors_query = detector.detectAndCompute(query_img_gray, None)
    
    # Initialize FLANN-based matcher
    FLANN_INDEX_KDTREE = 1
    index_params = dict(algorithm=FLANN_INDEX_KDTREE, trees=5)
    search_params = dict(checks=50)
    matcher = cv2.FlannBasedMatcher(index_params, search_params)
    
    # Open video capture
    if video_path is None:
        cap = cv2.VideoCapture(0)
    else:
        cap = cv2.VideoCapture(video_path)
    
    # For performance, we can skip frames
    frame_skip = 2
    frame_count = 0
    
    # For stability, we can use a moving average of detection results
    last_transformations = []
    
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        
        # Skip frames for better performance
        frame_count += 1
        if frame_count % frame_skip != 0:
            # Still display the frame, but without processing
            cv2.imshow('Object Detection', frame)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
            continue
        
        # Process the frame
        frame_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        
        # Detect features in frame
        keypoints_frame, descriptors_frame = detector.detectAndCompute(frame_gray, None)
        
        # Skip if no features found
        if descriptors_frame is None or len(descriptors_frame) < 4:
            cv2.imshow('Object Detection', frame)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
            continue
        
        # Match features
        try:
            matches = matcher.knnMatch(descriptors_query, descriptors_frame, k=2)
            
            # Apply ratio test
            good_matches = []
            for match in matches:
                if len(match) == 2:
                    m, n = match
                    if m.distance < 0.75 * n.distance:
                        good_matches.append(m)
        except Exception as e:
            print(f"Matching error: {e}")
            cv2.imshow('Object Detection', frame)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
            continue
        
        # Draw bounding box if enough matches
        min_matches = 10  # Increased threshold for more stable detection
        if len(good_matches) >= min_matches:
            # Extract matched keypoints
            src_pts = np.float32([keypoints_query[m.queryIdx].pt for m in good_matches]).reshape(-1, 1, 2)
            dst_pts = np.float32([keypoints_frame[m.trainIdx].pt for m in good_matches]).reshape(-1, 1, 2)
            
            # Find homography with increased RANSAC threshold for stability
            try:
                M, mask = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, 8.0)
                
                if M is not None:
                    # Add to our list of recent transformations (for stability)
                    last_transformations.append(M)
                    if len(last_transformations) > 3:  # Keep last 3
                        last_transformations.pop(0)
                    
                    # Use average of recent transformations if available
                    if len(last_transformations) > 1:
                        M = np.mean(last_transformations, axis=0)
                    
                    # Transform object corners
                    h, w = query_img_gray.shape
                    pts = np.float32([[0, 0], [0, h-1], [w-1, h-1], [w-1, 0]]).reshape(-1, 1, 2)
                    
                    try:
                        dst = cv2.perspectiveTransform(pts, M)
                        # Draw outline with thicker line
                        cv2.polylines(frame, [np.int32(dst)], True, (0, 255, 0), 3, cv2.LINE_AA)
                        
                        # Add text label
                        cv2.putText(frame, "Detected Object", tuple(np.int32(dst)[0][0]), 
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
                    except cv2.error:
                        pass
            except cv2.error as e:
                print(f"Homography error: {e}")
        
        # Display frame
        cv2.imshow('Object Detection', frame)
        
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    
    cap.release()
    cv2.destroyAllWindows()
