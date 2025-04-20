"""
Feature extraction module for image processing.

This module contains functions to extract features from images
using various algorithms such as SIFT and ORB.
"""

import cv2
import numpy as np

def extract_sift_features(image):
    """
    Extract SIFT features from an image.
    
    Args:
        image: A grayscale image (numpy array)
        
    Returns:
        keypoints: List of keypoints
        descriptors: Numpy array of descriptors
    """
    # Initialize SIFT detector
    sift = cv2.SIFT_create()
    
    # Find keypoints and descriptors
    keypoints, descriptors = sift.detectAndCompute(image, None)
    
    return keypoints, descriptors

def extract_orb_features(image, n_features=1000):
    """
    Extract ORB features from an image.
    
    Args:
        image: A grayscale image (numpy array)
        n_features: Maximum number of features to retain (int)
        
    Returns:
        keypoints: List of keypoints
        descriptors: Numpy array of descriptors
    """
    # Initialize ORB detector
    orb = cv2.ORB_create(nfeatures=n_features)
    
    # Find keypoints and descriptors
    keypoints, descriptors = orb.detectAndCompute(image, None)
    
    return keypoints, descriptors

def create_feature_matcher(method='sift'):
    """
    Create a feature matcher based on the chosen method.
    
    Args:
        method: The feature extraction method ('sift' or 'orb')
        
    Returns:
        matcher: A feature matcher object
    """
    if method.lower() == 'sift':
        # FLANN parameters for SIFT
        FLANN_INDEX_KDTREE = 1
        index_params = dict(algorithm=FLANN_INDEX_KDTREE, trees=5)
        search_params = dict(checks=50)
        matcher = cv2.FlannBasedMatcher(index_params, search_params)
    elif method.lower() == 'orb':
        # For ORB, use BFMatcher with Hamming distance
        matcher = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=False)
    else:
        raise ValueError(f"Unsupported method: {method}")
    
    return matcher

def match_features(matcher, descriptors1, descriptors2, ratio_threshold=0.7):
    """
    Match features between two sets of descriptors using Lowe's ratio test.
    
    Args:
        matcher: A feature matcher object
        descriptors1: First set of descriptors
        descriptors2: Second set of descriptors
        ratio_threshold: Threshold for Lowe's ratio test (float)
        
    Returns:
        good_matches: List of good matches that pass the ratio test
    """
    matches = matcher.knnMatch(descriptors1, descriptors2, k=2)
    
    # Apply Lowe's ratio test
    good_matches = []
    for match in matches:
        if len(match) == 2:  # Sometimes fewer than 2 matches are returned
            m, n = match
            if m.distance < ratio_threshold * n.distance:
                good_matches.append(m)
    
    return good_matches
