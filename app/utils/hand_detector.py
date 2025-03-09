#!/usr/bin/env python3
# hand_detector.py - Utility for hand detection and preprocessing using MediaPipe

import cv2
import numpy as np
import mediapipe as mp
import tensorflow as tf

class HandDetector:
    """
    Class for detecting and preprocessing hand gestures using MediaPipe.
    Used for both training data preparation and real-time inference.
    """
    def __init__(self, 
                 static_image_mode=False,
                 max_num_hands=2,
                 min_detection_confidence=0.7,
                 min_tracking_confidence=0.5):
        """
        Initialize the HandDetector with MediaPipe Hands.
        
        Args:
            static_image_mode: Whether to treat the input images as a batch of static 
                images, or as a video stream.
            max_num_hands: Maximum number of hands to detect.
            min_detection_confidence: Minimum confidence value for hand detection to be 
                considered successful.
            min_tracking_confidence: Minimum confidence value for the hand landmarks to be 
                considered tracked successfully.
        """
        self.mp_hands = mp.solutions.hands
        self.mp_drawing = mp.solutions.drawing_utils
        self.mp_drawing_styles = mp.solutions.drawing_styles
        
        self.hands = self.mp_hands.Hands(
            static_image_mode=static_image_mode,
            max_num_hands=max_num_hands,
            min_detection_confidence=min_detection_confidence,
            min_tracking_confidence=min_tracking_confidence
        )
        
    def find_hands(self, img, draw=True):
        """
        Find hands in an RGB image.
        
        Args:
            img: Input image (BGR format from OpenCV)
            draw: Whether to draw the hand landmarks on the image
            
        Returns:
            img: Image with hand landmarks drawn if draw=True
            results: MediaPipe hand detection results
        """
        # Convert BGR to RGB
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        
        # Process the image and detect hands
        results = self.hands.process(img_rgb)
        
        # Draw hand landmarks if requested
        if draw and results.multi_hand_landmarks:
            for hand_landmarks in results.multi_hand_landmarks:
                self.mp_drawing.draw_landmarks(
                    img,
                    hand_landmarks,
                    self.mp_hands.HAND_CONNECTIONS,
                    self.mp_drawing_styles.get_default_hand_landmarks_style(),
                    self.mp_drawing_styles.get_default_hand_connections_style()
                )
        
        return img, results
    
    def extract_hand_region(self, img, hand_landmarks, padding=20, target_size=(224, 224)):
        """
        Extract and preprocess a single hand region from the image.
        
        Args:
            img: Input image (BGR format from OpenCV)
            hand_landmarks: MediaPipe hand landmarks for a single hand
            padding: Additional padding around the hand region (pixels)
            target_size: Output size for the extracted hand region
            
        Returns:
            Cropped and preprocessed hand image, or None if extraction fails
        """
        if img is None or hand_landmarks is None:
            return None
        
        try:
            # Get image dimensions
            h, w, c = img.shape
            
            # Validate landmark data
            if not hand_landmarks.landmark:
                return None
            
            # Find bounding box from landmarks
            x_min, y_min = w, h
            x_max, y_max = 0, 0
            
            for landmark in hand_landmarks.landmark:
                x, y = int(landmark.x * w), int(landmark.y * h)
                x_min = max(0, min(x_min, x))
                y_min = max(0, min(y_min, y))
                x_max = min(w, max(x_max, x))
                y_max = min(h, max(y_max, y))
            
            # Add padding
            x_min = max(0, x_min - padding)
            y_min = max(0, y_min - padding)
            x_max = min(w, x_max + padding)
            y_max = min(h, y_max + padding)
            
            # Ensure we have a valid bounding box with minimum dimensions
            if x_min >= x_max or y_min >= y_max or (x_max - x_min) < 10 or (y_max - y_min) < 10:
                return None
            
            # Crop the hand region
            hand_img = img[y_min:y_max, x_min:x_max].copy()
            
            # Verify we have a valid cropped image
            if hand_img is None or hand_img.shape[0] == 0 or hand_img.shape[1] == 0:
                return None
            
            # Resize to target size
            hand_img = cv2.resize(hand_img, target_size, interpolation=cv2.INTER_AREA)
            
            return hand_img
            
        except Exception as e:
            print(f"Error in extract_hand_region: {e}")
            return None
    
    def preprocess_for_model(self, hand_img):
        """
        Preprocess the hand image for model inference.
        
        Args:
            hand_img: Cropped hand image
            
        Returns:
            Preprocessed image ready for model inference
        """
        if hand_img is None:
            return None
        
        try:
            # Convert to RGB
            hand_img_rgb = cv2.cvtColor(hand_img, cv2.COLOR_BGR2RGB)
            
            # Verify image has correct dimensions (224x224x3)
            if hand_img_rgb.shape[0] != 224 or hand_img_rgb.shape[1] != 224 or hand_img_rgb.shape[2] != 3:
                print(f"Invalid image shape for model: {hand_img_rgb.shape}")
                # Resize again to ensure correct dimensions
                hand_img_rgb = cv2.resize(hand_img_rgb, (224, 224), interpolation=cv2.INTER_AREA)
            
            # Normalize pixel values to [0, 1]
            hand_img_rgb = hand_img_rgb.astype(np.float32) / 255.0
            
            # Add batch dimension
            hand_img_rgb = np.expand_dims(hand_img_rgb, axis=0)
            
            return hand_img_rgb
            
        except Exception as e:
            print(f"Error preprocessing image: {e}")
            import traceback
            traceback.print_exc()
            return None
    
    def process_frame(self, img, draw=True):
        """
        Process a frame to find and draw hand landmarks.
        
        Args:
            img: Input image (BGR format from OpenCV)
            draw: Whether to draw the hand landmarks on the image
            
        Returns:
            processed_img: Image with hand landmarks drawn if draw=True
            hand_imgs: List of cropped hand images for recognition, or empty list if no hands detected
        """
        if img is None:
            print("Error: Empty image passed to process_frame")
            return None, []
        
        # Find hands in the image
        processed_img, results = self.find_hands(img, draw)
        
        # Extract hand regions for each detected hand
        hand_imgs = []
        
        if results.multi_hand_landmarks:
            for idx, hand_landmarks in enumerate(results.multi_hand_landmarks):
                try:
                    # Extract hand region for this hand
                    hand_img = self.extract_hand_region(img, hand_landmarks, padding=20)
                    if hand_img is not None:
                        hand_imgs.append(hand_img)
                except Exception as e:
                    print(f"Error extracting hand region for hand {idx}: {e}")
        
        return processed_img, hand_imgs
    
    def get_hand_landmarks(self, results):
        """
        Extract normalized hand landmarks from detection results.
        
        Args:
            results: MediaPipe hand detection results
            
        Returns:
            Normalized hand landmarks as a numpy array, or None if no hand detected
        """
        if not results.multi_hand_landmarks:
            return None
        
        # Extract landmark coordinates
        hand_landmarks = results.multi_hand_landmarks[0]
        landmarks = []
        
        for landmark in hand_landmarks.landmark:
            landmarks.extend([landmark.x, landmark.y, landmark.z])
        
        return np.array(landmarks) 