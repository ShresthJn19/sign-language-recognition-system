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
                 max_num_hands=1,
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
    
    def extract_hand_region(self, img, results, padding=20, target_size=(224, 224)):
        """
        Extract and preprocess the hand region from the image.
        
        Args:
            img: Input image (BGR format from OpenCV)
            results: MediaPipe hand detection results
            padding: Additional padding around the hand region (pixels)
            target_size: Output size for the extracted hand region
            
        Returns:
            Cropped and preprocessed hand image, or None if no hand is detected
        """
        if not results.multi_hand_landmarks:
            return None
        
        try:
            # Get image dimensions
            h, w, c = img.shape
            
            # Get the bounding box for the first detected hand
            hand_landmarks = results.multi_hand_landmarks[0]
            x_min, y_min = w, h
            x_max, y_max = 0, 0
            
            # Find bounding box coordinates
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
            
            # Ensure we have a valid bounding box
            if x_min >= x_max or y_min >= y_max:
                print("Invalid bounding box detected")
                return None
                
            # Crop the image to the hand region
            hand_img = img[y_min:y_max, x_min:x_max]
            
            # If bounding box is somehow invalid, return None
            if hand_img.size == 0 or hand_img.shape[0] == 0 or hand_img.shape[1] == 0:
                print("Hand image has invalid dimensions:", hand_img.shape if hasattr(hand_img, 'shape') else "no shape")
                return None
            
            # Resize to target size with explicit interpolation method
            hand_img = cv2.resize(hand_img, target_size, interpolation=cv2.INTER_AREA)
            
            # Verify the resized image has the correct dimensions
            if hand_img.shape[0] != target_size[1] or hand_img.shape[1] != target_size[0]:
                print(f"Resizing failed. Expected: {target_size}, Got: {hand_img.shape[:2]}")
                return None
                
            return hand_img
            
        except Exception as e:
            print(f"Error extracting hand region: {e}")
            import traceback
            traceback.print_exc()
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
    
    def process_frame(self, frame, draw=True, target_size=(224, 224)):
        """
        Process a frame for hand detection and model inference.
        
        Args:
            frame: Input frame from video feed (BGR format from OpenCV)
            draw: Whether to draw hand landmarks on the frame
            target_size: Target size for model input
            
        Returns:
            frame: Original frame with hand landmarks drawn if draw=True
            processed_img: Preprocessed hand image for model inference, or None if no hand detected
        """
        # Verify frame is valid
        if frame is None or frame.size == 0:
            print("Error: Received empty frame in process_frame")
            # Create a blank frame with error message
            blank_frame = np.zeros((480, 640, 3), dtype=np.uint8)
            cv2.putText(blank_frame, "Invalid camera input", (50, 240), 
                       cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
            return blank_frame, None
            
        try:
            # Find hands in the frame
            processed_frame, results = self.find_hands(frame, draw=draw)
            
            # Extract and preprocess hand region if detected
            if results.multi_hand_landmarks:
                hand_img = self.extract_hand_region(frame, results, target_size=target_size)
                processed_img = self.preprocess_for_model(hand_img)
                return processed_frame, processed_img
            
            return processed_frame, None
            
        except Exception as e:
            print(f"Error in hand detection: {e}")
            import traceback
            traceback.print_exc()
            # Return the original frame and None for processed image
            return frame, None
    
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