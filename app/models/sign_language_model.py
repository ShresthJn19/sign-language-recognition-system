#!/usr/bin/env python3
# sign_language_model.py - Utility for loading and using the sign language recognition model

import os
import json
import numpy as np
import tensorflow as tf
import time
import cv2
import traceback

class SignLanguageModel:
    """
    Class for loading and using the trained sign language recognition model.
    Handles inference, prediction stabilization, and performance monitoring.
    """
    def __init__(self, model_dir='app/models/saved'):
        """
        Initialize the model from the saved model files.
        
        Args:
            model_dir: Directory containing the saved model and class mapping
        """
        self.model_dir = model_dir
        self.model_path = os.path.join(model_dir, 'sign_language_model.h5')
        self.class_mapping_path = os.path.join(model_dir, 'class_mapping.json')
        
        # Check if model files exist before loading
        self._verify_model_files()
        
        # Load the model
        self.model = self.load_model()
        
        # Load class mapping
        self.class_mapping = self.load_class_mapping()
        
        # Recent predictions for stabilization
        self.recent_predictions = []
        self.prediction_window = 5  # Number of frames to consider for stabilization
        
        # Performance tracking
        self.inference_times = []
    
    def _verify_model_files(self):
        """Verify that the required model files exist"""
        if not os.path.exists(self.model_path):
            print(f"ERROR: Model file not found at {self.model_path}")
            print("Creating a placeholder model...")
            self._create_placeholder_model()
        
        if not os.path.exists(self.class_mapping_path):
            print(f"ERROR: Class mapping file not found at {self.class_mapping_path}")
            print("Creating a default class mapping...")
            self._create_default_class_mapping()
    
    def _create_placeholder_model(self):
        """Create a placeholder model if the original is missing"""
        try:
            from tensorflow.keras.applications import MobileNetV3Small
            from tensorflow.keras.models import Model
            from tensorflow.keras.layers import Dense, GlobalAveragePooling2D, Dropout, BatchNormalization
            
            # Create directory if it doesn't exist
            os.makedirs(os.path.dirname(self.model_path), exist_ok=True)
            
            # Create a base model
            base_model = MobileNetV3Small(
                weights=None,  # Don't load weights
                include_top=False,
                input_shape=(224, 224, 3)
            )
            
            # Add custom layers
            x = base_model.output
            x = GlobalAveragePooling2D()(x)
            x = BatchNormalization()(x)
            x = Dense(256, activation='relu')(x)
            x = Dropout(0.5)(x)
            x = BatchNormalization()(x)
            x = Dense(128, activation='relu')(x)
            x = Dropout(0.3)(x)
            predictions = Dense(35, activation='softmax')(x)  # 26 letters + 9 digits
            
            model = Model(inputs=base_model.input, outputs=predictions)
            
            # Compile the model
            model.compile(
                optimizer='adam',
                loss='categorical_crossentropy',
                metrics=['accuracy']
            )
            
            # Save the model
            model.save(self.model_path)
            print(f"Placeholder model saved to {self.model_path}")
        except Exception as e:
            print(f"Error creating placeholder model: {e}")
            print(traceback.format_exc())
    
    def _create_default_class_mapping(self):
        """Create a default class mapping if original is missing"""
        try:
            # Create directory if it doesn't exist
            os.makedirs(os.path.dirname(self.class_mapping_path), exist_ok=True)
            
            # Create a mapping for letters A-Z and numbers 1-9
            class_mapping = {}
            for i in range(26):
                class_mapping[str(i)] = chr(i + 65)  # A-Z
            for i in range(9):
                class_mapping[str(i + 26)] = str(i + 1)  # 1-9
            
            with open(self.class_mapping_path, 'w') as f:
                json.dump(class_mapping, f, indent=4)
            print(f"Default class mapping saved to {self.class_mapping_path}")
        except Exception as e:
            print(f"Error creating default class mapping: {e}")
            print(traceback.format_exc())
        
    def load_model(self):
        """
        Load the trained model from file.
        
        Returns:
            Loaded TensorFlow model
        """
        try:
            # Try multiple approaches to load the model
            try:
                # Standard loading approach
                model = tf.keras.models.load_model(self.model_path)
                print(f"Model loaded successfully from {self.model_path}")
                return model
            except Exception as first_error:
                print(f"First loading attempt failed: {first_error}")
                
                try:
                    # Try loading with custom_objects
                    model = tf.keras.models.load_model(self.model_path, compile=False)
                    print(f"Model loaded successfully (without compilation) from {self.model_path}")
                    return model
                except Exception as second_error:
                    print(f"Second loading attempt failed: {second_error}")
                    raise Exception(f"All model loading attempts failed: {first_error} | {second_error}")
        except Exception as e:
            print(f"Error loading model: {e}")
            print(traceback.format_exc())
            
            # Create an empty model as a fallback
            print("Creating an empty model as fallback...")
            input_layer = tf.keras.layers.Input(shape=(224, 224, 3))
            output_layer = tf.keras.layers.Dense(35, activation='softmax')(input_layer)
            fallback_model = tf.keras.Model(inputs=input_layer, outputs=output_layer)
            return fallback_model
    
    def load_class_mapping(self):
        """
        Load the class mapping from file.
        
        Returns:
            Dictionary mapping class indices to class names
        """
        try:
            with open(self.class_mapping_path, 'r') as f:
                class_mapping = json.load(f)
            print(f"Class mapping loaded successfully from {self.class_mapping_path}")
            return class_mapping
        except Exception as e:
            print(f"Error loading class mapping: {e}")
            print(traceback.format_exc())
            
            # Create a default class mapping if file not found
            return {str(i): chr(i + 65) if i < 26 else str(i - 26 + 1) for i in range(35)}
    
    def predict(self, image):
        """
        Predict the sign language class from a preprocessed image.
        
        Args:
            image: Preprocessed image (batch of 1)
            
        Returns:
            class_name: Predicted class name
            confidence: Confidence score for the prediction
        """
        if image is None:
            return None, 0.0
        
        try:
            # Safety check: Ensure image has the correct dimensions
            if not isinstance(image, np.ndarray):
                print(f"Image is not a numpy array: {type(image)}")
                return None, 0.0
                
            # Check if image has the right shape
            expected_shape = (1, 224, 224, 3)  # Batch, height, width, channels
            if image.shape != expected_shape:
                print(f"Image has wrong shape: {image.shape}, expected: {expected_shape}")
                
                # Try to reshape/resize if possible
                if len(image.shape) == 3:  # Missing batch dimension
                    image = np.expand_dims(image, axis=0)
                    print(f"Added batch dimension, new shape: {image.shape}")
                
                if image.shape != expected_shape:
                    print("Could not correct image shape, aborting prediction")
                    return None, 0.0
            
            # Measure inference time
            start_time = time.time()
            
            # Make prediction
            predictions = self.model.predict(image, verbose=0)
            
            # Record inference time
            inference_time = time.time() - start_time
            self.inference_times.append(inference_time)
            
            # Get the predicted class and confidence
            predicted_class_idx = np.argmax(predictions[0])
            confidence = float(predictions[0][predicted_class_idx])
            
            # Map the class index to class name
            class_name = self.class_mapping.get(str(predicted_class_idx), "Unknown")
            
            return class_name, confidence
        except Exception as e:
            print(f"Error making prediction: {e}")
            print(traceback.format_exc())
            return None, 0.0
    
    def get_average_inference_time(self):
        """
        Calculate the average inference time for performance monitoring.
        
        Returns:
            Average inference time in milliseconds
        """
        if not self.inference_times:
            return 0
        
        # Keep only the last 100 measurements
        if len(self.inference_times) > 100:
            self.inference_times = self.inference_times[-100:]
        
        # Calculate average in milliseconds
        avg_time_ms = np.mean(self.inference_times) * 1000
        return avg_time_ms
    
    def stabilize_prediction(self, class_name, confidence):
        """
        Stabilize predictions over multiple frames to reduce jitter.
        
        Args:
            class_name: Current predicted class name
            confidence: Current confidence score
            
        Returns:
            stabilized_class: Stabilized class name
            stabilized_confidence: Confidence for the stabilized prediction
        """
        if class_name is None:
            # If no hand detected, clear recent predictions
            self.recent_predictions = []
            return None, 0.0
        
        # Add current prediction
        self.recent_predictions.append((class_name, confidence))
        
        # Keep only the most recent predictions
        if len(self.recent_predictions) > self.prediction_window:
            self.recent_predictions.pop(0)
        
        # Count occurrences of each class
        class_counts = {}
        class_confidences = {}
        
        for pred_class, conf in self.recent_predictions:
            if pred_class in class_counts:
                class_counts[pred_class] += 1
                class_confidences[pred_class] += conf
            else:
                class_counts[pred_class] = 1
                class_confidences[pred_class] = conf
        
        # Find the most common class
        most_common_class = max(class_counts.items(), key=lambda x: x[1])[0]
        avg_confidence = class_confidences[most_common_class] / class_counts[most_common_class]
        
        return most_common_class, avg_confidence
    
    def visualize_prediction(self, frame, class_name, confidence):
        """
        Add visualization of the prediction to the frame.
        
        Args:
            frame: The video frame
            class_name: Predicted class name
            confidence: Confidence score
            
        Returns:
            Annotated frame
        """
        if class_name is None:
            text = "No hand detected"
            cv2.putText(frame, text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 0, 255), 2)
            return frame
        
        # Add prediction text
        text = f"{class_name} ({confidence:.2f})"
        cv2.putText(frame, text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)
        
        # Add inference time
        avg_time_ms = self.get_average_inference_time()
        time_text = f"Inference: {avg_time_ms:.1f} ms"
        cv2.putText(frame, time_text, (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 0, 0), 1)
        
        return frame 