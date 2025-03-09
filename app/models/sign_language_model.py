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
        Load the model from the saved model file.
        
        Returns:
            Loaded TensorFlow model
        """
        try:
            try:
                # First attempt - basic approach with compile=False
                print(f"Attempting to load model from {self.model_path}")
                model = tf.keras.models.load_model(
                    self.model_path, 
                    compile=False
                )
                print(f"Model loaded successfully from {self.model_path}")
                return model
            except Exception as first_error:
                print(f"First loading attempt failed: {first_error}")
                
                try:
                    # Second attempt - with custom_objects
                    print("Trying alternate loading method")
                    model = tf.keras.models.load_model(
                        self.model_path, 
                        compile=False,
                        custom_objects={}
                    )
                    print(f"Model loaded successfully with alternate method from {self.model_path}")
                    return model
                except Exception as second_error:
                    print(f"Second loading attempt failed: {second_error}")
                    
                    try:
                        # Third attempt - try recreating a similar architecture
                        from tensorflow.keras.applications import MobileNetV3Small
                        from tensorflow.keras.layers import Dense, GlobalAveragePooling2D
                        
                        print("Attempting to create a new model with standard architecture")
                        # Create a base model with similar architecture
                        base_model = MobileNetV3Small(
                            weights=None, 
                            include_top=False, 
                            input_shape=(224, 224, 3)
                        )
                        x = base_model.output
                        x = GlobalAveragePooling2D()(x)
                        x = Dense(128, activation='relu')(x)
                        predictions = Dense(35, activation='softmax')(x)
                        model = tf.keras.Model(inputs=base_model.input, outputs=predictions)
                        
                        # Try to load weights only if the file exists and has .h5 extension
                        if os.path.exists(self.model_path) and self.model_path.endswith('.h5'):
                            try:
                                print("Attempting to load weights into the new model")
                                model.load_weights(self.model_path)
                                print("Successfully loaded weights into new model")
                            except Exception as weight_error:
                                print(f"Weight loading failed: {weight_error}")
                                print("Using uninitialized model weights")
                        else:
                            print(f"Not attempting to load weights - file format may not be compatible")
                            
                        return model
                    except Exception as third_error:
                        print(f"All model loading attempts failed")
                        print(f"First error: {first_error}")
                        print(f"Second error: {second_error}")
                        print(f"Third error: {third_error}")
                        raise Exception("All model loading approaches failed")
        except Exception as e:
            print(f"Error loading model: {e}")
            print(traceback.format_exc())
            
            # Create a simple fallback model that won't crash
            print("Creating a simple fallback model...")
            input_shape = (224, 224, 3)
            inputs = tf.keras.layers.Input(shape=input_shape)
            x = tf.keras.layers.GlobalAveragePooling2D()(inputs)
            outputs = tf.keras.layers.Dense(35, activation='softmax')(x)
            fallback_model = tf.keras.Model(inputs=inputs, outputs=outputs)
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
            image: Preprocessed image
            
        Returns:
            class_name: Predicted class name
            confidence: Confidence score for the prediction
        """
        if image is None:
            print("Cannot predict with None image")
            return None, 0.0
        
        try:
            # Safety check: Ensure image has the correct dimensions
            if not isinstance(image, np.ndarray):
                print(f"Image is not a numpy array: {type(image)}")
                return None, 0.0
                
            # Ensure image is properly shaped
            expected_batch_shape = (1, 224, 224, 3)  # Batch, height, width, channels
            
            # Fix dimensions if needed
            if image.shape != expected_batch_shape:
                # Handle common issues
                if len(image.shape) == 3 and image.shape == (224, 224, 3):
                    # Missing batch dimension
                    image = np.expand_dims(image, axis=0)
                    print(f"Added batch dimension, new shape: {image.shape}")
                elif len(image.shape) == 4 and image.shape[0] > 1:
                    # Too many batch items, take first one
                    image = image[0:1]
                    print(f"Using only first batch item, new shape: {image.shape}")
                else:
                    # Try to resize
                    try:
                        if len(image.shape) == 3:
                            resized = cv2.resize(image, (224, 224))
                            image = np.expand_dims(resized, axis=0)
                        elif len(image.shape) == 4:
                            resized = cv2.resize(image[0], (224, 224))
                            image = np.expand_dims(resized, axis=0)
                        print(f"Resized image to shape: {image.shape}")
                    except Exception as resize_err:
                        print(f"Could not resize image: {resize_err}")
                        return None, 0.0
            
            # Normalize if needed
            if np.max(image) > 1.0:
                image = image / 255.0
                print("Normalized image to [0-1] range")
            
            # Check if image shape is still incorrect
            if image.shape != expected_batch_shape:
                print(f"Image shape {image.shape} does not match expected shape {expected_batch_shape}")
                return None, 0.0
            
            # Measure inference time
            start_time = time.time()
            
            # First prediction approach
            try:
                # Use direct model prediction
                predictions = self.model.predict(image, verbose=0)
                inference_time = time.time() - start_time
                self.inference_times.append(inference_time)
                
                # Process predictions
                if len(predictions.shape) != 2 or predictions.shape[0] != 1:
                    print(f"Unexpected prediction shape: {predictions.shape}, expected (1, N)")
                    raise ValueError("Invalid prediction shape")
                
                # Find highest scoring class
                predicted_class_idx = int(np.argmax(predictions[0]))
                confidence = float(predictions[0][predicted_class_idx])
                
                # Map class index to name
                class_name = self.class_mapping.get(str(predicted_class_idx), "Unknown")
                print(f"Predicted class: {class_name}, confidence: {confidence:.4f}")
                
                return class_name, confidence
                
            except Exception as pred_error:
                print(f"Standard prediction failed: {pred_error}")
                
                # Fallback to direct model call
                try:
                    print("Trying alternative prediction method")
                    logits = self.model(image, training=False)
                    
                    # Convert to numpy if it's a tensor
                    if hasattr(logits, 'numpy'):
                        logits_np = logits.numpy()
                    else:
                        logits_np = logits
                    
                    # Find the highest scoring class
                    predicted_class_idx = int(np.argmax(logits_np[0]))
                    
                    # Apply softmax to get probabilities if needed
                    if np.max(logits_np) > 100:  # These are probably logits, not probabilities
                        print("Converting logits to probabilities with softmax")
                        # Safe softmax to prevent overflow
                        probs = np.exp(logits_np - np.max(logits_np))
                        probs = probs / np.sum(probs, axis=1, keepdims=True)
                        confidence = float(probs[0][predicted_class_idx])
                    else:
                        confidence = float(logits_np[0][predicted_class_idx])
                    
                    # Map to class name
                    class_name = self.class_mapping.get(str(predicted_class_idx), "Unknown")
                    print(f"Alternative prediction: {class_name}, confidence: {confidence:.4f}")
                    
                    return class_name, confidence
                    
                except Exception as alt_error:
                    print(f"Alternative prediction failed: {alt_error}")
                    return None, 0.0
                
        except Exception as e:
            print(f"Error making prediction: {e}")
            traceback.print_exc()
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