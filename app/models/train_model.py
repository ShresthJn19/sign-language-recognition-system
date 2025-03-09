#!/usr/bin/env python3
# train_model.py - Train a sign language recognition model using MobileNetV3

import os
import argparse
import numpy as np
import tensorflow as tf
from tensorflow.keras.applications import MobileNetV3Small
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D, Dropout, BatchNormalization
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping, ReduceLROnPlateau
import matplotlib.pyplot as plt
import time
import platform
import ssl

# Handle SSL certificate issues (especially on macOS)
def configure_ssl():
    """Configure SSL certificate verification"""
    # Check if we're running on macOS
    if platform.system() == 'Darwin':
        # Check if PYTHONHTTPSVERIFY environment variable is set
        https_verify = os.environ.get('PYTHONHTTPSVERIFY', '1')
        if https_verify == '0':
            print("SSL certificate verification disabled via environment variable")
            ssl._create_default_https_context = ssl._create_unverified_context
            return True
            
        # Check if we're running in a virtual environment
        if hasattr(sys, 'real_prefix') or (hasattr(sys, 'base_prefix') and sys.base_prefix != sys.prefix):
            print("Running in a virtual environment on macOS - disabling SSL certificate verification")
            ssl._create_default_https_context = ssl._create_unverified_context
            return True
            
        # Prompt the user
        try:
            response = input("SSL certificate issues are common on macOS. Disable SSL verification? (y/n) [y]: ")
            if response.lower() in ['', 'y', 'yes']:
                print("Disabling SSL certificate verification")
                ssl._create_default_https_context = ssl._create_unverified_context
                return True
        except Exception:
            # If we can't get user input (e.g., in a script), default to disabling
            print("Running non-interactively on macOS - disabling SSL certificate verification")
            ssl._create_default_https_context = ssl._create_unverified_context
            return True
    
    return False

def create_model(num_classes, input_shape=(224, 224, 3)):
    """Create a MobileNetV3 model for sign language recognition"""
    # Create the weights directory if it doesn't exist
    weights_dir = os.path.join(os.path.dirname(__file__), 'weights')
    os.makedirs(weights_dir, exist_ok=True)
    weights_path = os.path.join(weights_dir, 'weights_mobilenet_v3_small_224_1.0_float_no_top_v2.h5')
    
    # Try to download weights if they don't exist
    if not os.path.exists(weights_path):
        print(f"Pre-trained weights not found at {weights_path}")
        print("Attempting to download weights (this may take a while)...")
        
        try:
            # Use an alternative method to download the weights if having SSL issues
            from tensorflow.keras.utils import get_file
            weights_url = "https://storage.googleapis.com/tensorflow/keras-applications/mobilenet_v3/weights_mobilenet_v3_small_224_1.0_float_no_top_v2.h5"
            get_file(
                weights_path,
                weights_url,
                cache_subdir='',
                cache_dir=weights_dir
            )
            print(f"Weights downloaded successfully to {weights_path}")
        except Exception as e:
            print(f"Failed to download weights: {e}")
            print("Will continue with randomly initialized weights")
            # Continue with randomly initialized weights if download fails
            weights_path = None
    
    # Load the MobileNetV3Small model with pre-trained weights
    base_model = MobileNetV3Small(
        weights=weights_path if os.path.exists(weights_path) else None,
        include_top=False,
        input_shape=input_shape
    )
    
    # Freeze the base model layers
    for layer in base_model.layers:
        layer.trainable = False
    
    # Add custom classification head
    x = base_model.output
    x = GlobalAveragePooling2D()(x)
    x = BatchNormalization()(x)
    x = Dense(256, activation='relu')(x)
    x = Dropout(0.5)(x)
    x = BatchNormalization()(x)
    x = Dense(128, activation='relu')(x)
    x = Dropout(0.3)(x)
    predictions = Dense(num_classes, activation='softmax')(x)
    
    model = Model(inputs=base_model.input, outputs=predictions)
    
    return model, base_model

def setup_data_generators(train_dir, val_dir, test_dir, img_size=(224, 224), batch_size=32):
    """Set up data generators for training, validation, and testing"""
    # Data augmentation for training
    train_datagen = ImageDataGenerator(
        rescale=1./255,
        rotation_range=20,
        width_shift_range=0.2,
        height_shift_range=0.2,
        shear_range=0.2,
        zoom_range=0.2,
        horizontal_flip=True,
        fill_mode='nearest'
    )
    
    # Only rescaling for validation and test sets
    val_datagen = ImageDataGenerator(rescale=1./255)
    test_datagen = ImageDataGenerator(rescale=1./255)
    
    # Flow from directory
    train_generator = train_datagen.flow_from_directory(
        train_dir,
        target_size=img_size,
        batch_size=batch_size,
        class_mode='categorical',
        shuffle=True
    )
    
    val_generator = val_datagen.flow_from_directory(
        val_dir,
        target_size=img_size,
        batch_size=batch_size,
        class_mode='categorical',
        shuffle=False
    )
    
    test_generator = test_datagen.flow_from_directory(
        test_dir,
        target_size=img_size,
        batch_size=batch_size,
        class_mode='categorical',
        shuffle=False
    )
    
    return train_generator, val_generator, test_generator

def fine_tune_model(model, base_model, train_generator, val_generator, model_dir, epochs=15):
    """Train the model in two phases: feature extraction and fine-tuning"""
    # First phase: train only the top layers
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
        loss='categorical_crossentropy',
        metrics=['accuracy']
    )
    
    # Set up callbacks
    checkpoint_dir = os.path.join(model_dir, 'checkpoints')
    os.makedirs(checkpoint_dir, exist_ok=True)
    
    callbacks = [
        ModelCheckpoint(
            os.path.join(checkpoint_dir, 'model_phase1_{epoch:02d}_{val_accuracy:.4f}.h5'),
            monitor='val_accuracy',
            save_best_only=True,
            mode='max'
        ),
        EarlyStopping(
            monitor='val_accuracy',
            patience=5,
            restore_best_weights=True
        ),
        ReduceLROnPlateau(
            monitor='val_loss',
            factor=0.2,
            patience=3,
            min_lr=1e-6
        )
    ]
    
    # Train the model
    print("Phase 1: Training top layers...")
    history_phase1 = model.fit(
        train_generator,
        epochs=epochs,
        validation_data=val_generator,
        callbacks=callbacks
    )
    
    # Second phase: fine-tuning - unfreeze the last few layers of the base model
    print("Phase 2: Fine-tuning the model...")
    # Unfreeze the last 30 layers
    for layer in base_model.layers[-30:]:
        layer.trainable = True
    
    # Recompile the model with a lower learning rate
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=1e-4),
        loss='categorical_crossentropy',
        metrics=['accuracy']
    )
    
    # Update callbacks for phase 2
    callbacks = [
        ModelCheckpoint(
            os.path.join(checkpoint_dir, 'model_phase2_{epoch:02d}_{val_accuracy:.4f}.h5'),
            monitor='val_accuracy',
            save_best_only=True,
            mode='max'
        ),
        EarlyStopping(
            monitor='val_accuracy',
            patience=8,
            restore_best_weights=True
        ),
        ReduceLROnPlateau(
            monitor='val_loss',
            factor=0.2,
            patience=3,
            min_lr=1e-7
        )
    ]
    
    # Continue training
    history_phase2 = model.fit(
        train_generator,
        epochs=epochs,
        validation_data=val_generator,
        callbacks=callbacks
    )
    
    return history_phase1, history_phase2, model

def plot_training_history(history1, history2, output_dir):
    """Plot the training and validation accuracy/loss curves"""
    # Combine histories
    acc = history1.history['accuracy'] + history2.history['accuracy']
    val_acc = history1.history['val_accuracy'] + history2.history['val_accuracy']
    loss = history1.history['loss'] + history2.history['loss']
    val_loss = history1.history['val_loss'] + history2.history['val_loss']
    
    epochs = range(1, len(acc) + 1)
    
    # Create figure directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    
    # Plot training & validation accuracy
    plt.figure(figsize=(12, 5))
    plt.subplot(1, 2, 1)
    plt.plot(epochs, acc, 'b', label='Training acc')
    plt.plot(epochs, val_acc, 'r', label='Validation acc')
    plt.axvline(x=len(history1.history['accuracy']), color='green', linestyle='--')
    plt.title('Training and validation accuracy')
    plt.legend()
    
    # Plot training & validation loss
    plt.subplot(1, 2, 2)
    plt.plot(epochs, loss, 'b', label='Training loss')
    plt.plot(epochs, val_loss, 'r', label='Validation loss')
    plt.axvline(x=len(history1.history['loss']), color='green', linestyle='--')
    plt.title('Training and validation loss')
    plt.legend()
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'training_history.png'))
    plt.close()

def evaluate_model(model, test_generator, model_dir):
    """Evaluate the model on the test set and save metrics"""
    print("Evaluating model on test set...")
    test_results = model.evaluate(test_generator)
    
    # Save test results
    with open(os.path.join(model_dir, 'test_results.txt'), 'w') as f:
        f.write(f"Test loss: {test_results[0]:.4f}\n")
        f.write(f"Test accuracy: {test_results[1]:.4f}\n")
    
    return test_results

def convert_to_onnx(model, model_dir):
    """Convert the TensorFlow model to ONNX format"""
    try:
        import tf2onnx
        import onnx
        
        # Create ONNX directory if it doesn't exist
        onnx_dir = os.path.join(model_dir, 'onnx')
        os.makedirs(onnx_dir, exist_ok=True)
        
        # Define the ONNX model path
        onnx_model_path = os.path.join(onnx_dir, 'sign_language_model.onnx')
        
        # Convert the model
        print("Converting model to ONNX format...")
        input_signature = [tf.TensorSpec([None, 224, 224, 3], tf.float32, name='input')]
        onnx_model, _ = tf2onnx.convert.from_keras(model, input_signature, opset=13)
        
        # Save the ONNX model
        onnx.save(onnx_model, onnx_model_path)
        print(f"ONNX model saved at: {onnx_model_path}")
    except ImportError:
        print("Warning: tf2onnx or onnx not installed. Skipping ONNX conversion.")
    except Exception as e:
        print(f"Error converting to ONNX: {e}")
        print("Continuing without ONNX conversion.")

def save_class_mapping(generator, model_dir):
    """Save the class indices mapping for later use in inference"""
    class_indices = generator.class_indices
    class_mapping = {v: k for k, v in class_indices.items()}
    
    # Save the mapping
    import json
    with open(os.path.join(model_dir, 'class_mapping.json'), 'w') as f:
        json.dump(class_mapping, f, indent=4)

def main():
    # Configure SSL for macOS systems
    ssl_configured = configure_ssl()
    
    parser = argparse.ArgumentParser(description='Train a sign language recognition model')
    parser.add_argument('--train_dir', type=str, default='app/data/processed/train',
                      help='Directory containing training data')
    parser.add_argument('--val_dir', type=str, default='app/data/processed/val',
                      help='Directory containing validation data')
    parser.add_argument('--test_dir', type=str, default='app/data/processed/test',
                      help='Directory containing test data')
    parser.add_argument('--model_dir', type=str, default='app/models/saved',
                      help='Directory to save the model')
    parser.add_argument('--epochs', type=int, default=15,
                      help='Number of epochs for each training phase')
    parser.add_argument('--batch_size', type=int, default=32,
                      help='Batch size for training')
    parser.add_argument('--img_size', type=int, default=224,
                      help='Image size for training')
    parser.add_argument('--disable_ssl_verify', action='store_true',
                      help='Disable SSL certificate verification')
    
    args = parser.parse_args()
    
    # Check if SSL verification was requested to be disabled
    if args.disable_ssl_verify and not ssl_configured:
        print("Disabling SSL certificate verification as requested")
        ssl._create_default_https_context = ssl._create_unverified_context
    
    # Create model directory if it doesn't exist
    os.makedirs(args.model_dir, exist_ok=True)
    
    # Setup data generators
    img_size = (args.img_size, args.img_size)
    try:
        train_generator, val_generator, test_generator = setup_data_generators(
            args.train_dir, args.val_dir, args.test_dir, 
            img_size=img_size, 
            batch_size=args.batch_size
        )
        
        # Get the number of classes
        num_classes = len(train_generator.class_indices)
        print(f"Number of classes: {num_classes}")
        
        # Create the model
        model, base_model = create_model(num_classes, input_shape=(*img_size, 3))
        
        # Print model summary
        model.summary()
        
        # Train the model
        start_time = time.time()
        history1, history2, trained_model = fine_tune_model(
            model, base_model, train_generator, val_generator, 
            args.model_dir, epochs=args.epochs
        )
        
        # Plot training history
        figures_dir = os.path.join(args.model_dir, 'figures')
        plot_training_history(history1, history2, figures_dir)
        
        # Evaluate the model
        test_results = evaluate_model(trained_model, test_generator, args.model_dir)
        
        # Save the final model
        final_model_path = os.path.join(args.model_dir, 'sign_language_model.h5')
        trained_model.save(final_model_path)
        print(f"Final model saved at: {final_model_path}")
        
        # Save class mapping
        save_class_mapping(train_generator, args.model_dir)
        
        # Convert to ONNX
        convert_to_onnx(trained_model, args.model_dir)
        
        # Print total training time
        total_time = time.time() - start_time
        hours, remainder = divmod(total_time, 3600)
        minutes, seconds = divmod(remainder, 60)
        print(f"Total training time: {int(hours)}h {int(minutes)}m {int(seconds)}s")
        
        # Print final results
        print(f"Final test accuracy: {test_results[1]:.4f}")
        
    except FileNotFoundError as e:
        print(f"ERROR: Dataset not found - {e}")
        print("Make sure you've run the dataset splitting script:")
        print("  python scripts/split_dataset.py")
        return
    except Exception as e:
        print(f"ERROR: {e}")
        import traceback
        traceback.print_exc()
        return

if __name__ == "__main__":
    import sys
    main() 