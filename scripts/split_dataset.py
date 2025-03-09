#!/usr/bin/env python3
# split_dataset.py - Script to split sign language dataset into train, val, and test sets

import os
import shutil
import random
import argparse
from pathlib import Path
import numpy as np
from tqdm import tqdm

# Define train, validation, test split ratios
TRAIN_RATIO = 0.7
VAL_RATIO = 0.15
TEST_RATIO = 0.15

def create_directories(base_dir):
    """Create the necessary directories for the dataset splits"""
    # Create main directories
    os.makedirs(os.path.join(base_dir, "train"), exist_ok=True)
    os.makedirs(os.path.join(base_dir, "val"), exist_ok=True)
    os.makedirs(os.path.join(base_dir, "test"), exist_ok=True)
    
    # Create class subdirectories
    src_dir = os.path.join(os.getcwd(), "data")
    classes = [d for d in os.listdir(src_dir) if os.path.isdir(os.path.join(src_dir, d))]
    
    for split in ["train", "val", "test"]:
        for cls in classes:
            os.makedirs(os.path.join(base_dir, split, cls), exist_ok=True)
    
    return classes

def split_dataset(src_dir, dest_dir, seed=42):
    """Split the dataset into train, validation and test sets"""
    random.seed(seed)
    np.random.seed(seed)
    
    # Get all class directories
    classes = [d for d in os.listdir(src_dir) if os.path.isdir(os.path.join(src_dir, d))]
    
    # Create the output directories
    split_dirs = create_directories(dest_dir)
    
    print(f"Found {len(classes)} classes. Starting split...")
    
    # Process each class directory
    for cls in tqdm(classes, desc="Processing classes"):
        # Get all images for this class
        cls_dir = os.path.join(src_dir, cls)
        images = [f for f in os.listdir(cls_dir) if f.endswith('.jpg')]
        random.shuffle(images)
        
        # Calculate split sizes
        total_images = len(images)
        train_size = int(total_images * TRAIN_RATIO)
        val_size = int(total_images * VAL_RATIO)
        
        # Split images into sets
        train_images = images[:train_size]
        val_images = images[train_size:train_size + val_size]
        test_images = images[train_size + val_size:]
        
        # Copy files to their respective directories
        for img in train_images:
            shutil.copy2(
                os.path.join(cls_dir, img),
                os.path.join(dest_dir, "train", cls, img)
            )
        
        for img in val_images:
            shutil.copy2(
                os.path.join(cls_dir, img),
                os.path.join(dest_dir, "val", cls, img)
            )
        
        for img in test_images:
            shutil.copy2(
                os.path.join(cls_dir, img),
                os.path.join(dest_dir, "test", cls, img)
            )
        
        print(f"Class {cls}: {len(train_images)} train, {len(val_images)} val, {len(test_images)} test")

def main():
    parser = argparse.ArgumentParser(description='Split sign language dataset into train, val, and test sets')
    parser.add_argument('--src', type=str, default='data',
                        help='Source directory containing the original dataset')
    parser.add_argument('--dest', type=str, default='app/data/processed',
                        help='Destination directory for the split dataset')
    parser.add_argument('--seed', type=int, default=42,
                        help='Random seed for reproducibility')
    
    args = parser.parse_args()
    
    src_dir = os.path.abspath(args.src)
    dest_dir = os.path.abspath(args.dest)
    
    print(f"Source directory: {src_dir}")
    print(f"Destination directory: {dest_dir}")
    
    # Split the dataset
    split_dataset(src_dir, dest_dir, args.seed)
    
    print("Dataset split completed successfully")
    print(f"Train set: {os.path.join(dest_dir, 'train')}")
    print(f"Validation set: {os.path.join(dest_dir, 'val')}")
    print(f"Test set: {os.path.join(dest_dir, 'test')}")

if __name__ == "__main__":
    main() 