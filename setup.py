#!/usr/bin/env python3
# setup.py - Setup script for the sign language recognition system

import os
import sys
import argparse
import subprocess
import platform
import shutil
from pathlib import Path

def check_python_version():
    """Check if Python version is 3.11 or higher"""
    if sys.version_info < (3, 11):
        print("ERROR: Python 3.11 or higher is required")
        print(f"Current version: {platform.python_version()}")
        return False
    return True

def create_virtual_environment():
    """Create a virtual environment if it doesn't exist"""
    if os.path.exists("venv"):
        print("✅ Virtual environment already exists")
        return True
    
    try:
        print("Creating virtual environment...")
        subprocess.run([sys.executable, "-m", "venv", "venv"], check=True)
        print("✅ Virtual environment created")
        return True
    except subprocess.CalledProcessError:
        print("❌ Failed to create virtual environment")
        return False

def install_requirements():
    """Install requirements in the virtual environment"""
    pip_path = os.path.join("venv", "bin", "pip") if os.name != "nt" else os.path.join("venv", "Scripts", "pip.exe")
    
    if not os.path.exists(pip_path):
        print("❌ Virtual environment not found")
        return False
    
    try:
        print("Installing requirements...")
        subprocess.run([pip_path, "install", "-r", "requirements.txt"], check=True)
        print("✅ Requirements installed")
        return True
    except subprocess.CalledProcessError:
        print("❌ Failed to install requirements")
        return False

def check_dataset():
    """Check if the dataset exists"""
    # Check raw data
    raw_data_path = os.path.join("data")
    if not os.path.exists(raw_data_path):
        print("❌ Dataset not found in 'data' directory")
        return False
    
    # Check if dataset has letter directories
    letters = [chr(i) for i in range(ord('A'), ord('Z')+1)]
    numbers = [str(i) for i in range(1, 10)]
    classes = letters + numbers
    
    missing_classes = []
    for cls in classes:
        class_path = os.path.join(raw_data_path, cls)
        if not os.path.exists(class_path):
            missing_classes.append(cls)
    
    if missing_classes:
        print(f"❌ Missing classes in dataset: {', '.join(missing_classes)}")
        return False
    
    print("✅ Dataset found")
    return True

def setup_data_directories():
    """Setup data directories for processed data"""
    os.makedirs("app/data/raw", exist_ok=True)
    os.makedirs("app/data/processed/train", exist_ok=True)
    os.makedirs("app/data/processed/val", exist_ok=True)
    os.makedirs("app/data/processed/test", exist_ok=True)
    os.makedirs("app/models/saved", exist_ok=True)
    
    # Print message
    print("✅ Data directories created")
    return True

def split_dataset():
    """Run the dataset splitting script"""
    print("Splitting dataset...")
    try:
        python_exec = os.path.join("venv", "bin", "python") if os.name != "nt" else os.path.join("venv", "Scripts", "python.exe")
        subprocess.run([python_exec, "scripts/split_dataset.py"], check=True)
        print("✅ Dataset split")
        return True
    except subprocess.CalledProcessError:
        print("❌ Failed to split dataset")
        return False

def train_model():
    """Train the model"""
    print("Training model...")
    try:
        python_exec = os.path.join("venv", "bin", "python") if os.name != "nt" else os.path.join("venv", "Scripts", "python.exe")
        
        if platform.system() == 'Darwin':  # macOS
            # Create an SSL context to avoid certificate verification failures
            print("Detected macOS: Setting SSL certificate verification to none...")
            os.environ['SSL_CERT_FILE'] = ''
            os.environ['PYTHONHTTPSVERIFY'] = '0'
        
        subprocess.run([python_exec, "app/models/train_model.py"], check=True)
        print("✅ Model trained")
        return True
    except subprocess.CalledProcessError:
        print("❌ Failed to train model")
        return False

def copy_dataset():
    """Copy dataset to app/data/raw if necessary"""
    source_dir = os.path.join("data")
    target_dir = os.path.join("app", "data", "raw")
    
    if not os.path.exists(source_dir):
        print("❌ Dataset not found in 'data' directory")
        return False
    
    # Check if dataset already copied
    if os.path.exists(target_dir) and any(os.scandir(target_dir)):
        print("✅ Dataset already in app/data/raw")
        return True
    
    print("Copying dataset to app/data/raw...")
    try:
        # Create target directory if it doesn't exist
        os.makedirs(target_dir, exist_ok=True)
        
        # Copy each class directory
        for item in os.listdir(source_dir):
            source_item = os.path.join(source_dir, item)
            target_item = os.path.join(target_dir, item)
            
            if os.path.isdir(source_item):
                if os.path.exists(target_item):
                    shutil.rmtree(target_item)
                shutil.copytree(source_item, target_item)
        
        print("✅ Dataset copied to app/data/raw")
        return True
    except Exception as e:
        print(f"❌ Failed to copy dataset: {e}")
        return False

def run_app():
    """Run the application"""
    print("Starting application...")
    try:
        python_exec = os.path.join("venv", "bin", "python") if os.name != "nt" else os.path.join("venv", "Scripts", "python.exe")
        subprocess.run([python_exec, "run.py"], check=False)
        return True
    except subprocess.CalledProcessError:
        print("❌ Failed to start application")
        return False
    except KeyboardInterrupt:
        print("\nApplication stopped")
        return True

def main():
    parser = argparse.ArgumentParser(description='Setup the sign language recognition system')
    parser.add_argument('--skip-venv', action='store_true', help='Skip virtual environment creation')
    parser.add_argument('--skip-install', action='store_true', help='Skip requirements installation')
    parser.add_argument('--skip-dataset', action='store_true', help='Skip dataset preparation')
    parser.add_argument('--skip-train', action='store_true', help='Skip model training')
    parser.add_argument('--run', action='store_true', help='Run the application after setup')
    
    args = parser.parse_args()
    
    print("=" * 60)
    print("Sign Language Recognition System Setup")
    print("=" * 60)
    
    # Check Python version
    if not check_python_version():
        return
    
    # Create virtual environment
    if not args.skip_venv and not create_virtual_environment():
        return
    
    # Install requirements
    if not args.skip_install and not install_requirements():
        return
    
    # Setup directories
    setup_data_directories()
    
    # Check and copy dataset
    if not args.skip_dataset:
        if check_dataset():
            if not copy_dataset():
                return
            
            if not split_dataset():
                return
    
    # Train model
    if not args.skip_train and not args.skip_dataset:
        if not train_model():
            print("Note: You can still run the application without a trained model,")
            print("but the recognition features will not work.")
    
    print("=" * 60)
    print("Setup complete!")
    print("=" * 60)
    print("To run the application:")
    print("  python run.py")
    print("=" * 60)
    
    # Run the application if requested
    if args.run:
        run_app()

if __name__ == "__main__":
    main() 