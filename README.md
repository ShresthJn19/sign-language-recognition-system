# Real-Time Sign Language Recognition System

A Python-based real-time sign language recognition system using computer vision and deep learning. This system recognizes sign language gestures from a live video feed and converts them to text and speech.

## Features

- Real-time sign language recognition for the American Sign Language (ASL) alphabet and numbers
- Web-based interface with live video capture
- Text and speech output of recognized signs
- High-accuracy deep learning model using MobileNetV3

## Project Structure

```
.
├── app/                    # Main application directory
│   ├── backend/            # FastAPI backend server
│   ├── frontend/           # Web frontend (HTML/CSS/JS)
│   ├── models/             # Model training and inference code
│   ├── data/               # Data processing and storage
│   │   ├── raw/            # Original dataset
│   │   └── processed/      # Processed and split dataset
│   │       ├── train/      # Training set
│   │       ├── val/        # Validation set
│   │       └── test/       # Test set
│   └── utils/              # Utility functions
├── scripts/                # Helper scripts
│   └── split_dataset.py    # Script to split dataset
├── setup.py                # Setup script
├── run.py                  # Run application script
├── requirements.txt        # Project dependencies
└── README.md               # Project documentation
```

## Getting Started

### Easiest Way: Using the Setup Script

We've created a setup script that automates the entire process:

```bash
# Run the setup script
python setup.py

# To run the setup script and immediately start the application
python setup.py --run
```

The setup script will:

1. Check Python version
2. Create a virtual environment
3. Install dependencies
4. Prepare the dataset
5. Train the model
6. Start the application (if --run is specified)

### Manual Setup

If you prefer to set up manually:

1. Clone the repository:

   ```
   git clone <repository-url>
   cd sign-language-recognition
   ```

2. Create and activate a virtual environment:

   ```
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

3. Install the required dependencies:

   ```
   pip install -r requirements.txt
   ```

4. Make sure the dataset is in the correct location:

   ```
   # Dataset should be in the 'data' directory with subdirectories for each letter/number
   ```

5. Prepare the dataset:

   ```
   python scripts/split_dataset.py
   ```

6. Train the model:

   ```
   python app/models/train_model.py
   ```

7. Start the application:

   ```
   python run.py
   ```

8. Open your browser and navigate to:
   ```
   http://localhost:8000
   ```

## Troubleshooting

### SSL Certificate Verification Error

If you get an SSL certificate verification error when training the model (especially on macOS), use one of these solutions:

#### Option 1: Run with SSL verification disabled (easiest)

```bash
# macOS/Linux
PYTHONHTTPSVERIFY=0 python app/models/train_model.py

# Windows PowerShell
$env:PYTHONHTTPSVERIFY=0; python app/models/train_model.py
```

#### Option 2: Install certificates for your Python installation

For macOS, you can run the Install Certificates.command script that comes with Python:

```bash
# Find your Python installation directory
python -c "import os, sys; print(os.path.dirname(sys.executable))"

# Run the certificates script
cd /path/to/your/python/directory
./Install\ Certificates.command
```

#### Option 3: Use the setup script

Our setup script automatically handles SSL certificate issues on macOS:

```bash
python setup.py
```

### Application Fails to Start

If the application fails to start:

1. Make sure you've trained the model first:

   ```bash
   python app/models/train_model.py
   ```

2. Check for error messages in the console.

3. The application will still run even without a trained model, but it will display a warning and the recognition features won't work.

### Camera Access Issues

If the application can't access your camera:

1. Make sure your browser has permissions to access the camera.
2. Try using a different browser (Chrome or Firefox recommended).
3. Ensure no other application is using the camera.

## Development

### Train Your Own Model

To train the sign language recognition model on your own dataset:

1. Place your dataset in `data/` with subdirectories for each sign
2. Run the dataset splitting script:
   ```
   python scripts/split_dataset.py
   ```
3. Train the model using:
   ```
   python app/models/train_model.py
   ```

### Project Components

1. **Backend**: FastAPI server that handles video processing and model inference
2. **Frontend**: Web interface for displaying the video feed and recognition results
3. **Model Training**: Scripts for preprocessing data and training the sign language recognition model
4. **Utilities**: Helper functions for data processing, visualization, and evaluation

## Technologies Used

- **Web Framework**: FastAPI, Socket.IO
- **Computer Vision**: OpenCV, Mediapipe
- **Deep Learning**: TensorFlow, MobileNetV3, ONNX
- **Data Processing**: NumPy, Pandas, scikit-learn
- **Text to Speech**: pyttsx3

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Acknowledgements

- Dataset: Indian Sign Language alphabet and numbers dataset
- [MediaPipe](https://github.com/google/mediapipe) for hand landmark detection
- [TensorFlow](https://www.tensorflow.org/) for deep learning framework
