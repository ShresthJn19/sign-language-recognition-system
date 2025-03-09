#!/usr/bin/env python3
# main.py - FastAPI backend for the sign language recognition system

import os
import sys
import cv2
import numpy as np
import base64
import json
import time
import asyncio
import traceback
from typing import Dict, Optional, List
import uvicorn
from fastapi import FastAPI, Request, Response, status, Form, Body
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from contextlib import asynccontextmanager
from fastapi.middleware.cors import CORSMiddleware
import logging

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler()
    ]
)
logger = logging.getLogger("sign_language_app")

# Add parent directory to path for imports
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Import local modules
try:
    from utils.hand_detector import HandDetector
    from utils.text_to_speech import TextToSpeech
    from models.sign_language_model import SignLanguageModel
except ImportError as e:
    logger.error(f"Error importing modules: {e}")
    traceback.print_exc()
    sys.exit(1)

# Setup global variables for model and utilities
model = None
hand_detector = None
tts = None

@asynccontextmanager
async def lifespan(app: FastAPI):
    # Initialize model and utilities at startup
    global model, hand_detector, tts
    logger.info("Initializing model and utilities...")
    
    try:
        # Create model directory if it doesn't exist
        os.makedirs("app/models/saved", exist_ok=True)
        
        # Initialize hand detector with more robust error handling
        try:
            hand_detector = HandDetector(static_image_mode=False, min_detection_confidence=0.7)
            logger.info("Hand detector initialized successfully")
        except Exception as he:
            logger.error(f"Failed to initialize hand detector: {he}")
            logger.debug(traceback.format_exc())
            hand_detector = None
        
        # Initialize text-to-speech engine with error handling
        try:
            tts = TextToSpeech(rate=150, volume=1.0)
            logger.info("Text-to-speech engine initialized successfully")
        except Exception as te:
            logger.error(f"Failed to initialize text-to-speech: {te}")
            logger.debug(traceback.format_exc())
            tts = None
        
        # Check if model exists, if not, print a warning
        model_path = "app/models/saved/sign_language_model.h5"
        if os.path.exists(model_path):
            # Initialize model with better error handling
            try:
                logger.info(f"Loading model from {model_path}")
                model = SignLanguageModel(model_dir="app/models/saved")
                logger.info("Model loaded successfully")
                
                # Perform a test prediction with better error handling
                try:
                    logger.info("Testing model with dummy input...")
                    # Create a valid test image with the right shape and normalization
                    dummy_input = np.zeros((224, 224, 3), dtype=np.float32)
                    class_name, confidence = model.predict(dummy_input)
                    
                    if class_name is not None:
                        logger.info(f"Model test prediction successful: {class_name} with confidence {confidence:.4f}")
                    else:
                        logger.warning("Model loaded but test prediction returned None")
                        logger.warning("Model may work with real input data, continuing anyway")
                except Exception as pe:
                    logger.error(f"Model test prediction failed: {pe}")
                    logger.debug(traceback.format_exc())
                    logger.warning("Using model anyway - it may work with real input data")
            except Exception as e:
                logger.error(f"Failed to load model: {e}")
                logger.debug(traceback.format_exc())
                logger.warning("Attempting to create a dummy model to prevent application crash")
                try:
                    # Last resort - create a simple model
                    logger.warning("Creating a simple fallback model")
                    import tensorflow as tf
                    input_shape = (224, 224, 3)
                    inputs = tf.keras.layers.Input(shape=input_shape)
                    x = tf.keras.layers.GlobalAveragePooling2D()(inputs)
                    outputs = tf.keras.layers.Dense(35, activation='softmax')(x)
                    model = tf.keras.Model(inputs=inputs, outputs=outputs)
                    
                    # Just to be safe, define the class mapping
                    model.class_mapping = {str(i): chr(i + 65) if i < 26 else str(i - 26 + 1) for i in range(35)}
                    
                    # Simple stabilize_prediction and visualize_prediction functions
                    def stabilize_prediction(cls, conf):
                        return cls, conf
                    
                    def visualize_prediction(frame, cls, conf):
                        return frame
                    
                    # Add these methods to the model
                    model.stabilize_prediction = stabilize_prediction
                    model.visualize_prediction = visualize_prediction
                    model.inference_times = []
                    model.predict = lambda x: (None, 0.0)
                except Exception as fallback_error:
                    logger.error(f"Failed to create fallback model: {fallback_error}")
                    model = None
        else:
            logger.warning(f"Model file not found at {model_path}")
            logger.info("To use the recognition feature, first train the model with:")
            logger.info("  python app/models/train_model.py")
            model = None
            
        logger.info("Initialization complete")
            
    except Exception as e:
        logger.error(f"ERROR during initialization: {e}")
        logger.debug(traceback.format_exc())
        # We don't exit here, as we want the app to start even with errors
        # This allows the frontend to load and display appropriate messages
    
    yield
    
    # Cleanup resources at shutdown
    logger.info("Cleaning up resources...")
    
    try:
        # Clean up MediaPipe resources if hand detector was initialized
        if hand_detector is not None and hasattr(hand_detector, 'hands'):
            hand_detector.hands.close()
            logger.info("Hand detector resources cleaned up")
    except Exception as ce:
        logger.error(f"Error during cleanup: {ce}")

# Create FastAPI app
app = FastAPI(
    title="Sign Language Recognition",
    description="Real-time sign language recognition with FastAPI and TensorFlow",
    version="1.0.0",
    lifespan=lifespan
)

# CORS settings to allow browser access
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Allow all origins
    allow_credentials=True,
    allow_methods=["*"],  # Allow all methods
    allow_headers=["*"],  # Allow all headers
)

# Custom middleware to handle malformed requests
@app.middleware("http")
async def log_and_handle_errors(request: Request, call_next):
    # Record request start time
    start_time = time.time()
    
    try:
        # Process the request
        response = await call_next(request)
        
        # Log slow requests (useful for optimization)
        process_time = time.time() - start_time
        if process_time > 1.0:  # Log requests taking more than 1 second
            logger.warning(f"Slow request: {request.method} {request.url.path} took {process_time:.2f}s")
        
        return response
    except Exception as e:
        # Log detailed error information
        logger.error(f"Request error: {request.method} {request.url.path} - {str(e)}")
        logger.debug(f"Request headers: {request.headers}")
        logger.debug(traceback.format_exc())
        
        # Return a JSON error response
        return JSONResponse(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            content={"error": str(e), "path": request.url.path}
        )

# Mount static files
try:
    app.mount("/static", StaticFiles(directory="app/frontend/static"), name="static")
    logger.info("Static files mounted successfully")
except Exception as e:
    logger.error(f"ERROR mounting static files: {e}")
    logger.debug(traceback.format_exc())

# Setup Jinja2 templates
try:
    templates = Jinja2Templates(directory="app/frontend/templates")
    logger.info("Templates initialized successfully")
except Exception as e:
    logger.error(f"ERROR initializing templates: {e}")
    logger.debug(traceback.format_exc())

# Serve the HTML frontend
@app.get("/", response_class=HTMLResponse)
async def get_index(request: Request):
    try:
        return templates.TemplateResponse("index.html", {"request": request})
    except Exception as e:
        logger.error(f"ERROR rendering template: {e}")
        logger.debug(traceback.format_exc())
        return HTMLResponse(content=f"<html><body><h1>Error loading template</h1><p>{str(e)}</p></body></html>")

# API endpoint to check system status
@app.get("/api/status")
async def get_status():
    return {
        "status": "ok",
        "model_loaded": model is not None,
        "hand_detector_loaded": hand_detector is not None,
        "tts_loaded": tts is not None
    }

# HTTP endpoint for frame processing with better error handling for malformed requests
@app.post("/api/process-frame")
async def http_process_frame(request: Request):
    try:
        # Get the body with a timeout to prevent hanging on malformed requests
        body_bytes = await asyncio.wait_for(request.body(), timeout=2.0)
        
        # Check if the body is empty
        if not body_bytes:
            logger.warning("Empty request body received")
            return JSONResponse(
                content={"error": "Empty request body", "timestamp": time.time()},
                status_code=400
            )
            
        # Try to decode the body as UTF-8 text
        try:
            body_str = body_bytes.decode('utf-8')
        except UnicodeDecodeError:
            logger.warning("Received binary data instead of text")
            return JSONResponse(
                content={"error": "Invalid data format. Expected text data", "timestamp": time.time()},
                status_code=400
            )
        
        # Process the frame
        processed_data = await process_frame(body_str)
        
        # Return the processed result
        return JSONResponse(content=processed_data)
    except asyncio.TimeoutError:
        logger.warning("Request timed out while reading body")
        return JSONResponse(
            content={"error": "Request timed out", "timestamp": time.time()},
            status_code=408
        )
    except Exception as e:
        logger.error(f"Error processing frame via HTTP: {e}")
        logger.debug(traceback.format_exc())
        return JSONResponse(
            content={"error": str(e), "timestamp": time.time()},
            status_code=400
        )

# Endpoint to get available TTS voices
@app.get("/api/voices")
async def get_voices():
    if tts is None:
        return JSONResponse({"error": "TTS engine not initialized"}, status_code=500)
    
    try:
        voices = tts.get_available_voices()
        return {"voices": voices}
    except Exception as e:
        logger.error(f"Error getting voices: {e}")
        logger.debug(traceback.format_exc())
        return JSONResponse({"error": str(e)}, status_code=500)

# Endpoint to set TTS voice
@app.post("/api/voices/{voice_id}")
async def set_voice(voice_id: str):
    if tts is None:
        return JSONResponse({"error": "TTS engine not initialized"}, status_code=500)
    
    try:
        tts.set_voice(voice_id)
        return {"success": True, "voice_id": voice_id}
    except Exception as e:
        logger.error(f"Error setting voice: {e}")
        logger.debug(traceback.format_exc())
        return JSONResponse({"error": str(e)}, status_code=400)

# Endpoint to speak text
@app.post("/api/speak")
async def speak_text(data: dict):
    if tts is None:
        return JSONResponse({"error": "TTS engine not initialized"}, status_code=500)
    
    try:
        text = data.get("text", "")
        if text:
            tts.speak(text)
        return {"success": True}
    except Exception as e:
        logger.error(f"Error speaking text: {e}")
        logger.debug(traceback.format_exc())
        return JSONResponse({"error": str(e)}, status_code=500)

# Process a frame from the client
async def process_frame(data):
    try:
        # Extract the base64 image data
        try:
            if "," not in data:
                logger.warning("Invalid data format - no comma found")
                return {
                    "error": "Invalid data format. Expected 'data:image/...,BASE64DATA'",
                    "status": "no_hand",
                    "timestamp": time.time()
                }
                
            _, base64_data = data.split(",", 1)
        except ValueError as e:
            logger.error(f"Invalid data format: {str(e)}")
            return {
                "error": "Invalid data format. Expected 'data:image/...,BASE64DATA'",
                "status": "no_hand",
                "timestamp": time.time()
            }
        
        # Decode the base64 image
        try:
            image_bytes = base64.b64decode(base64_data)
            nparr = np.frombuffer(image_bytes, np.uint8)
            frame = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        except Exception as e:
            logger.error(f"Error decoding image: {str(e)}")
            return {
                "error": f"Error decoding image: {str(e)}",
                "status": "no_hand",
                "timestamp": time.time()
            }
        
        # Check if frame was properly decoded
        if frame is None or frame.size == 0:
            logger.error("Error: Received empty or corrupt image frame")
            return {
                "error": "Invalid image data received",
                "status": "no_hand",
                "timestamp": time.time()
            }
        
        # Get frame dimensions for debugging
        h, w = frame.shape[:2]
        logger.debug(f"Processing frame with dimensions: {w}x{h}")
        
        # Check if hand detector is available
        if hand_detector is None:
            logger.error("Hand detector not initialized")
            return {
                "error": "Hand detector not available",
                "status": "error",
                "timestamp": time.time()
            }
            
        # Process the frame with the hand detector
        try:
            processed_frame, hand_imgs = hand_detector.process_frame(frame, draw=True)
            
            # Ensure we have a valid processed frame
            if processed_frame is None:
                processed_frame = frame
                logger.warning("Hand detector returned None for processed_frame, using original")
        except Exception as hd_error:
            logger.error(f"Hand detector error: {hd_error}")
            processed_frame = frame
            hand_imgs = []
        
        # Initialize predictions list
        predictions = []
        
        # Process each detected hand
        for idx, hand_img in enumerate(hand_imgs):
            if model is not None:
                try:
                    # Preprocess the hand image
                    preprocessed_img = preprocess_hand_image(hand_img)
                    
                    if preprocessed_img is not None:
                        # Make prediction
                        class_name, confidence = model.predict(preprocessed_img)
                        
                        if class_name is not None:
                            # Stabilize the prediction
                            stable_class, stable_confidence = model.stabilize_prediction(class_name, confidence)
                            
                            # Add to predictions list
                            predictions.append({
                                "hand_index": idx,
                                "prediction": stable_class,
                                "confidence": float(stable_confidence) if stable_confidence else 0.0
                            })
                except Exception as pred_error:
                    logger.error(f"Error making prediction for hand {idx}: {pred_error}")
        
        # Encode the processed frame back to base64
        try:
            _, buffer = cv2.imencode(".jpg", processed_frame)
            processed_base64 = base64.b64encode(buffer).decode("utf-8")
        except Exception as enc_error:
            logger.error(f"Error encoding frame: {enc_error}")
            return {
                "error": "Error processing image",
                "status": "error",
                "timestamp": time.time()
            }
        
        # Determine status
        status = "ok" if hand_imgs else "no_hand"
            
        # Return the processed data
        return {
            "image": f"data:image/jpeg;base64,{processed_base64}",
            "predictions": predictions,
            "status": status,
            "timestamp": time.time()
        }
    except Exception as e:
        logger.error(f"Error processing frame: {e}")
        logger.debug(traceback.format_exc())
        return {
            "error": str(e),
            "status": "error",
            "timestamp": time.time()
        }

def preprocess_hand_image(hand_img):
    """Preprocess a hand image for model prediction"""
    try:
        if hand_img is None:
            logger.warning("Cannot preprocess None image")
            return None
            
        # Ensure the image has the right shape (224, 224, 3)
        if hand_img.shape != (224, 224, 3):
            hand_img = cv2.resize(hand_img, (224, 224))
        
        # Ensure the image is in float32 format and normalized to [0, 1]
        if hand_img.dtype != np.float32:
            if np.max(hand_img) > 1.0:
                hand_img = hand_img.astype(np.float32) / 255.0
            else:
                hand_img = hand_img.astype(np.float32)
        
        # Add batch dimension
        hand_img = np.expand_dims(hand_img, axis=0)
        
        return hand_img
        
    except Exception as e:
        logger.error(f"Error preprocessing hand image: {e}")
        return None

# Main entry point
if __name__ == "__main__":
    uvicorn.run(
        "app.backend.main:app", 
        host="0.0.0.0", 
        port=8000, 
        reload=True,
        log_level="info"
    ) 