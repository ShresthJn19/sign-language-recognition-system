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
        
        # Initialize hand detector
        hand_detector = HandDetector(static_image_mode=False, min_detection_confidence=0.7)
        logger.info("Hand detector initialized successfully")
        
        # Initialize text-to-speech engine
        tts = TextToSpeech(rate=150, volume=1.0)
        logger.info("Text-to-speech engine initialized successfully")
        
        # Check if model exists, if not, print a warning
        model_path = "app/models/saved/sign_language_model.h5"
        if os.path.exists(model_path):
            # Initialize model
            try:
                model = SignLanguageModel(model_dir="app/models/saved")
                logger.info("Model loaded successfully")
            except Exception as e:
                logger.warning(f"Failed to load model: {e}")
                logger.debug(traceback.format_exc())
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

# Middleware to handle invalid requests
@app.middleware("http")
async def catch_exceptions_middleware(request: Request, call_next):
    try:
        return await call_next(request)
    except Exception as e:
        logger.error(f"Request error: {e}")
        logger.debug(traceback.format_exc())
        return JSONResponse(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            content={"error": str(e)}
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

# HTTP endpoint for frame processing
@app.post("/api/process-frame")
async def http_process_frame(request: Request):
    try:
        # Get the body as text
        body_bytes = await request.body()
        body_str = body_bytes.decode('utf-8')
        
        # Process the frame
        processed_data = await process_frame(body_str)
        
        # Return the processed result
        return JSONResponse(content=processed_data)
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
            _, base64_data = data.split(",", 1)
        except ValueError as e:
            logger.error(f"Invalid data format: {str(e)}")
            return {
                "error": "Invalid data format. Expected 'data:image/...,BASE64DATA'",
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
                "timestamp": time.time()
            }
        
        # Check if frame was properly decoded
        if frame is None or frame.size == 0:
            logger.error("Error: Received empty or corrupt image frame")
            return {
                "error": "Invalid image data received",
                "timestamp": time.time()
            }
        
        # Get frame dimensions for debugging
        h, w = frame.shape[:2]
        logger.debug(f"Processing frame with dimensions: {w}x{h}")
        
        # Process the frame with the hand detector
        processed_frame, processed_img = hand_detector.process_frame(frame, draw=True)
        
        # Make a prediction if a hand is detected
        if processed_img is not None and model is not None:
            class_name, confidence = model.predict(processed_img)
            
            # Stabilize the prediction
            stable_class, stable_confidence = model.stabilize_prediction(class_name, confidence)
            
            # Visualize the prediction on the frame
            processed_frame = model.visualize_prediction(processed_frame, stable_class, stable_confidence)
            
            # Convert the prediction to speech
            if stable_class is not None and stable_confidence > 0.7 and tts is not None:
                # Convert letter/number to string for speech
                text_to_speak = stable_class
                
                # Speak the text
                tts.speak(text_to_speak)
        else:
            stable_class = None
            stable_confidence = 0.0
            # If no hand detected or error in processing, add text to the frame
            if processed_frame is not None:
                cv2.putText(processed_frame, "No hand detected", (10, 30), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 0, 255), 2)
        
        # Ensure we have a valid frame to return
        if processed_frame is None:
            logger.warning("Warning: No processed frame available. Using original frame.")
            processed_frame = frame
        
        # Encode the processed frame back to base64
        try:
            _, buffer = cv2.imencode(".jpg", processed_frame)
            processed_base64 = base64.b64encode(buffer).decode("utf-8")
        except Exception as enc_error:
            logger.error(f"Error encoding frame: {enc_error}")
            # Return a simple error image
            return {
                "error": "Error processing image",
                "timestamp": time.time()
            }
        
        # Return the processed data
        return {
            "image": f"data:image/jpeg;base64,{processed_base64}",
            "prediction": stable_class,
            "confidence": float(stable_confidence) if stable_confidence else 0.0,
            "timestamp": time.time()
        }
    except Exception as e:
        logger.error(f"Error processing frame: {e}")
        logger.debug(traceback.format_exc())
        return {
            "error": str(e),
            "timestamp": time.time()
        }

# Main entry point
if __name__ == "__main__":
    uvicorn.run(
        "app.backend.main:app", 
        host="0.0.0.0", 
        port=8000, 
        reload=True,
        log_level="info"
    ) 