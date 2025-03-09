/**
 * app.js - Frontend JavaScript for the sign language recognition application
 * Handles webcam access, WebSocket communication, and UI updates
 */

// DOM Elements
const videoElement = document.getElementById('video');
const canvasElement = document.getElementById('canvas');
const overlayElement = document.getElementById('overlay');
const loadingElement = document.getElementById('loading');
const errorElement = document.getElementById('error');
const predictionElement = document.getElementById('prediction');
const confidenceBarElement = document.getElementById('confidenceBar');
const confidenceValueElement = document.getElementById('confidenceValue');
const textDisplayElement = document.getElementById('textDisplay');
const thresholdValueElement = document.getElementById('thresholdValue');
const confidenceThresholdElement = document.getElementById('confidenceThreshold');
const toggleCameraButton = document.getElementById('toggleCamera');
const toggleHandLandmarksButton = document.getElementById('toggleHandLandmarks');
const clearTextButton = document.getElementById('clearText');
const speakTextButton = document.getElementById('speakText');
const autoSpeakCheckbox = document.getElementById('autoSpeak');
const voiceSelectElement = document.getElementById('voiceSelect');
const setupPanelElement = document.getElementById('setupPanel');
const chromeLocalAccessPanelElement = document.getElementById('chromeLocalAccessPanel');

// Application state
const state = {
    cameraActive: false,
    socket: null,
    showHandLandmarks: true,
    confidenceThreshold: 0.7,
    currentRecognition: {
        text: null,
        confidence: 0,
        time: 0
    },
    recognizedText: [],
    videoWidth: 640,
    videoHeight: 480,
    processingFrame: false,
    reconnectAttempts: 0,
    maxReconnectAttempts: 5,
    reconnectInterval: 2000, // ms
    systemStatus: {
        modelLoaded: false,
        handDetectorLoaded: false,
        ttsLoaded: false
    },
    setupComplete: false,
    retryCount: 0,
    maxRetries: 3,
    isChrome: /Chrome/.test(navigator.userAgent) && !/Edge/.test(navigator.userAgent),
    isLocalHost: window.location.hostname === 'localhost' || window.location.hostname === '127.0.0.1' || window.location.protocol === 'file:',
    processingActive: false
};

// Initialize the application
async function init() {
    console.log("Initializing application...");
    
    // Check if we're using Chrome on a local connection
    updateChromePanelVisibility();
    
    // Check system status
    try {
        await checkSystemStatus();
    } catch (error) {
        console.error("Error checking system status:", error);
        showError("Unable to connect to the server. Please make sure the server is running.");
        return;
    }
    
    // Set up event listeners
    setupEventListeners();
    
    // Toggle visibility of setup panel based on model status
    updateSetupPanelVisibility();
    
    // Load available voices
    loadAvailableVoices();
    
    // Delay the camera start to give the backend time to initialize
    setTimeout(async () => {
        // Start the webcam
        await startCamera();
        
        // Start frame processing if camera started successfully
        if (state.cameraActive) {
            startProcessing();
        }
    }, 1000);
}

// Update Chrome panel visibility
function updateChromePanelVisibility() {
    if (chromeLocalAccessPanelElement) {
        if (state.isChrome && state.isLocalHost) {
            chromeLocalAccessPanelElement.classList.remove('hidden');
        } else {
            chromeLocalAccessPanelElement.classList.add('hidden');
        }
    }
}

// Check system status
async function checkSystemStatus() {
    try {
        const response = await fetch('/api/status');
        if (!response.ok) {
            throw new Error(`Server returned ${response.status}: ${response.statusText}`);
        }
        
        const data = await response.json();
        state.systemStatus = {
            modelLoaded: data.model_loaded,
            handDetectorLoaded: data.hand_detector_loaded,
            ttsLoaded: data.tts_loaded
        };
        
        console.log("System status:", state.systemStatus);
        state.setupComplete = true;
        
        // Display warning if model is not loaded
        if (!state.systemStatus.modelLoaded) {
            showWarning("Sign language model not loaded. Recognition features will not work. Please train the model first using 'python app/models/train_model.py'");
        }
        
        return data;
    } catch (error) {
        console.error("Error checking system status:", error);
        state.setupComplete = false;
        throw error;
    }
}

// Update setup panel visibility based on model status
function updateSetupPanelVisibility() {
    if (setupPanelElement) {
        if (!state.systemStatus.modelLoaded) {
            setupPanelElement.classList.remove('hidden');
        } else {
            setupPanelElement.classList.add('hidden');
        }
    }
}

// Show a warning message
function showWarning(message) {
    const warningElement = document.createElement('div');
    warningElement.className = 'warning';
    warningElement.innerHTML = `<p>${message}</p>`;
    
    // Add to overlay
    overlayElement.classList.remove('hidden');
    overlayElement.innerHTML = '';
    overlayElement.appendChild(warningElement);
    
    // Hide warning after 10 seconds
    setTimeout(() => {
        if (document.contains(warningElement)) {
            overlayElement.classList.add('hidden');
        }
    }, 10000);
}

// Show an error message
function showError(message) {
    // Use innerHTML to allow HTML formatting in error messages
    errorElement.innerHTML = message;
    overlayElement.classList.remove('hidden');
    loadingElement.classList.add('hidden');
    errorElement.classList.remove('hidden');
}

// Set up event listeners
function setupEventListeners() {
    // Start button
    document.getElementById('startButton').addEventListener('click', async () => {
        if (!state.cameraActive) {
            await startCamera();
            if (state.cameraActive) {
                startProcessing();
            }
        } else {
            startProcessing();
        }
    });

    // Stop button
    document.getElementById('stopButton').addEventListener('click', () => {
        stopProcessing();
    });

    // Camera toggle button
    document.getElementById('cameraToggle').addEventListener('click', async () => {
        if (state.cameraActive) {
            stopProcessing();
            stopCamera();
        } else {
            await startCamera();
            if (state.cameraActive) {
                startProcessing();
            }
        }
    });
    
    // Toggle hand landmarks button
    toggleHandLandmarksButton.addEventListener('click', () => {
        state.showHandLandmarks = !state.showHandLandmarks;
        toggleHandLandmarksButton.textContent = state.showHandLandmarks ? 'Hide Landmarks' : 'Show Landmarks';
    });
    
    // Clear text button
    clearTextButton.addEventListener('click', () => {
        state.recognizedText = [];
        updateTextDisplay();
    });
    
    // Speak text button
    speakTextButton.addEventListener('click', () => {
        speakCurrentText();
    });
    
    // Confidence threshold slider
    confidenceThresholdElement.addEventListener('input', (e) => {
        state.confidenceThreshold = parseFloat(e.target.value);
        thresholdValueElement.textContent = state.confidenceThreshold.toFixed(1);
    });
    
    // Voice selection dropdown
    voiceSelectElement.addEventListener('change', (e) => {
        const voiceId = e.target.value;
        if (voiceId) {
            setVoice(voiceId);
        }
    });
    
    // Window resize event
    window.addEventListener('resize', adjustCanvasSize);
}

// Start the webcam
async function startCamera() {
    try {
        // Show loading animation
        overlayElement.classList.remove('hidden');
        loadingElement.classList.remove('hidden');
        errorElement.classList.add('hidden');
        
        // Ensure MediaDevices API is accessible
        if (!navigator.mediaDevices) {
            console.error("MediaDevices API not available - trying to add polyfill");
            // Attempt to recreate the mediaDevices object (backup approach)
            navigator.mediaDevices = {};
            
            // Add getUserMedia if not available
            if (!navigator.mediaDevices.getUserMedia) {
                navigator.mediaDevices.getUserMedia = function(constraints) {
                    var getUserMedia = navigator.webkitGetUserMedia || 
                                      navigator.mozGetUserMedia || 
                                      navigator.msGetUserMedia;
                    
                    if (!getUserMedia) {
                        throw new Error("Camera API not available in your browser");
                    }
                    
                    return new Promise(function(resolve, reject) {
                        getUserMedia.call(navigator, constraints, resolve, reject);
                    });
                };
            }
        }
        
        console.log("Attempting to access camera...");
        
        // Request webcam access with multiple attempts
        let stream = null;
        let error = null;
        
        // First attempt: Try with ideal dimensions
        try {
            console.log("Camera attempt 1: Ideal dimensions");
            stream = await navigator.mediaDevices.getUserMedia({
                video: {
                    width: { ideal: state.videoWidth },
                    height: { ideal: state.videoHeight }
                },
                audio: false
            });
        } catch (err) {
            console.warn("First camera attempt failed:", err);
            error = err;
            
            // Second attempt: Try with minimal constraints
            try {
                console.log("Camera attempt 2: Basic video");
                stream = await navigator.mediaDevices.getUserMedia({
                    video: true,
                    audio: false
                });
            } catch (err2) {
                console.warn("Second camera attempt failed:", err2);
                error = err2;
                
                // Third attempt: Try using deprecated API directly (for older Chrome)
                try {
                    console.log("Camera attempt 3: Legacy API");
                    if (navigator.webkitGetUserMedia) {
                        stream = await new Promise((resolve, reject) => {
                            navigator.webkitGetUserMedia(
                                { video: true, audio: false },
                                resolve,
                                reject
                            );
                        });
                    } else {
                        throw new Error("Legacy API not available");
                    }
                } catch (err3) {
                    console.error("All camera access attempts failed");
                    error = err3;
                }
            }
        }
        
        if (!stream) {
            throw error || new Error("Could not access camera");
        }
        
        console.log("Camera access successful!");
        
        // Attach stream to video element
        videoElement.srcObject = stream;
        state.cameraActive = true;
        toggleCameraButton.textContent = 'Stop Camera';
        
        // Wait for video to be ready
        await new Promise(resolve => {
            videoElement.onloadedmetadata = () => {
                resolve();
            };
            
            // Fallback if onloadedmetadata doesn't fire
            setTimeout(resolve, 1000);
        });
        
        // Set canvas size
        adjustCanvasSize();
        
        // Hide loading overlay
        overlayElement.classList.add('hidden');
        
        // Start processing frames
        if (state.socket && state.socket.readyState === WebSocket.OPEN) {
            processFrame();
        }
    } catch (error) {
        console.error('Error accessing webcam:', error);
        
        // Construct a helpful error message based on the error
        let errorMessage = "Camera access denied. ";
        
        // Check for specific Chrome error messages
        if (error.name === 'NotAllowedError' || error.name === 'PermissionDeniedError') {
            errorMessage = `<p>Camera access blocked. Please check your browser settings:</p>
                <ol>
                    <li>Click the lock/site settings icon in your address bar</li>
                    <li>Make sure Camera permission is set to "Allow"</li>
                    <li>Reload the page and try again</li>
                </ol>
                <p>If running locally, try using a development server:</p>
                <pre>python -m http.server 8000</pre>`;
        } else if (error.message && error.message.includes("MediaDevices")) {
            errorMessage = `<p>Browser API error: ${error.message}</p>
                <p>Try using a modern browser like Chrome, Firefox, or Edge.</p>`;
        } else {
            errorMessage += `<p>${error.message}</p>`;
        }
        
        // Show error message
        showError(errorMessage);
        
        state.cameraActive = false;
        toggleCameraButton.textContent = 'Retry Camera';
    }
}

// Stop the webcam
function stopCamera() {
    if (videoElement.srcObject) {
        // Stop all tracks
        videoElement.srcObject.getTracks().forEach(track => track.stop());
        videoElement.srcObject = null;
    }
    
    state.cameraActive = false;
    toggleCameraButton.textContent = 'Start Camera';
    
    // Show overlay
    overlayElement.classList.remove('hidden');
    loadingElement.classList.add('hidden');
    errorElement.classList.add('hidden');
    
    // Clear canvas
    const ctx = canvasElement.getContext('2d');
    ctx.clearRect(0, 0, canvasElement.width, canvasElement.height);
}

// Adjust canvas size to match video dimensions
function adjustCanvasSize() {
    const videoContainer = videoElement.parentElement;
    const containerWidth = videoContainer.clientWidth;
    const containerHeight = videoContainer.clientHeight;
    
    canvasElement.width = containerWidth;
    canvasElement.height = containerHeight;
}

// Connect to the WebSocket server
function connectWebSocket() {
    // Close existing socket if any
    if (state.socket) {
        if (state.socket.readyState === WebSocket.OPEN || 
            state.socket.readyState === WebSocket.CONNECTING) {
            try {
                state.socket.close();
            } catch (e) {
                console.error("Error closing existing socket:", e);
            }
        }
        state.socket = null;
    }
    
    // Create new WebSocket connection
    const protocol = window.location.protocol === 'https:' ? 'wss:' : 'ws:';
    const wsUrl = `${protocol}//${window.location.host}/ws`;
    
    console.log(`Connecting to WebSocket at ${wsUrl}`);
    
    try {
        state.socket = new WebSocket(wsUrl);
    } catch (error) {
        console.error("Error creating WebSocket:", error);
        showError(`Failed to create WebSocket connection: ${error.message}`);
        return;
    }
    
    // Socket open event
    state.socket.onopen = () => {
        console.log('WebSocket connection established');
        state.reconnectAttempts = 0;
        
        // Start processing frames if camera is active
        if (state.cameraActive) {
            processFrame();
        }
    };
    
    // Socket message event
    state.socket.onmessage = (event) => {
        try {
            if (typeof event.data !== 'string') {
                console.warn("Received non-string data from server:", event.data);
                state.processingFrame = false;
                setTimeout(processFrame, 100);
                return;
            }
            
            const data = JSON.parse(event.data);
            
            // Handle error message
            if (data.error) {
                console.error('Server error:', data.error);
                showWarning(`Server error: ${data.error}`);
                
                // Continue anyway
                state.processingFrame = false;
                setTimeout(processFrame, 100);
                return;
            }
            
            // Update the UI with the received prediction
            updatePrediction(data);
            
            // Continue processing frames
            state.processingFrame = false;
            setTimeout(processFrame, 50);
        } catch (error) {
            console.error("Error parsing WebSocket message:", error, "Raw data:", event.data);
            state.processingFrame = false;
            setTimeout(processFrame, 500);
        }
    };
    
    // Socket close event
    state.socket.onclose = (event) => {
        console.log(`WebSocket connection closed: Code ${event.code}, Reason: ${event.reason || 'No reason provided'}`);
        
        // Attempt to reconnect
        if (state.reconnectAttempts < state.maxReconnectAttempts) {
            state.reconnectAttempts++;
            console.log(`Reconnecting (${state.reconnectAttempts}/${state.maxReconnectAttempts})...`);
            
            // Increase delay with each attempt (backoff)
            const delay = state.reconnectInterval * Math.pow(1.5, state.reconnectAttempts - 1);
            setTimeout(() => {
                connectWebSocket();
            }, delay);
        } else {
            console.error('Maximum reconnect attempts reached');
            showError("Connection to server lost. Please reload the page to reconnect.");
        }
    };
    
    // Socket error event
    state.socket.onerror = (error) => {
        console.error('WebSocket error:', error);
    };
}

// Connect to the server and start processing frames
function startProcessing() {
    console.log("Starting frame processing via HTTP polling");
    
    // Set flag to indicate processing is active
    state.processingActive = true;
    
    // Start the polling loop
    processFrame();
}

// Stop processing frames
function stopProcessing() {
    console.log("Stopping frame processing");
    state.processingActive = false;
}

// Process a single frame
async function processFrame() {
    // Skip if already processing a frame or processing is not active
    if (state.processingFrame || !state.processingActive || !state.cameraActive) {
        return;
    }
    
    state.processingFrame = true;
    
    try {
        // Check if video element is ready
        if (!videoElement.srcObject || 
            videoElement.videoWidth === 0 || 
            videoElement.videoHeight === 0) {
            state.processingFrame = false;
            setTimeout(processFrame, 500);
            return;
        }
        
        // Draw video frame to canvas
        const ctx = canvasElement.getContext('2d');
        ctx.drawImage(videoElement, 0, 0, canvasElement.width, canvasElement.height);
        
        // Get the canvas data as base64 image - with lower quality to reduce size
        try {
            const imageData = canvasElement.toDataURL('image/jpeg', 0.6);
            
            // Send the image data to the server using fetch
            const response = await fetch('/api/process-frame', {
                method: 'POST',
                body: imageData,
                headers: {
                    'Content-Type': 'text/plain',
                },
            });
            
            if (!response.ok) {
                throw new Error(`HTTP error! status: ${response.status}`);
            }
            
            const data = await response.json();
            
            // Handle error message
            if (data.error) {
                console.error('Server error:', data.error);
                showWarning(`Server error: ${data.error}`);
                
                // Continue anyway
                state.processingFrame = false;
                if (state.processingActive) {
                    setTimeout(processFrame, 500);
                }
                return;
            }
            
            // Update the UI with the received prediction
            updatePrediction(data);
            
            // Continue processing frames
            state.processingFrame = false;
            if (state.processingActive) {
                setTimeout(processFrame, 50);
            }
        } catch (error) {
            console.error('Error processing frame:', error);
            state.processingFrame = false;
            if (state.processingActive) {
                setTimeout(processFrame, 1000);
            }
        }
    } catch (error) {
        console.error('Error in processFrame:', error);
        state.processingFrame = false;
        if (state.processingActive) {
            setTimeout(processFrame, 1000);
        }
    }
}

// Update the UI with the received prediction
function updatePrediction(data) {
    if (data.prediction) {
        document.getElementById('prediction').textContent = data.prediction;
        if (state.textToSpeechEnabled) {
            speak(data.prediction);
        }
    }
    if (data.confidence) {
        document.getElementById('confidence').textContent = 
            `Confidence: ${(data.confidence * 100).toFixed(1)}%`;
    }
    if (data.status === 'no_hand') {
        document.getElementById('prediction').textContent = 'No hand detected';
        document.getElementById('confidence').textContent = '';
    }
    
    // Reset retry count on successful frame processing
    state.retryCount = 0;
    
    // If there's a valid prediction
    if (data.prediction && data.confidence > state.confidenceThreshold) {
        // Update confidence bar
        const confidencePercent = Math.round(data.confidence * 100);
        confidenceBarElement.style.width = `${confidencePercent}%`;
        confidenceValueElement.textContent = `${confidencePercent}%`;
        
        // Check if prediction changed
        const newPrediction = data.prediction !== state.currentRecognition.text;
        
        // Update current recognition
        state.currentRecognition = {
            text: data.prediction,
            confidence: data.confidence,
            time: Date.now()
        };
        
        // Update prediction display with animation if changed
        if (newPrediction) {
            predictionElement.textContent = data.prediction;
            predictionElement.classList.add('highlight');
            
            // Remove highlight animation after it completes
            setTimeout(() => {
                predictionElement.classList.remove('highlight');
            }, 300);
            
            // Add to recognized text
            addToRecognizedText(data.prediction);
            
            // Speak the text if auto-speak is enabled
            if (autoSpeakCheckbox.checked) {
                speakText(data.prediction);
            }
        }
    } else {
        // Reset if no prediction or low confidence
        if (data.prediction === null) {
            predictionElement.textContent = '-';
            confidenceBarElement.style.width = '0%';
            confidenceValueElement.textContent = '0%';
            
            // Clear current recognition after some time with no detection
            if (Date.now() - state.currentRecognition.time > 1000) {
                state.currentRecognition.text = null;
            }
        }
    }
    
    // Update the image
    if (data.image && state.showHandLandmarks) {
        const img = new Image();
        img.onload = () => {
            const ctx = canvasElement.getContext('2d');
            ctx.drawImage(img, 0, 0, canvasElement.width, canvasElement.height);
        };
        img.src = data.image;
    }
}

// Add recognized text to the display
function addToRecognizedText(text) {
    // Check if the last item is the same as the new text
    if (state.recognizedText.length > 0 && state.recognizedText[state.recognizedText.length - 1] === text) {
        // Don't add duplicates
        return;
    }
    
    // Add the text
    state.recognizedText.push(text);
    
    // Update the display
    updateTextDisplay();
}

// Update the text display
function updateTextDisplay() {
    textDisplayElement.textContent = state.recognizedText.join('');
}

// Speak the given text
function speakText(text) {
    // Use the backend TTS (already handled by the server)
    // This function is just for completeness
}

// Speak the current text in the text display
function speakCurrentText() {
    const text = textDisplayElement.textContent;
    if (text) {
        // Send a request to speak the entire text
        fetch('/api/speak', {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json',
            },
            body: JSON.stringify({ text }),
        }).catch(error => {
            console.error('Error requesting text-to-speech:', error);
        });
    }
}

// Load available TTS voices
function loadAvailableVoices() {
    if (!state.systemStatus.ttsLoaded) {
        voiceSelectElement.innerHTML = '<option value="">Text-to-speech not available</option>';
        return;
    }
    
    fetch('/api/voices')
        .then(response => response.json())
        .then(data => {
            if (data.voices && data.voices.length > 0) {
                // Clear the select element
                voiceSelectElement.innerHTML = '';
                
                // Add voices to the select element
                data.voices.forEach(voice => {
                    const option = document.createElement('option');
                    option.value = voice.id;
                    option.textContent = voice.name;
                    voiceSelectElement.appendChild(option);
                });
            } else {
                voiceSelectElement.innerHTML = '<option value="">No voices available</option>';
            }
        })
        .catch(error => {
            console.error('Error loading voices:', error);
            voiceSelectElement.innerHTML = '<option value="">Error loading voices</option>';
        });
}

// Set the TTS voice
function setVoice(voiceId) {
    fetch(`/api/voices/${voiceId}`, {
        method: 'POST',
    })
        .then(response => response.json())
        .then(data => {
            if (data.success) {
                console.log(`Voice set to: ${voiceId}`);
            } else if (data.error) {
                console.error('Error setting voice:', data.error);
                showWarning(`Error setting voice: ${data.error}`);
            }
        })
        .catch(error => {
            console.error('Error setting voice:', error);
            showWarning(`Error setting voice: ${error.message}`);
        });
}

// Initialize when the page loads
document.addEventListener('DOMContentLoaded', init);