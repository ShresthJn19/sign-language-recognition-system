/**
 * Sign Language Recognition - Frontend JavaScript
 */

// Application state
const state = {
    // Camera state
    cameraActive: false,
    videoStream: null,
    
    // Recognition state
    recognizedSigns: [],
    
    // Text-to-speech state
    ttsEnabled: true,
    selectedVoice: null,
    availableVoices: [],
    
    // UI state
    confidenceThreshold: 0.65,
    
    // System state
    modelLoaded: false,
    
    // Processing state
    processingActive: false,
    processingFrame: false,
    lastFrameTime: 0,
    frameSendInterval: 100, // ms between frames
};

// DOM Elements
const elements = {
    // Status elements
    modelStatus: document.getElementById('modelStatus'),
    cameraStatus: document.getElementById('cameraStatus'),
    
    // Video elements
    video: document.getElementById('video'),
    canvas: document.getElementById('canvas'),
    videoOverlay: document.getElementById('video-overlay'),
    
    // Control buttons
    cameraToggle: document.getElementById('cameraToggle'),
    fullscreenToggle: document.getElementById('fullscreenToggle'),
    
    // Prediction elements
    prediction: document.getElementById('prediction'),
    confidenceBar: document.getElementById('confidenceBar'),
    confidenceValue: document.getElementById('confidenceValue'),
    hand1: document.getElementById('hand1'),
    hand2: document.getElementById('hand2'),
    
    // Text-to-speech elements
    textOutput: document.getElementById('textOutput'),
    autoSpeakToggle: document.getElementById('autoSpeakToggle'),
    voiceSelect: document.getElementById('voiceSelect'),
    speakButton: document.getElementById('speakButton'),
    clearTextButton: document.getElementById('clearTextButton'),
    
    // Notification area
    notifications: document.getElementById('notifications')
};

// Initialize the application
async function init() {
    console.log('Initializing application...');
    
    // Check browser compatibility first
    checkBrowserCompatibility();
    
    // Check system status
    await checkSystemStatus();
    
    // Initialize text-to-speech
    initTextToSpeech();
    
    // Set up event listeners
    setupEventListeners();
}

// Check browser compatibility
function checkBrowserCompatibility() {
    // Check if running in a secure context (needed for camera in some browsers)
    const isSecureContext = window.isSecureContext || 
                           location.protocol === 'https:' || 
                           location.hostname === 'localhost' || 
                           location.hostname === '127.0.0.1';
                           
    if (!isSecureContext) {
        showNotification('For security reasons, camera access requires HTTPS. Some features may not work.', 'warning');
    }
    
    // Check if MediaDevices API is available directly
    const hasMediaDevices = !!(navigator.mediaDevices && navigator.mediaDevices.getUserMedia);
    if (!hasMediaDevices) {
        // Check if we can use a polyfill
        const hasLegacyAPI = !!(navigator.getUserMedia || 
                              navigator.webkitGetUserMedia || 
                              navigator.mozGetUserMedia || 
                              navigator.msGetUserMedia);
        
        if (hasLegacyAPI) {
            showNotification('Using legacy camera API. For best experience, use Chrome or Firefox.', 'warning');
        } else {
            showNotification('Your browser does not support camera access. Please try Chrome or Firefox.', 'error');
            // Disable camera button
            if (elements.cameraToggle) {
                elements.cameraToggle.disabled = true;
                elements.cameraToggle.title = 'Camera not supported in this browser';
            }
        }
    }
    
    // Log user agent for debugging
    console.log('Browser:', navigator.userAgent);
}

// Check system status
async function checkSystemStatus() {
    try {
        const response = await fetch('/api/status');
        if (!response.ok) {
            throw new Error(`Server returned ${response.status}: ${response.statusText}`);
        }
        
        const data = await response.json();
        
        // Update model status
        state.modelLoaded = data.model_loaded;
        updateModelStatus(data.model_loaded);
        
        return data;
    } catch (error) {
        console.error('Error checking system status:', error);
        showNotification('Error connecting to the server. Please make sure it is running.', 'error');
        throw error;
    }
}

// Update model status display
function updateModelStatus(isLoaded) {
    if (elements.modelStatus) {
        elements.modelStatus.innerHTML = isLoaded ? 
            '<i class="fas fa-brain"></i> <span>Model: Loaded</span>' : 
            '<i class="fas fa-brain"></i> <span>Model: Not loaded</span>';
        
        if (!isLoaded) {
            showNotification('Sign language model not loaded. Recognition will not work.', 'warning');
        }
    }
}

// Update camera status display
function updateCameraStatus(isActive) {
    if (elements.cameraStatus) {
        elements.cameraStatus.innerHTML = isActive ? 
            '<i class="fas fa-video"></i> <span>Camera: Active</span>' : 
            '<i class="fas fa-video-slash"></i> <span>Camera: Off</span>';
    }
    
    // Update camera button text
    if (elements.cameraToggle) {
        elements.cameraToggle.innerHTML = isActive ? 
            '<i class="fas fa-video-slash"></i> <span>Stop Camera</span>' : 
            '<i class="fas fa-video"></i> <span>Start Camera</span>';
    }
}

// Set up event listeners
function setupEventListeners() {
    // Camera toggle
    if (elements.cameraToggle) {
        elements.cameraToggle.addEventListener('click', toggleCamera);
    }
    
    // Fullscreen toggle
    if (elements.fullscreenToggle) {
        elements.fullscreenToggle.addEventListener('click', toggleFullscreen);
    }
    
    // Text-to-speech toggle
    if (elements.autoSpeakToggle) {
        elements.autoSpeakToggle.addEventListener('change', e => {
            state.ttsEnabled = e.target.checked;
        });
    }
    
    // Voice selection
    if (elements.voiceSelect) {
        elements.voiceSelect.addEventListener('change', e => {
        // Get user media
        const constraints = { 
            video: { 
                width: { ideal: 640 }, 
                height: { ideal: 480 },
                facingMode: 'user'
            },
            audio: false 
        };
                    height: { ideal: 480 },
        const stream = await navigator.mediaDevices.getUserMedia(constraints);
                audio: false 
            },
            // Attempt 2: Any video
            { 
                video: true,
        // Set video source
        elements.video.srcObject = stream;
        
                audio: false 
        await new Promise(resolve => {
            // Attempt 3: Lower resolution
                elements.video.play().then(resolve);
            }
        let lastError = null;
        for (const constraint of constraints) {
            try {
                console.log('Trying camera with constraints:', constraint);
                stream = await navigator.mediaDevices.getUserMedia(constraint);
                if (stream) {
                    console.log('Camera access successful with constraints:', constraint);
                    break;
                }
            } catch (error) {
                console.warn('Camera access failed with constraints:', constraint, error);
                lastError = error;
            }
        }
        
        // If all attempts failed
        elements.videoOverlay.style.display = 'none';
            throw lastError || new Error('Could not access camera after multiple attempts');
        }
        // Show error notification
        showNotification(`Camera access error: ${error.message}`, 'error');
            };
            
            // Fallback if onloadedmetadata doesn't fire
            setTimeout(resolve, 2000);
        });
        
        // Show canvas, hide overlay
        elements.videoOverlay.style.display = 'none';
        elements.canvas.style.display = 'block';
        
        // Update UI
        updateCameraStatus(true);
        elements.cameraToggle.disabled = false;
        
        // Start processing frames
        state.processingActive = true;
        processVideoFrames();
        
    } catch (error) {
        console.error('Error accessing camera:', error);
        elements.videoOverlay.innerHTML = `<p>Camera error: ${error.message}</p><p>Please check browser permissions</p>`;
        elements.cameraToggle.disabled = false;
        
        // Show detailed error notification
        let errorMessage = 'Camera access error: ' + error.message;
        
        // Provide more helpful messages for common errors
        if (error.name === 'NotAllowedError' || error.name === 'PermissionDeniedError') {
            errorMessage = 'Camera permission denied. Please allow camera access in your browser settings.';
        } else if (error.name === 'NotFoundError') {
            errorMessage = 'No camera found. Please connect a camera and try again.';
        } else if (error.name === 'NotReadableError' || error.name === 'TrackStartError') {
            errorMessage = 'Camera is in use by another application or not readable.';
        } else if (error.name === 'OverconstrainedError') {
            errorMessage = 'Camera cannot satisfy the requested constraints.';
        } else if (error.name === 'TypeError' && error.message.includes('getUserMedia')) {
            errorMessage = 'Browser security policy is preventing camera access. Try using HTTPS or localhost.';
        }
        
        showNotification(errorMessage, 'error');
    }
}

// Stop camera
function stopCamera() {
    // Stop all tracks
    if (state.videoStream) {
        state.videoStream.getTracks().forEach(track => track.stop());
        state.videoStream = null;
    }
    
    // Update state
    state.cameraActive = false;
    state.processingActive = false;
    
    // Clear video source
    elements.video.srcObject = null;
    
    // Update UI
    updateCameraStatus(false);
    elements.videoOverlay.style.display = 'flex';
    elements.videoOverlay.innerHTML = '<p>Camera stopped</p>';
}

// Toggle fullscreen
function toggleFullscreen() {
    const videoContainer = document.querySelector('.video-container');
    
    if (!document.fullscreenElement) {
        if (videoContainer.requestFullscreen) {
            videoContainer.requestFullscreen();
        } else if (videoContainer.webkitRequestFullscreen) {
            videoContainer.webkitRequestFullscreen();
        } else if (videoContainer.msRequestFullscreen) {
            videoContainer.msRequestFullscreen();
        }
        elements.fullscreenToggle.innerHTML = '<i class="fas fa-compress"></i> <span>Exit Fullscreen</span>';
    } else {
        if (document.exitFullscreen) {
            document.exitFullscreen();
        } else if (document.webkitExitFullscreen) {
            document.webkitExitFullscreen();
        } else if (document.msExitFullscreen) {
            document.msExitFullscreen();
        }
        elements.fullscreenToggle.innerHTML = '<i class="fas fa-expand"></i> <span>Fullscreen</span>';
    }
}

// Process video frames
async function processVideoFrames() {
    if (!state.cameraActive || !state.processingActive) {
        return;
    }
    
    const now = Date.now();
    
    // Limit frame rate to avoid overwhelming the server
    if (!state.processingFrame && now - state.lastFrameTime > state.frameSendInterval) {
        state.processingFrame = true;
        state.lastFrameTime = now;
        
        try {
            const ctx = elements.canvas.getContext('2d');
            
            // Set canvas dimensions to match video
            elements.canvas.width = elements.video.videoWidth;
            elements.canvas.height = elements.video.videoHeight;
            
            // Draw the current frame to the canvas
            ctx.drawImage(elements.video, 0, 0, elements.canvas.width, elements.canvas.height);
            
            // Get the frame as a base64 image
            const imageData = elements.canvas.toDataURL('image/jpeg', 0.7); // Reduced quality to decrease payload size
            
            // Send the frame to the server
            const response = await fetch('/api/process-frame', {
                method: 'POST',
                body: imageData,
                headers: {
                    'Content-Type': 'text/plain',
                }
            });
            
            if (!response.ok) {
                throw new Error(`Server returned ${response.status}: ${response.statusText}`);
            }
            
            const data = await response.json();
            
            if (data.error) {
                console.error('Server error:', data.error);
            } else {
                // Update the canvas with the processed image
                const img = new Image();
                img.onload = () => {
                    ctx.drawImage(img, 0, 0, elements.canvas.width, elements.canvas.height);
                };
                img.src = data.image;
                
                // Update predictions
                updatePredictions(data.predictions);
            }
        } catch (error) {
            console.error('Error processing frame:', error);
        } finally {
            state.processingFrame = false;
        }
    }
    
    // Schedule next frame
    requestAnimationFrame(processVideoFrames);
}

// Update predictions display
function updatePredictions(predictions) {
    // Clear previous display
    elements.hand1.querySelector('.hand-prediction').textContent = '-';
    elements.hand2.querySelector('.hand-prediction').textContent = '-';
    
    // No predictions
    if (!predictions || predictions.length === 0) {
        elements.prediction.textContent = 'No hands detected';
        elements.confidenceBar.style.width = '0%';
        elements.confidenceValue.textContent = '0%';
        return;
    }
    
    // Process each prediction
    predictions.forEach((pred, index) => {
        const handElement = index === 0 ? elements.hand1 : elements.hand2;
        
        if (handElement) {
            handElement.querySelector('.hand-prediction').textContent = 
                `${pred.prediction} (${Math.round(pred.confidence * 100)}%)`;
        }
        
        // Use first hand as main prediction
        if (index === 0) {
            // Update main prediction display
            elements.prediction.textContent = pred.prediction;
            
            // Update confidence display
            const confidencePercent = Math.round(pred.confidence * 100);
            elements.confidenceBar.style.width = `${confidencePercent}%`;
            elements.confidenceValue.textContent = `${confidencePercent}%`;
            
            // Add to recognized signs if confidence is high enough
            if (pred.confidence >= state.confidenceThreshold) {
                addRecognizedSign(pred.prediction);
            }
        }
    });
}

// Add a recognized sign to the output
function addRecognizedSign(sign) {
    // Check if we already have this sign as the last recognized
    if (state.recognizedSigns.length > 0 && 
        state.recognizedSigns[state.recognizedSigns.length - 1] === sign) {
        return; // Skip duplicate consecutive signs
    }
    
    // Add the sign
    state.recognizedSigns.push(sign);
    
    // Update display
    updateTextOutput();
    
    // Speak if enabled
    if (state.ttsEnabled) {
        speakText(sign);
    }
}

// Update text output display
function updateTextOutput() {
    if (elements.textOutput) {
        if (state.recognizedSigns.length > 0) {
            elements.textOutput.innerHTML = `<p>${state.recognizedSigns.join(' ')}</p>`;
        } else {
            elements.textOutput.innerHTML = `<p>Your recognized signs will appear here</p>`;
        }
    }
}

// Text-to-speech functions
function initTextToSpeech() {
    if ('speechSynthesis' in window) {
        // Get available voices
        const getVoices = () => {
            state.availableVoices = window.speechSynthesis.getVoices();
            populateVoiceSelect();
        };
        
        // Chrome loads voices asynchronously
        if (window.speechSynthesis.onvoiceschanged !== undefined) {
            window.speechSynthesis.onvoiceschanged = getVoices;
        }
        
        // Get voices immediately for Firefox
        getVoices();
        
        // Also fetch server voices
        fetchServerVoices();
    } else {
        console.warn('Text-to-speech not supported by this browser');
        showNotification('Text-to-speech is not supported by your browser', 'warning');
        
        if (elements.autoSpeakToggle) {
            elements.autoSpeakToggle.disabled = true;
        }
        
        if (elements.speakButton) {
            elements.speakButton.disabled = true;
        }
    }
}

// Fetch server voices
async function fetchServerVoices() {
    try {
        const response = await fetch('/api/voices');
        if (response.ok) {
            const data = await response.json();
            if (data.voices) {
                populateVoiceSelect(data.voices);
            }
        }
    } catch (error) {
        console.error('Error fetching voices:', error);
    }
}

// Populate voice select dropdown
function populateVoiceSelect(serverVoices) {
    if (!elements.voiceSelect) return;
    
    // Clear existing options
    elements.voiceSelect.innerHTML = '';
    
    // Add default option
    const defaultOption = document.createElement('option');
    defaultOption.value = '';
    defaultOption.textContent = 'Default Voice';
    elements.voiceSelect.appendChild(defaultOption);
    
    // If we have server voices, use those
    if (serverVoices && serverVoices.length > 0) {
        for (const voice of serverVoices) {
            const option = document.createElement('option');
            option.value = voice.id;
            option.textContent = voice.name;
            elements.voiceSelect.appendChild(option);
        }
    } 
    // Otherwise use browser voices
    else if (state.availableVoices && state.availableVoices.length > 0) {
        for (const voice of state.availableVoices) {
            const option = document.createElement('option');
            option.value = voice.name;
            option.textContent = `${voice.name} (${voice.lang})`;
            elements.voiceSelect.appendChild(option);
        }
    }
    
    // Show the select
    elements.voiceSelect.style.display = 'block';
}

// Set TTS voice
async function setVoice(voiceId) {
    if (!voiceId) return;
    
    try {
        const response = await fetch(`/api/voices/${voiceId}`, {
            method: 'POST'
        });
        
        if (!response.ok) {
            throw new Error(`Server returned ${response.status}`);
        }
        
        console.log(`Voice set to ${voiceId}`);
    } catch (error) {
        console.error('Error setting voice:', error);
        showNotification('Error setting voice. Using browser TTS instead.', 'warning');
    }
}

// Speak text
async function speakText(text) {
    if (!text) return;
    
    try {
        // Try server-side TTS first
        const response = await fetch('/api/speak', {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json'
            },
            body: JSON.stringify({ text })
        });
        
        if (!response.ok) {
            throw new Error('Server TTS failed');
        }
    } catch (error) {
        console.warn('Server TTS failed, using browser TTS:', error);
        
        // Fallback to browser TTS
        if ('speechSynthesis' in window) {
            const utterance = new SpeechSynthesisUtterance(text);
            
            // Set voice if selected
            if (state.selectedVoice) {
                const voice = state.availableVoices.find(v => v.name === state.selectedVoice);
                if (voice) {
                    utterance.voice = voice;
                }
            }
            
            window.speechSynthesis.speak(utterance);
        }
    }
}

// Show notification
function showNotification(message, type = 'info') {
    const notification = document.createElement('div');
    notification.className = `notification ${type}`;
    notification.innerHTML = message;
    
    elements.notifications.appendChild(notification);
    
    // Remove notification after 5 seconds
    setTimeout(() => {
        notification.style.opacity = '0';
        setTimeout(() => {
            notification.remove();
        }, 300);
    }, 5000);
}

// Initialize on page load
document.addEventListener('DOMContentLoaded', init);