#!/usr/bin/env python3
# text_to_speech.py - Utility for converting recognized text to speech

import pyttsx3
import threading
import queue
import time

class TextToSpeech:
    """
    Class for converting text to speech using pyttsx3.
    Includes a background worker to prevent blocking the main thread.
    """
    def __init__(self, rate=150, volume=1.0, voice=None):
        """
        Initialize the text-to-speech engine.
        
        Args:
            rate: Speech rate (words per minute)
            volume: Volume (0.0 to 1.0)
            voice: Voice ID (if None, uses default voice)
        """
        # Initialize the TTS engine
        self.engine = pyttsx3.init()
        self.engine.setProperty('rate', rate)
        self.engine.setProperty('volume', volume)
        
        # Set voice if specified
        if voice is not None:
            self.engine.setProperty('voice', voice)
        
        # Initialize the worker thread and queue
        self.text_queue = queue.Queue()
        self.worker_thread = threading.Thread(target=self._worker, daemon=True)
        self.worker_thread.start()
        
        # Track the last spoken text and time to avoid repetition
        self.last_text = ""
        self.last_time = 0
        self.min_interval = 1.5  # Minimum time between repeating the same text (seconds)
    
    def speak(self, text):
        """
        Convert text to speech (non-blocking).
        Adds the text to a queue for processing by the worker thread.
        
        Args:
            text: Text to convert to speech
        """
        # Check if this is a repetition
        current_time = time.time()
        if text == self.last_text and (current_time - self.last_time) < self.min_interval:
            return
        
        # Update last text and time
        self.last_text = text
        self.last_time = current_time
        
        # Add to queue
        self.text_queue.put(text)
    
    def _worker(self):
        """
        Worker thread for processing the text queue.
        Runs continuously in the background.
        """
        while True:
            try:
                # Get text from queue (blocks until text is available)
                text = self.text_queue.get()
                
                # Skip empty text
                if not text:
                    self.text_queue.task_done()
                    continue
                
                # Speak the text
                self.engine.say(text)
                self.engine.runAndWait()
                
                # Mark task as done
                self.text_queue.task_done()
            except Exception as e:
                print(f"Error in TTS worker: {e}")
    
    def get_available_voices(self):
        """
        Get a list of available voices.
        
        Returns:
            List of voice IDs and names
        """
        voices = self.engine.getProperty('voices')
        voice_list = []
        
        for voice in voices:
            voice_list.append({
                'id': voice.id,
                'name': voice.name,
                'languages': voice.languages,
                'gender': voice.gender
            })
        
        return voice_list
    
    def set_voice(self, voice_id):
        """
        Set the voice by ID.
        
        Args:
            voice_id: ID of the voice to use
        """
        self.engine.setProperty('voice', voice_id)
    
    def set_rate(self, rate):
        """
        Set the speech rate.
        
        Args:
            rate: Speech rate (words per minute)
        """
        self.engine.setProperty('rate', rate)
    
    def set_volume(self, volume):
        """
        Set the volume.
        
        Args:
            volume: Volume (0.0 to 1.0)
        """
        self.engine.setProperty('volume', volume)
        
    def speak_blocking(self, text):
        """
        Convert text to speech (blocking).
        This method blocks until speech is complete.
        
        Args:
            text: Text to convert to speech
        """
        self.engine.say(text)
        self.engine.runAndWait() 