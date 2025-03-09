#!/usr/bin/env python3
# run.py - Script to start the sign language recognition application

import argparse
import uvicorn
import logging

def main():
    parser = argparse.ArgumentParser(description='Start the sign language recognition application')
    parser.add_argument('--host', type=str, default='0.0.0.0',
                      help='Host to run the server on')
    parser.add_argument('--port', type=int, default=8000,
                      help='Port to run the server on')
    parser.add_argument('--reload', action='store_true',
                      help='Enable auto-reload for development')
    parser.add_argument('--debug', action='store_true',
                      help='Enable debug mode with more verbose logging')
    
    args = parser.parse_args()
    
    # Configure logging
    log_level = logging.DEBUG if args.debug else logging.INFO
    logging.basicConfig(
        level=log_level,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    
    # Set up logger for this script
    logger = logging.getLogger("run")
    logger.setLevel(log_level)
    
    # Log startup info
    logger.info(f"Starting sign language recognition server at http://{args.host}:{args.port}")
    if args.reload:
        logger.info("Auto-reload enabled (development mode)")
    
    # Configure Uvicorn with optimized settings for WebSockets
    uvicorn_config = {
        "app": "app.backend.main:app",
        "host": args.host, 
        "port": args.port, 
        "reload": args.reload,
        "log_level": "debug" if args.debug else "info",
        "ws_max_size": 16 * 1024 * 1024,  # 16MB max WebSocket message size
        "ws_ping_interval": 20.0,         # Send ping every 20 seconds
        "ws_ping_timeout": 30.0,          # Wait 30 seconds for pong response
        "timeout_keep_alive": 120,        # Keep connections alive longer
    }
    
    # Start the server with optimized settings
    uvicorn.run(**uvicorn_config)

if __name__ == "__main__":
    main() 