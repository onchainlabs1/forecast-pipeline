#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Script to start the API and dashboard for the sales forecasting project.
This script checks if the API is running, starts it if necessary, and then starts the dashboard.
"""

import os
import sys
import subprocess
import time
import logging
import socket
import requests
from pathlib import Path

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)

# Define paths and ports
PYTHON_PATH = os.environ.get("PYTHON_PATH", "python")  # Use system python by default or from env var
API_PORT = int(os.environ.get("API_PORT", 8000))
API_HOST = os.environ.get("API_HOST", "localhost")
API_MODULE = "src.api.main"
DASHBOARD_MODULE = "src.dashboard.app"

def is_port_in_use(port, host='localhost'):
    """Checks if a port is in use."""
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        return s.connect_ex((host, port)) == 0

def is_api_healthy():
    """Checks if the API is responding correctly."""
    try:
        response = requests.get(f"http://{API_HOST}:{API_PORT}/health")
        return response.status_code == 200
    except Exception as e:
        logger.error(f"Error checking API: {e}")
        return False

def init_database():
    """Initializes the database if necessary."""
    db_path = Path("data/db/sales_forecasting.db")
    
    if db_path.exists():
        logger.info("Database already exists. Skipping initialization.")
        return True
        
    logger.info("Initializing database...")
    
    # Create necessary directories
    os.makedirs("data/db", exist_ok=True)
    os.makedirs("data/raw", exist_ok=True)
    
    try:
        # Execute the database initialization module
        subprocess.run(
            [PYTHON_PATH, "-m", "src.database.init_db"],
            check=True
        )
        logger.info("Database initialized successfully.")
        return True
    except subprocess.CalledProcessError as e:
        logger.error(f"Error initializing database: {e}")
        return False

def start_api():
    """Starts the API in a separate process."""
    logger.info("Starting API...")
    
    # Check if the API is already running
    if is_port_in_use(API_PORT):
        if is_api_healthy():
            logger.info(f"API is already running at {API_HOST}:{API_PORT}")
            return True
        else:
            logger.warning("API is running, but not responding correctly.")
            return False
    
    # Start the API in a separate process
    try:
        api_process = subprocess.Popen(
            [PYTHON_PATH, "-m", API_MODULE],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            universal_newlines=True
        )
        
        # Wait for the API to start
        logger.info("Waiting for API to start...")
        for _ in range(10):
            time.sleep(1)
            if is_api_healthy():
                logger.info("API started successfully!")
                return True
                
        logger.error("Timeout exceeded while waiting for API to start.")
        return False
    except Exception as e:
        logger.error(f"Error starting API: {e}")
        return False

def start_dashboard():
    """Starts the Streamlit dashboard."""
    logger.info("Starting dashboard...")
    
    try:
        # Start the dashboard (this process blocks until the dashboard is closed)
        subprocess.run(
            [PYTHON_PATH, "-m", "streamlit", "run", f"src/dashboard/app.py"],
            check=True
        )
        logger.info("Dashboard closed.")
        return True
    except subprocess.CalledProcessError as e:
        logger.error(f"Error running dashboard: {e}")
        return False
    except KeyboardInterrupt:
        logger.info("Dashboard interrupted by user.")
        return True

def main():
    """Main function."""
    logger.info("Starting sales forecasting application...")
    
    # Initialize database
    if not init_database():
        logger.error("Failed to initialize database. Aborting.")
        return 1
    
    # Start API
    if not start_api():
        logger.error("Failed to start API. Aborting.")
        return 1
    
    # Start dashboard
    if not start_dashboard():
        logger.error("Failed to start dashboard.")
        return 1
    
    return 0

if __name__ == "__main__":
    sys.exit(main()) 