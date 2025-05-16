#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Script to start the Streamlit dashboard for the sales forecasting project.
"""

import os
import sys
import subprocess
import time

def main():
    """Main function to start the dashboard."""
    print("Starting the sales forecasting dashboard...")
    
    # Check if the API is running
    api_running = False
    try:
        # Try to make a request to the API
        import requests
        response = requests.get("http://localhost:8000/health")
        if response.status_code == 200:
            print("API is already running")
            api_running = True
    except:
        pass
    
    # Start the API if not running
    api_process = None
    if not api_running:
        print("Starting the API...")
        api_cmd = [sys.executable, "-m", "src.api.main"]
        api_process = subprocess.Popen(api_cmd)
        print(f"API started (PID: {api_process.pid})")
        # Wait for the API to start
        print("Waiting for API to start...")
        time.sleep(5)
    
    # Start the dashboard
    print("Starting the Streamlit dashboard...")
    try:
        # Try to use the streamlit module directly
        streamlit_cmd = [sys.executable, "-m", "streamlit", "run", "src/dashboard/app.py"]
        subprocess.run(streamlit_cmd)
    except Exception as e:
        print(f"Error starting the dashboard: {e}")
    
    # Terminate the API if we started it
    if api_process:
        print("Terminating the API...")
        api_process.terminate()
        api_process.wait()
        print("API terminated.")

if __name__ == "__main__":
    main() 