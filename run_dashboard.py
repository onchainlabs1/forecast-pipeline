#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Script to start the Streamlit dashboard for the sales forecasting project.
"""

import os
import sys
import subprocess
import time

# Configure ports from environment variables
API_PORT = int(os.environ.get("API_PORT", 8000))
DASHBOARD_PORT = int(os.environ.get("DASHBOARD_PORT", 8501))

def main():
    """Main function to start the dashboard."""
    print(f"Starting the sales forecasting dashboard on port {DASHBOARD_PORT}...")
    
    # Check if the API is running
    api_running = False
    try:
        # Try to make a request to the API
        import requests
        response = requests.get(f"http://localhost:{API_PORT}/health")
        if response.status_code == 200:
            print(f"API is already running on port {API_PORT}")
            api_running = True
    except:
        pass
    
    # Start the API if not running
    api_process = None
    if not api_running:
        print(f"Starting the API on port {API_PORT}...")
        api_cmd = [sys.executable, "-m", "src.api.main"]
        api_env = os.environ.copy()
        api_env["PORT"] = str(API_PORT)
        api_process = subprocess.Popen(api_cmd, env=api_env)
        print(f"API started (PID: {api_process.pid})")
        # Wait for the API to start
        print("Waiting for API to start...")
        time.sleep(5)
    
    # Start the dashboard
    print(f"Starting the Streamlit dashboard on port {DASHBOARD_PORT}...")
    try:
        # Try to use the streamlit module directly
        streamlit_cmd = [sys.executable, "-m", "streamlit", "run", "src/dashboard/app.py", "--server.port", str(DASHBOARD_PORT)]
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