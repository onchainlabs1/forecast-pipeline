#!/usr/bin/env python
"""
Script to start the complete system with API, landing page (lovable) and dashboard (Streamlit)
"""

import os
import sys
import subprocess
import time
import webbrowser
import threading

# Define configurable ports
API_PORT = int(os.environ.get("API_PORT", 8000))
LANDING_PORT = int(os.environ.get("LANDING_PORT", 8002))
DASHBOARD_PORT = int(os.environ.get("DASHBOARD_PORT", 8501))

def run_api_server():
    """Runs the API server"""
    script_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), 
                              "src", "api", "main.py")
    
    print(f"Starting API server on port {API_PORT}...")
    api_process = subprocess.Popen([
        sys.executable, script_path
    ], env={**os.environ, "PORT": str(API_PORT)})
    time.sleep(3)  # Wait for API to start
    return api_process

def run_landing_page():
    """Runs the landing page server"""
    script_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), 
                              "src", "landing", "server.py")
    
    print(f"Starting landing page on port {LANDING_PORT}...")
    landing_process = subprocess.Popen([
        sys.executable, script_path
    ], env={**os.environ, "PORT": str(LANDING_PORT)})
    return landing_process

def run_streamlit():
    """Runs the Streamlit dashboard"""
    script_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), 
                              "src", "dashboard", "app.py")
    
    print(f"Starting Streamlit dashboard on port {DASHBOARD_PORT}...")
    streamlit_process = subprocess.Popen([
        "streamlit", "run", script_path, "--server.port", str(DASHBOARD_PORT)
    ])
    return streamlit_process

def open_pages():
    """Opens browser windows for the different pages after a brief delay"""
    time.sleep(5)  # Wait for servers to start
    webbrowser.open(f"http://localhost:{LANDING_PORT}")
    print(f"Landing page opened at http://localhost:{LANDING_PORT}")
    time.sleep(1)
    webbrowser.open(f"http://localhost:{DASHBOARD_PORT}")
    print(f"Dashboard opened at http://localhost:{DASHBOARD_PORT}")

if __name__ == "__main__":
    # Start the API
    api_process = run_api_server()
    
    # Start the landing page
    landing_process = run_landing_page()
    
    # Start the Streamlit dashboard
    streamlit_process = run_streamlit()
    
    # Open browsers
    threading.Thread(target=open_pages).start()
    
    try:
        # Keep the main process running
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        print("\nShutting down application...")
        api_process.terminate()
        landing_process.terminate()
        streamlit_process.terminate()
        print("All processes terminated.")
        
        sys.exit(0) 