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

def run_api_server():
    """Runs the API server"""
    script_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), 
                              "src", "api", "main.py")
    
    print("Starting API server...")
    api_process = subprocess.Popen([
        sys.executable, script_path
    ])
    time.sleep(3)  # Wait for API to start
    return api_process

def run_landing_page():
    """Runs the landing page server"""
    script_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), 
                              "src", "landing", "server.py")
    
    print("Starting landing page...")
    landing_process = subprocess.Popen([
        sys.executable, script_path
    ])
    return landing_process

def run_streamlit():
    """Runs the Streamlit dashboard"""
    script_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), 
                              "src", "dashboard", "app.py")
    
    print("Starting Streamlit dashboard...")
    streamlit_process = subprocess.Popen([
        "streamlit", "run", script_path
    ])
    return streamlit_process

def open_pages():
    """Opens browser windows for the different pages after a brief delay"""
    time.sleep(5)  # Wait for servers to start
    webbrowser.open("http://localhost:8002")
    print("Landing page opened at http://localhost:8002")
    time.sleep(1)
    webbrowser.open("http://localhost:8501")
    print("Dashboard opened at http://localhost:8501")

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