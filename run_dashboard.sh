#!/bin/bash

# Script to run both the API and dashboard

# Use environment variable or default to "python"
PYTHON_PATH=${PYTHON_PATH:-python}
API_PORT=${API_PORT:-8000}
DASHBOARD_PORT=${DASHBOARD_PORT:-8501}

# Start the API in the background
echo "Starting the API server on port $API_PORT..."
API_PORT=$API_PORT $PYTHON_PATH -m src.api.main &
API_PID=$!

# Wait for the API to start up
echo "Waiting for API to start (5 seconds)..."
sleep 5

# Start the dashboard 
echo "Starting the dashboard on port $DASHBOARD_PORT..."
$PYTHON_PATH -m streamlit run src/dashboard/app.py --server.port $DASHBOARD_PORT

# When the dashboard is closed, kill the API
echo "Dashboard closed, stopping API server..."
kill $API_PID
