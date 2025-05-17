#!/bin/bash

# Script to run both the API and dashboard

# Use environment variable or default to "python"
PYTHON_PATH=${PYTHON_PATH:-python}

# Start the API in the background
echo "Starting the API server..."
$PYTHON_PATH -m src.api.main &
API_PID=$!

# Wait for the API to start up
echo "Waiting for API to start (5 seconds)..."
sleep 5

# Start the dashboard - usando o comando correto para o streamlit
echo "Starting the dashboard..."
$PYTHON_PATH -m streamlit run src/dashboard/app.py --server.port 8501

# When the dashboard is closed, kill the API
echo "Dashboard closed, stopping API server..."
kill $API_PID
