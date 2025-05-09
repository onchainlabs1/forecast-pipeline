#!/bin/bash

# Script to run both the API and dashboard

# Start the API in the background
echo "Starting the API server..."
python -m src.api.main &
API_PID=$!

# Wait for the API to start up
echo "Waiting for API to start (5 seconds)..."
sleep 5

# Start the dashboard
echo "Starting the dashboard..."
python -m src.dashboard.app

# When the dashboard is closed, kill the API
echo "Dashboard closed, stopping API server..."
kill $API_PID
