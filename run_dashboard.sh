#!/bin/bash

# Script to run both the API and dashboard

# Use the full path to Python from Anaconda
PYTHON_PATH="/Users/fabio/anaconda3/bin/python"

# Start the API in the background
echo "Starting the API server..."
$PYTHON_PATH -m src.api.main &
API_PID=$!

# Wait for the API to start up
echo "Waiting for API to start (5 seconds)..."
sleep 5

# Start the dashboard - usando o comando correto para o streamlit
echo "Starting the dashboard..."
$PYTHON_PATH -c "import streamlit.web.cli as stcli; stcli.main()" src/dashboard/app.py

# When the dashboard is closed, kill the API
echo "Dashboard closed, stopping API server..."
kill $API_PID
