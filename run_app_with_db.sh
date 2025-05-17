#!/bin/bash

# Script to run the sales forecasting application with database support

echo "Starting Sales Forecasting App with Database Support"
echo "===================================================="

# Function to initialize the database if it doesn't exist
init_database() {
    echo "Checking if database initialization is required..."
    
    if [ -f "data/db/sales_forecasting.db" ]; then
        echo "Database already exists, skipping initialization."
        return 0
    fi
    
    echo "Database does not exist, initializing..."
    
    # Create necessary directories
    mkdir -p data/db
    mkdir -p data/raw
    
    # Initialize the database
    echo "Running database initialization..."
    
    # Use environment variable or default to "python"
    PYTHON_PATH=${PYTHON_PATH:-python}
    
    $PYTHON_PATH -m src.database.init_db
    
    if [ $? -eq 0 ]; then
        echo "Database initialized successfully."
        return 0
    else
        echo "Failed to initialize database."
        return 1
    fi
}

# Initialize the database
init_database

# Use the full path to Python from Anaconda
PYTHON_PATH="/Users/fabio/anaconda3/bin/python"

# Start the API server in the background
echo "Starting API server..."
$PYTHON_PATH -m src.api.main &
API_PID=$!

# Wait for API to start
echo "Waiting for API to start..."
sleep 5

# Check if API is running
if ! curl -s http://localhost:8000/health > /dev/null; then
    echo "Error: API failed to start. Exiting."
    kill $API_PID 2>/dev/null
    exit 1
fi

echo "API server running at http://localhost:8000"

# Start the dashboard
echo "Starting Streamlit dashboard..."
$PYTHON_PATH -m src.dashboard.app

# When the dashboard is closed, stop the API server
echo "Stopping API server..."
kill $API_PID 2>/dev/null

echo "Application stopped." 