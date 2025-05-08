#!/bin/bash

# Script to run the sales forecasting application with database support

echo "Starting Sales Forecasting App with Database Support"
echo "===================================================="

# Check if the database has been initialized
if [ ! -f "data/db/sales_forecasting.db" ]; then
    echo "Database not found. Initializing database..."
    ./init_database.sh
    
    if [ $? -ne 0 ]; then
        echo "Error: Database initialization failed. Exiting."
        exit 1
    fi
fi

# Start the API server in the background
echo "Starting API server..."
python -m src.api.main &
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
python -m src.dashboard.app

# When the dashboard is closed, stop the API server
echo "Stopping API server..."
kill $API_PID 2>/dev/null

echo "Application stopped." 