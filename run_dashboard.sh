#!/bin/bash

# Script to start both the API and the Streamlit dashboard

echo "Starting the Sales Forecasting Application..."

# Check if API is already running on port 8000
if lsof -Pi :8000 -sTCP:LISTEN -t >/dev/null ; then
    echo "API already running on port 8000"
else
    echo "Starting the API server..."
    # Start API in the background
    python src/api/main.py &
    API_PID=$!
    echo "API started with PID: $API_PID"
    
    # Wait for API to be ready
    echo "Waiting for API to be ready..."
    for i in {1..10}; do
        if curl -s http://localhost:8000/health | grep -q "healthy"; then
            echo "API is ready!"
            break
        fi
        
        if [ $i -eq 10 ]; then
            echo "API failed to start properly. Please check the logs."
            kill $API_PID
            exit 1
        fi
        
        echo "Waiting for API... ($i/10)"
        sleep 2
    done
fi

# Start Streamlit dashboard
echo "Starting Streamlit dashboard..."
streamlit run src/dashboard/app.py

# When the Streamlit dashboard is closed, check if we need to cleanup the API process
if [ -n "$API_PID" ]; then
    echo "Stopping API process with PID: $API_PID"
    kill $API_PID
fi

echo "Application shutdown complete." 