#!/bin/bash

echo "=== Forecast Pipeline Dashboard - Startup Script ==="
echo "This script will start all necessary services for the application."

# Project base directory (current directory)
BASE_DIR="$(pwd)"
# Use python from PATH or override with PYTHON_PATH env var
PYTHON=${PYTHON_PATH:-python}

# API port configuration
API_PORT=${API_PORT:-8000}
LANDING_PORT=${LANDING_PORT:-8002}
DASHBOARD_PORT=${DASHBOARD_PORT:-8501}

# Kill existing processes (if any)
echo "Terminating existing processes..."
pkill -f "uvicorn src.landing" || echo "No landing page process running"
pkill -f "uvicorn main:app" || echo "No API process running"
pkill -f "uvicorn src.api.main:app" || echo "No API process running"
pkill -f "streamlit run" || echo "No Dashboard process running"

# Wait a bit to ensure processes have terminated
echo "Waiting for complete process termination..."
sleep 3

# Check for ports in use
echo "Checking ports in use..."
if lsof -i:$API_PORT > /dev/null 2>&1; then
  echo "WARNING: Port $API_PORT is already in use. The API may not start correctly."
fi

if lsof -i:$LANDING_PORT > /dev/null 2>&1; then
  echo "WARNING: Port $LANDING_PORT is already in use. The landing page may not start correctly."
fi

if lsof -i:$DASHBOARD_PORT > /dev/null 2>&1; then
  echo "WARNING: Port $DASHBOARD_PORT is already in use. The Dashboard may not start correctly."
fi

# Create logs directory if it doesn't exist
mkdir -p "$BASE_DIR/logs"

# Start API in background
echo "Starting API (http://localhost:$API_PORT)..."
cd "$BASE_DIR" && $PYTHON -m uvicorn src.api.main:app --host 0.0.0.0 --port $API_PORT > logs/api.log 2>&1 &
API_PID=$!
echo "API started with PID $API_PID"

# Wait for API to initialize
echo "Waiting for API initialization..."
sleep 5

# Start landing page in background
echo "Starting Landing Page (http://localhost:$LANDING_PORT)..."
cd "$BASE_DIR" && PORT=$LANDING_PORT $PYTHON -m uvicorn src.landing.server:app --host 0.0.0.0 --port $LANDING_PORT > logs/landing.log 2>&1 &
LANDING_PID=$!
echo "Landing Page started with PID $LANDING_PID"

# Wait for landing page to initialize
echo "Waiting for landing page initialization..."
sleep 3

# Start dashboard in background
echo "Starting Dashboard (http://localhost:$DASHBOARD_PORT)..."
cd "$BASE_DIR" && $PYTHON -m streamlit run src/dashboard/app.py --server.port=$DASHBOARD_PORT > logs/dashboard.log 2>&1 &
DASHBOARD_PID=$!
echo "Dashboard started with PID $DASHBOARD_PID"

# Show final information
echo ""
echo "=== All services started ==="
echo "API: http://localhost:$API_PORT"
echo "Landing Page: http://localhost:$LANDING_PORT"
echo "Dashboard: http://localhost:$DASHBOARD_PORT"
echo ""
echo "Access credentials:"
echo "Username: admin"
echo "Password: admin"
echo ""
echo "To terminate all services, run: bash $BASE_DIR/restart.sh stop"
echo "Logs are available in the directory: $BASE_DIR/logs/"

# Option to stop all services
if [ "$1" = "stop" ]; then
  echo "Shutting down all services..."
  pkill -f "uvicorn src.landing" || echo "No landing page process running"
  pkill -f "uvicorn main:app" || echo "No API process running" 
  pkill -f "uvicorn src.api.main:app" || echo "No API process running"
  pkill -f "streamlit run" || echo "No Dashboard process running"
  echo "All services terminated."
fi 