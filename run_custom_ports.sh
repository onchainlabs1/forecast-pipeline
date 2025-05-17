#!/bin/bash

# Script to run the complete application with custom ports

# Set default ports
export API_PORT=${API_PORT:-8000}
export LANDING_PORT=${LANDING_PORT:-8002}
export DASHBOARD_PORT=${DASHBOARD_PORT:-8501}

echo "Starting RetailPro AI with custom ports:"
echo "API:       http://localhost:$API_PORT"
echo "Landing:   http://localhost:$LANDING_PORT"
echo "Dashboard: http://localhost:$DASHBOARD_PORT"
echo "----------------------------------------"

# Run the complete application
python run_with_landing.py 