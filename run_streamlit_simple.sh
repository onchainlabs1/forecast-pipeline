#!/bin/bash

# Script to run the Streamlit dashboard directly

# Use environment variables with defaults
PYTHON_PATH=${PYTHON_PATH:-python}
DASHBOARD_PORT=${DASHBOARD_PORT:-8501}
API_PORT=${API_PORT:-8000}

# Run Streamlit
echo "Starting Streamlit dashboard on port $DASHBOARD_PORT..."
API_PORT=$API_PORT $PYTHON_PATH -m streamlit run src/dashboard/app.py --server.port $DASHBOARD_PORT "$@"

echo "Dashboard closed." 