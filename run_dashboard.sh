#!/bin/bash

# Script to start the Streamlit dashboard for sales forecasting

echo "Starting Streamlit Dashboard..."

# Ensure required packages are installed
echo "Checking dependencies..."
pip install -r requirements.txt

# Define environment variables if needed
export API_URL=http://localhost:8000
export MLFLOW_URL=http://localhost:5000

# Run the Streamlit app
echo "Launching dashboard..."
streamlit run src/dashboard/app.py --server.port 8501 --browser.serverAddress localhost 