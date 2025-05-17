#!/bin/bash

# Script to run the Streamlit dashboard directly

# Use environment variable or default to "python"
PYTHON_PATH=${PYTHON_PATH:-python}

# Run Streamlit
echo "Starting Streamlit dashboard..."
$PYTHON_PATH -m streamlit run src/dashboard/app.py "$@"

echo "Dashboard closed." 