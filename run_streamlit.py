#!/usr/bin/env python
import os
import subprocess
import sys

print("Starting Streamlit dashboard...")

# Get ports from environment variables or use defaults
API_PORT = os.environ.get("API_PORT", "8000")
DASHBOARD_PORT = os.environ.get("DASHBOARD_PORT", "8501")

# Set environment variables
os.environ["API_URL"] = f"http://localhost:{API_PORT}"
os.environ["MLFLOW_URL"] = "http://localhost:5000"
os.environ["DISABLE_MLFLOW"] = "True"  # Desabilitar MLflow para evitar erros

# Run Streamlit
streamlit_path = "streamlit"
streamlit_args = ["run", "src/dashboard/app.py", "--server.port", DASHBOARD_PORT]

try:
    subprocess.run([streamlit_path] + streamlit_args)
except Exception as e:
    print(f"Error starting Streamlit: {e}")
    sys.exit(1) 