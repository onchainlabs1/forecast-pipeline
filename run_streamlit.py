#!/usr/bin/env python
import os
import subprocess
import sys

print("Starting Streamlit dashboard...")

# Set environment variables
os.environ["API_URL"] = "http://localhost:8000"
os.environ["MLFLOW_URL"] = "http://localhost:5000"
os.environ["DISABLE_MLFLOW"] = "True"  # Desabilitar MLflow para evitar erros

# Run Streamlit
streamlit_path = "streamlit"
streamlit_args = ["run", "src/dashboard/app.py", "--server.port", "8501"]

try:
    subprocess.run([streamlit_path] + streamlit_args)
except Exception as e:
    print(f"Error starting Streamlit: {e}")
    sys.exit(1) 