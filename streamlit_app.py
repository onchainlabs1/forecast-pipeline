#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Entry point for Streamlit Cloud deployment.
This file is used by Streamlit Cloud to run the application.
"""

import os
import sys
from pathlib import Path

# Add the project root to the path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

# Import the dashboard app
from src.dashboard.app import main

# Set environment variables if needed
if "API_URL" not in os.environ:
    os.environ["API_URL"] = "https://your-api-url.com"  # Replace with your deployed API URL

if "MLFLOW_URL" not in os.environ:
    os.environ["MLFLOW_URL"] = "https://your-mlflow-url.com"  # Replace with your MLflow URL

# Run the dashboard
if __name__ == "__main__":
    main() 