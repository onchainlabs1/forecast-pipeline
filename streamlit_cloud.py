#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Entry point for Streamlit Cloud deployment.
This version is simplified to reduce dependencies.
"""

import os
import sys
from pathlib import Path

# Add the project root to the path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

# Set environment variables
os.environ["API_URL"] = "https://forecast-pipeline-2.onrender.com"
os.environ["DISABLE_MLFLOW"] = "True"

# Import and run the dashboard
from src.dashboard.app import main

if __name__ == "__main__":
    main() 