#!/usr/bin/env python

"""
Minimal bootstrap file for Streamlit Cloud deployment.
Contains only essential dependencies for deployment stability.
"""

import os
import sys
import traceback
from pathlib import Path
import streamlit as st

# Set environment variables - API is on Render.com
os.environ["API_URL"] = "https://forecast-pipeline-2.onrender.com"
os.environ["DISABLE_MLFLOW"] = "True"

# Configure page
st.set_page_config(page_title="Sales Forecast Dashboard", layout="wide")
st.title("Store Sales Forecast Dashboard")

try:
    st.info("Initializing dashboard...")
    
    # Add project root to path
    project_root = Path(__file__).parent
    sys.path.insert(0, str(project_root))
    
    # Try to import and run the main app function
    with st.spinner("Loading dashboard..."):
        try:
            # Import with detailed error handling
            try:
                from src.dashboard.app import main
            except ImportError as import_err:
                st.error(f"Failed to import dashboard module: {str(import_err)}")
                st.code(traceback.format_exc())
                st.stop()
                
            # Run the main application
            main()
            
        except Exception as run_error:
            st.error(f"Error running dashboard: {str(run_error)}")
            st.code(traceback.format_exc())
            
except Exception as e:
    st.error(f"Initialization error: {str(e)}")
    st.markdown("### Detailed Error Information")
    st.code(traceback.format_exc())
    
    # System information
    st.markdown("### System Information")
    st.code(f"Python version: {sys.version}")
    st.code(f"Working directory: {os.getcwd()}")
    st.code(f"sys.path: {sys.path}")
    
    # List directory contents
    st.markdown("### Directory Contents")
    files = list(Path(".").glob("*"))
    for file in files[:20]:
        st.text(f"- {file}")
        
    # List src/dashboard contents if it exists
    dashboard_dir = Path("src/dashboard")
    if dashboard_dir.exists():
        st.markdown("### Dashboard Directory Contents")
        for file in dashboard_dir.glob("*"):
            st.text(f"- {file}")
            
    # Test API connection
    st.markdown("### API Connection Test")
    try:
        import requests
        response = requests.get(os.environ["API_URL"] + "/health", timeout=5)
        st.json(response.json())
    except Exception as api_error:
        st.error(f"Failed to connect to API: {str(api_error)}")

    # Show installed packages
    st.markdown("### Installed Packages")
    try:
        import pkg_resources
        installed_packages = sorted([f"{pkg.key}=={pkg.version}" for pkg in pkg_resources.working_set])
        st.code("\n".join(installed_packages))
    except Exception as pkg_err:
        st.error(f"Failed to list packages: {str(pkg_err)}") 