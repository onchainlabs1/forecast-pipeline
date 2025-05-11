#!/usr/bin/env python

"""
Ultra-minimal Streamlit Cloud bootstrap file.
Only uses absolute minimum required packages to avoid dependency issues.
"""

import os
import sys
import streamlit as st
from pathlib import Path

# Define API endpoint and disable MLflow
os.environ["API_URL"] = "https://forecast-pipeline-2.onrender.com"
os.environ["DISABLE_MLFLOW"] = "True"

# NÃO CONFIGURAMOS A PÁGINA AQUI - A CONFIG SERÁ FEITA NO app.py
# Show loading message
st.info("Conectando à API de previsão...")

# Try to import the main module with error handling
try:
    # Add project root to path
    project_root = Path(__file__).parent
    sys.path.insert(0, str(project_root))
    
    # Test API connection
    import requests
    try:
        response = requests.get(os.environ["API_URL"] + "/health", timeout=5)
        if response.status_code == 200:
            st.success(f"Conectado com sucesso à API: {response.json()}")
        else:
            st.error(f"API retornou código de erro: {response.status_code}")
            st.json(response.json())
    except Exception as api_err:
        st.error(f"Falha ao conectar com a API: {str(api_err)}")
        st.warning("O dashboard requer conexão com a API. Verifique se a API está rodando.")
    
    # Import and run the dashboard 
    try:
        from src.dashboard.app import main
        main()
    except ImportError as e:
        st.error(f"Falha ao importar o dashboard: {str(e)}")
        st.code(f"Python path: {sys.path}")
        
        # List dashboard directory contents
        dashboard_dir = Path("src/dashboard")
        if dashboard_dir.exists():
            st.write("### Arquivos do dashboard:")
            for file in dashboard_dir.glob("*"):
                st.text(f"- {file}")
    
except Exception as e:
    st.error(f"Erro: {str(e)}")
    
    # Show system info for debugging
    st.write("### System Info")
    st.code(f"Python version: {sys.version}")
    st.code(f"Working directory: {os.getcwd()}")
    
    # Show env vars (excluding sensitive ones)
    st.write("### Environment Variables")
    env_vars = {k: v for k, v in os.environ.items() if not any(s in k.lower() for s in ['key', 'token', 'secret', 'password'])}
    st.json(env_vars)
    
    # Show installed packages
    st.write("### Installed Packages")
    try:
        import pkg_resources
        packages = sorted([f"{pkg.key}=={pkg.version}" for pkg in pkg_resources.working_set])
        st.code("\n".join(packages[:50]))  # Show first 50 packages
    except Exception as pkg_err:
        st.write(f"Error listing packages: {str(pkg_err)}") 