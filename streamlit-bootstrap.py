#!/usr/bin/env python

"""
Arquivo de bootstrap extremamente simples para o Streamlit Cloud.
"""

import os
import sys
import traceback
import streamlit as st

# Configurar variáveis de ambiente
os.environ["API_URL"] = "https://forecast-pipeline-2.onrender.com"
os.environ["DISABLE_MLFLOW"] = "True"

# Interface simplificada para depuração
st.set_page_config(page_title="Sales Forecast", layout="wide")
st.title("Sales Forecast Dashboard")

try:
    st.info("Inicializando aplicativo...")
    
    # Adicionar raiz do projeto ao path
    import sys
    from pathlib import Path
    project_root = Path(__file__).parent
    sys.path.insert(0, str(project_root))
    
    # Tentar importar e executar a app
    with st.spinner("Carregando dashboard..."):
        from src.dashboard.app import main
        main()
        
except Exception as e:
    st.error(f"Error initializing dashboard: {str(e)}")
    st.markdown("### Detailed Error Information")
    st.code(traceback.format_exc())
    
    # Informações do sistema
    st.markdown("### System Information")
    st.code(f"Python version: {sys.version}")
    st.code(f"Working directory: {os.getcwd()}")
    
    # Listar conteúdo do diretório
    st.markdown("### Directory Contents")
    files = list(Path(".").glob("*"))
    for file in files[:20]:
        st.text(f"- {file}")
        
    # Listar conteúdo do src/dashboard
    dashboard_dir = Path("src/dashboard")
    if dashboard_dir.exists():
        st.markdown("### Dashboard Directory Contents")
        for file in dashboard_dir.glob("*"):
            st.text(f"- {file}")
            
    # Verificar a conexão com a API
    st.markdown("### API Connection Test")
    import requests
    try:
        response = requests.get(os.environ["API_URL"] + "/health", timeout=5)
        st.json(response.json())
    except Exception as api_error:
        st.error(f"Failed to connect to API: {str(api_error)}") 