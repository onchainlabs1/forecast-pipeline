#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Entry point simplificado para o Streamlit Cloud.
Este arquivo é usado pelo Streamlit Cloud para iniciar o dashboard.
"""

import os
import sys
from pathlib import Path

# Adicionar a raiz do projeto ao path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

# Configurar variáveis de ambiente
os.environ["API_URL"] = "https://forecast-pipeline-2.onrender.com"
os.environ["DISABLE_MLFLOW"] = "True"

# Importar e executar o dashboard
try:
    from src.dashboard.app import main
    
    if __name__ == "__main__":
        main()
except Exception as e:
    import streamlit as st
    st.error(f"Erro ao iniciar o dashboard: {str(e)}")
    st.info("Verifique os logs para mais detalhes.")
    
    # Mostrar informações de diagnóstico
    st.write("### Informações de diagnóstico")
    st.write(f"Python version: {sys.version}")
    st.write(f"Current directory: {os.getcwd()}")
    st.write(f"Files in directory:")
    files = list(Path(".").glob("*"))
    for file in files[:20]:  # Mostrar só os primeiros 20 arquivos
        st.write(f"- {file}") 