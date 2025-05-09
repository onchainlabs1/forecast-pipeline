#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Ponto de entrada simplificado para o Streamlit Cloud.
"""

import os
import sys
import traceback
from pathlib import Path

# Configurar variáveis de ambiente
os.environ["API_URL"] = "https://forecast-pipeline-2.onrender.com"
os.environ["DISABLE_MLFLOW"] = "True"

# Adicionar raiz do projeto ao path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

# Tentar importar e executar o dashboard
try:
    import streamlit as st
    from src.dashboard.app import main
    
    if __name__ == "__main__":
        main()
except ImportError as e:
    # Mostrar informações de diagnóstico
    print(f"Erro de importação: {e}")
    import streamlit as st
    st.error(f"Erro ao iniciar o dashboard: {e}")
    st.write("### Diagnóstico")
    st.code(traceback.format_exc())
    st.write(f"Python: {sys.version}")
    st.write(f"Path: {sys.path}")
except Exception as e:
    # Mostrar informações gerais de erro
    import streamlit as st
    st.error(f"Erro ao iniciar o dashboard: {e}")
    st.write("### Diagnóstico")
    st.code(traceback.format_exc())
    st.write(f"Python: {sys.version}")
    st.write(f"Path: {sys.path}")
    
    # Listar arquivos
    st.write("### Arquivos no diretório")
    files = list(Path(".").glob("*"))
    for file in files[:20]:
        st.write(f"- {file}") 