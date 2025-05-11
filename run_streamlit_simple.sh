#!/bin/bash

# Script para executar apenas o dashboard Streamlit
echo "Iniciando o dashboard com Streamlit..."

# Caminho para o Python do Anaconda
PYTHON_PATH="/Users/fabio/anaconda3/bin/python"

# Execute o dashboard diretamente
$PYTHON_PATH -m streamlit run src/dashboard/app.py

echo "Dashboard encerrado." 