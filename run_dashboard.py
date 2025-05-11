#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Script para iniciar o dashboard Streamlit do projeto de previsão de vendas.
"""

import os
import sys
import subprocess
import time

def main():
    """Função principal para iniciar o dashboard."""
    print("Iniciando o dashboard de previsão de vendas...")
    
    # Verifica se a API está rodando
    api_running = False
    try:
        # Tenta fazer uma requisição para a API
        import requests
        response = requests.get("http://localhost:8000/health")
        if response.status_code == 200:
            print("API já está rodando")
            api_running = True
    except:
        pass
    
    # Inicia a API se não estiver rodando
    api_process = None
    if not api_running:
        print("Iniciando a API...")
        api_cmd = [sys.executable, "-m", "src.api.main"]
        api_process = subprocess.Popen(api_cmd)
        print(f"API iniciada (PID: {api_process.pid})")
        # Espera a API iniciar
        print("Aguardando API iniciar...")
        time.sleep(5)
    
    # Inicia o dashboard
    print("Iniciando o dashboard Streamlit...")
    try:
        # Tenta usar o módulo streamlit diretamente
        streamlit_cmd = [sys.executable, "-m", "streamlit", "run", "src/dashboard/app.py"]
        subprocess.run(streamlit_cmd)
    except Exception as e:
        print(f"Erro ao iniciar o dashboard: {e}")
    
    # Encerra a API se iniciamos ela
    if api_process:
        print("Encerrando a API...")
        api_process.terminate()
        api_process.wait()
        print("API encerrada.")

if __name__ == "__main__":
    main() 