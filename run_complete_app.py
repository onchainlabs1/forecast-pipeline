#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Script para iniciar a API e o dashboard do projeto de previsão de vendas.
Este script verifica se a API está rodando, inicia-a se necessário, e depois inicia o dashboard.
"""

import os
import sys
import subprocess
import time
import logging
import socket
import requests
from pathlib import Path

# Configurar logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)

# Definir caminhos
PYTHON_PATH = "/Users/fabio/anaconda3/bin/python"
API_PORT = 8000
API_HOST = "localhost"
API_MODULE = "src.api.main"
DASHBOARD_MODULE = "src.dashboard.app"

def is_port_in_use(port, host='localhost'):
    """Verifica se uma porta está em uso."""
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        return s.connect_ex((host, port)) == 0

def is_api_healthy():
    """Verifica se a API está respondendo corretamente."""
    try:
        response = requests.get(f"http://{API_HOST}:{API_PORT}/health")
        return response.status_code == 200
    except Exception as e:
        logger.error(f"Erro verificando API: {e}")
        return False

def init_database():
    """Inicializa o banco de dados se necessário."""
    db_path = Path("data/db/sales_forecasting.db")
    
    if db_path.exists():
        logger.info("Banco de dados já existe. Pulando inicialização.")
        return True
        
    logger.info("Inicializando banco de dados...")
    
    # Criar diretórios necessários
    os.makedirs("data/db", exist_ok=True)
    os.makedirs("data/raw", exist_ok=True)
    
    try:
        # Execute o módulo de inicialização do banco de dados
        subprocess.run(
            [PYTHON_PATH, "-m", "src.database.init_db"],
            check=True
        )
        logger.info("Banco de dados inicializado com sucesso.")
        return True
    except subprocess.CalledProcessError as e:
        logger.error(f"Erro ao inicializar banco de dados: {e}")
        return False

def start_api():
    """Inicia a API em um processo separado."""
    logger.info("Iniciando API...")
    
    # Verifica se a API já está rodando
    if is_port_in_use(API_PORT):
        if is_api_healthy():
            logger.info(f"API já está rodando em {API_HOST}:{API_PORT}")
            return True
        else:
            logger.warning("API está rodando, mas não está respondendo corretamente.")
            return False
    
    # Inicia a API em um processo separado
    try:
        api_process = subprocess.Popen(
            [PYTHON_PATH, "-m", API_MODULE],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            universal_newlines=True
        )
        
        # Aguarda um tempo para a API iniciar
        logger.info("Aguardando API iniciar...")
        for _ in range(10):
            time.sleep(1)
            if is_api_healthy():
                logger.info("API iniciada com sucesso!")
                return True
                
        logger.error("Tempo limite excedido ao aguardar API iniciar.")
        return False
    except Exception as e:
        logger.error(f"Erro ao iniciar API: {e}")
        return False

def start_dashboard():
    """Inicia o dashboard Streamlit."""
    logger.info("Iniciando dashboard...")
    
    try:
        # Inicia o dashboard (este processo bloqueia até o dashboard ser fechado)
        subprocess.run(
            [PYTHON_PATH, "-m", "streamlit", "run", f"src/dashboard/app.py"],
            check=True
        )
        logger.info("Dashboard encerrado.")
        return True
    except subprocess.CalledProcessError as e:
        logger.error(f"Erro ao executar dashboard: {e}")
        return False
    except KeyboardInterrupt:
        logger.info("Dashboard interrompido pelo usuário.")
        return True

def main():
    """Função principal."""
    logger.info("Iniciando aplicação de previsão de vendas...")
    
    # Inicializar banco de dados
    if not init_database():
        logger.error("Falha ao inicializar banco de dados. Abortando.")
        return 1
    
    # Iniciar API
    if not start_api():
        logger.error("Falha ao iniciar API. Abortando.")
        return 1
    
    # Iniciar dashboard
    if not start_dashboard():
        logger.error("Falha ao iniciar dashboard.")
        return 1
    
    return 0

if __name__ == "__main__":
    sys.exit(main()) 