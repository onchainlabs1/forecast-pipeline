#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Script para iniciar apenas a API do projeto de previsão de vendas.
Este script ajuda a diagnosticar problemas com a API.
"""

import os
import sys
import importlib.util
import logging

# Configurar logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)

def main():
    """Função principal para iniciar a API."""
    try:
        # Get API port from environment variable or use default
        api_port = os.environ.get("API_PORT", "8000")
        logger.info(f"Starting API on port {api_port}")
        
        # Set environment variable for the API to use
        os.environ["API_PORT"] = api_port
        
        logger.info("Tentando importar e iniciar a API...")
        
        # Importa dinamicamente o módulo da API
        api_module_path = os.path.join(os.getcwd(), "src", "api", "main.py")
        spec = importlib.util.spec_from_file_location("api_main", api_module_path)
        api_main = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(api_main)
        
        logger.info("API importada com sucesso. Serviço em execução.")
        logger.info("Pressione Ctrl+C para encerrar.")
        
        # Mantém o script rodando
        while True:
            pass
    except KeyboardInterrupt:
        logger.info("API encerrada pelo usuário.")
    except Exception as e:
        logger.error(f"Erro ao iniciar a API: {e}", exc_info=True)

if __name__ == "__main__":
    main() 