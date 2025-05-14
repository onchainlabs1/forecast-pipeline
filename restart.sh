#!/bin/bash

echo "=== Forecast Pipeline Dashboard - Script de Inicialização ==="
echo "Este script irá iniciar todos os serviços necessários para a aplicação."

# Diretório base do projeto
BASE_DIR="/Users/fabio/Desktop/mlproject"
PYTHON="/Users/fabio/anaconda3/bin/python"

# Matar processos existentes (se houver)
echo "Terminando processos existentes..."
pkill -f "uvicorn src.landing" || echo "Nenhum processo de landing page em execução"
pkill -f "uvicorn main:app" || echo "Nenhum processo de API em execução"
pkill -f "uvicorn src.api.main:app" || echo "Nenhum processo de API em execução"
pkill -f "streamlit run" || echo "Nenhum processo de Dashboard em execução"

# Esperar um pouco para garantir que os processos foram encerrados
echo "Aguardando finalização completa dos processos..."
sleep 3

# Verificar portas em uso
echo "Verificando portas em uso..."
if lsof -i:8000 > /dev/null 2>&1; then
  echo "AVISO: Porta 8000 já está em uso. A landing page pode não iniciar corretamente."
fi

if lsof -i:8002 > /dev/null 2>&1; then
  echo "AVISO: Porta 8002 já está em uso. A API pode não iniciar corretamente."
fi

if lsof -i:8501 > /dev/null 2>&1; then
  echo "AVISO: Porta 8501 já está em uso. O Dashboard pode não iniciar corretamente."
fi

# Iniciar landing page em background
echo "Iniciando Landing Page (http://localhost:8000)..."
cd "$BASE_DIR" && $PYTHON -m uvicorn src.landing.server:app --host 0.0.0.0 --port 8000 > logs/landing.log 2>&1 &
LANDING_PID=$!
echo "Landing Page iniciada com PID $LANDING_PID"

# Aguardar inicialização da landing page
echo "Aguardando inicialização da landing page..."
sleep 5

# Iniciar API em background
echo "Iniciando API (http://localhost:8002)..."
cd "$BASE_DIR/src/api" && $PYTHON -m uvicorn main:app --host 0.0.0.0 --port 8002 > ../../logs/api.log 2>&1 &
API_PID=$!
echo "API iniciada com PID $API_PID"

# Aguardar inicialização da API
echo "Aguardando inicialização da API..."
sleep 5

# Iniciar dashboard em background
echo "Iniciando Dashboard (http://localhost:8501)..."
cd "$BASE_DIR" && $PYTHON -m streamlit run src/dashboard/app.py --server.port=8501 > logs/dashboard.log 2>&1 &
DASHBOARD_PID=$!
echo "Dashboard iniciado com PID $DASHBOARD_PID"

# Criar diretório de logs se não existir
mkdir -p "$BASE_DIR/logs"

# Mostrar informações finais
echo ""
echo "=== Todos os serviços iniciados ==="
echo "Landing Page: http://localhost:8000"
echo "API: http://localhost:8002"
echo "Dashboard: http://localhost:8501"
echo ""
echo "Credenciais de acesso:"
echo "Usuário: admin"
echo "Senha: admin"
echo ""
echo "Para encerrar todos os serviços, execute: bash $BASE_DIR/restart.sh stop"
echo "Os logs estão disponíveis no diretório: $BASE_DIR/logs/"

# Opção para parar todos os serviços
if [ "$1" = "stop" ]; then
  echo "Encerrando todos os serviços..."
  pkill -f "uvicorn src.landing" || echo "Nenhum processo de landing page em execução"
  pkill -f "uvicorn main:app" || echo "Nenhum processo de API em execução" 
  pkill -f "uvicorn src.api.main:app" || echo "Nenhum processo de API em execução"
  pkill -f "streamlit run" || echo "Nenhum processo de Dashboard em execução"
  echo "Todos os serviços encerrados."
fi 