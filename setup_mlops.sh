#!/bin/bash

# Script para configurar componentes MLOps avançados no projeto

set -e  # Encerra script se algum comando falhar

echo "Iniciando configuração de componentes MLOps avançados..."

# Criar diretórios necessários
echo "Criando diretórios..."
mkdir -p data/raw data/processed models reports mlruns
mkdir -p monitoring/grafana-dashboards monitoring/grafana-datasources
mkdir -p airflow/dags airflow/logs airflow/plugins airflow/config
mkdir -p feature_store

# Gerar chave secreta para JWT
echo "Gerando chave secreta para autenticação JWT..."
SECRET_KEY=$(openssl rand -hex 32)
echo "SECRET_KEY=$SECRET_KEY" > .env

# Gerar chave Fernet para Airflow
echo "Gerando chave Fernet para Airflow..."
AIRFLOW_FERNET_KEY=$(python -c "from cryptography.fernet import Fernet; print(Fernet.generate_key().decode())")
echo "AIRFLOW_FERNET_KEY=$AIRFLOW_FERNET_KEY" >> .env

# Instalar dependências
echo "Instalando dependências..."
pip install -r requirements.txt

# Copiar o DAG do Airflow para o diretório correto
echo "Configurando Airflow..."
cp airflow/dags/sales_forecast_dag.py airflow/dags/

# Inicializar DVC
echo "Inicializando DVC..."
if [ ! -d ".dvc" ]; then
    dvc init
    echo "DVC inicializado."
else
    echo "DVC já inicializado."
fi

# Configurar MLflow
echo "Configurando MLflow..."
if [ ! -d "mlruns" ]; then
    mlflow ui --backend-store-uri ./mlruns &
    PID=$!
    sleep 5
    kill $PID
    echo "MLflow inicializado."
else
    echo "MLflow já inicializado."
fi

# Criar arquivo de configuração do Prometheus
echo "Configurando Prometheus..."
cat > monitoring/prometheus.yml << EOL
global:
  scrape_interval: 15s
  evaluation_interval: 15s

scrape_configs:
  - job_name: 'prometheus'
    static_configs:
      - targets: ['localhost:9090']

  - job_name: 'api'
    metrics_path: /metrics
    static_configs:
      - targets: ['api:8000']
EOL

# Dicas finais
echo "Configuração concluída!"
echo
echo "Para iniciar o ambiente completo, execute:"
echo "  docker-compose -f docker-compose.advanced.yml up -d"
echo
echo "Para executar apenas a API, execute:"
echo "  python src/api/main.py"
echo
echo "Para acessar o Airflow, abra:"
echo "  http://localhost:8080 (usuário: airflow, senha: airflow)"
echo
echo "Para acessar o MLflow, abra:"
echo "  http://localhost:5000"
echo
echo "Antes de fazer requisições à API, obtenha um token JWT com:"
echo "  curl -X POST http://localhost:8000/token -d 'username=admin&password=admin&scope=predictions:read'"
echo 