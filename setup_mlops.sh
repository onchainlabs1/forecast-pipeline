#!/bin/bash

# Script to set up advanced MLOps components in the project

set -e  # Exits script if any command fails

echo "Starting advanced MLOps components setup..."

# Create necessary directories
echo "Creating directories..."
mkdir -p data/raw data/processed models reports mlruns
mkdir -p monitoring/grafana-dashboards monitoring/grafana-datasources
mkdir -p airflow/dags airflow/logs airflow/plugins airflow/config
mkdir -p feature_store

# Generate secret key for JWT
echo "Generating secret key for JWT authentication..."
SECRET_KEY=$(openssl rand -hex 32)
echo "SECRET_KEY=$SECRET_KEY" > .env

# Generate Fernet key for Airflow
echo "Generating Fernet key for Airflow..."
AIRFLOW_FERNET_KEY=$(python -c "from cryptography.fernet import Fernet; print(Fernet.generate_key().decode())")
echo "AIRFLOW_FERNET_KEY=$AIRFLOW_FERNET_KEY" >> .env

# Install dependencies
echo "Installing dependencies..."
pip install -r requirements.txt

# Copy Airflow DAG to the correct directory
echo "Configuring Airflow..."
cp airflow/dags/sales_forecast_dag.py airflow/dags/

# Initialize DVC
echo "Initializing DVC..."
if [ ! -d ".dvc" ]; then
    dvc init
    echo "DVC initialized."
else
    echo "DVC already initialized."
fi

# Configure MLflow
echo "Configuring MLflow..."
if [ ! -d "mlruns" ]; then
    mlflow ui --backend-store-uri ./mlruns &
    PID=$!
    sleep 5
    kill $PID
    echo "MLflow initialized."
else
    echo "MLflow already initialized."
fi

# Create Prometheus configuration file
echo "Configuring Prometheus..."
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

# Final tips
echo "Setup completed!"
echo
echo "To start the complete environment, run:"
echo "  docker-compose -f docker-compose.advanced.yml up -d"
echo
echo "To run only the API, execute:"
echo "  python src/api/main.py"
echo
echo "To access Airflow, open:"
echo "  http://localhost:8080 (user: airflow, password: airflow)"
echo
echo "To access MLflow, open:"
echo "  http://localhost:5000"
echo
echo "Before making API requests, get a JWT token with:"
echo "  curl -X POST http://localhost:8000/token -d 'username=admin&password=admin&scope=predictions:read'"
echo 