"""
DAG para orquestrar o pipeline completo de previsão de vendas.
Este DAG executa todo o fluxo de trabalho do projeto, desde a ingestão de dados até o deploy do modelo.
"""

from datetime import datetime, timedelta
from airflow import DAG
from airflow.operators.python import PythonOperator
from airflow.operators.bash import BashOperator
from airflow.models import Variable
from airflow.utils.email import send_email
import os
import sys
import logging

# Adicione o diretório do projeto ao sys.path
project_dir = Variable.get("MLPROJECT_DIR", default_var="/app")
sys.path.append(project_dir)

# Importar módulos do projeto
from src.data.load_data import main as load_data
from src.data.preprocess import main as preprocess_data
from src.training_pipeline import AdvancedTrainingPipeline

# Configurar logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)

# Configuração padrão para o DAG
default_args = {
    'owner': 'mlops',
    'depends_on_past': False,
    'start_date': datetime(2023, 1, 1),
    'email': ['your_email@example.com'],
    'email_on_failure': True,
    'email_on_retry': False,
    'retries': 1,
    'retry_delay': timedelta(minutes=5),
}

# Função para carregar dados
def load_data_task():
    logger.info("Iniciando carregamento de dados")
    load_data()
    return "Dados carregados com sucesso"

# Função para pré-processar dados
def preprocess_data_task():
    logger.info("Iniciando pré-processamento de dados")
    preprocess_data()
    return "Dados pré-processados com sucesso"

# Função para treinar modelo
def train_model_task():
    logger.info("Iniciando treinamento do modelo")
    pipeline = AdvancedTrainingPipeline()
    pipeline.run_pipeline()
    return "Modelo treinado com sucesso"

# Função para avaliar drift e performance
def evaluate_drift_task():
    logger.info("Avaliando drift do modelo")
    pipeline = AdvancedTrainingPipeline()
    model_monitor = pipeline.model_monitor
    
    needs_retraining, reason = model_monitor.needs_retraining()
    if needs_retraining:
        logger.warning(f"Modelo precisa de retreinamento: {reason}")
        return "drift_detected"
    else:
        logger.info("Modelo estável, não requer retreinamento")
        return "model_stable"

# Função para enviar relatório por email
def send_report_email(**context):
    ti = context['ti']
    training_result = ti.xcom_pull(task_ids='train_model')
    drift_result = ti.xcom_pull(task_ids='evaluate_drift')
    
    subject = "Relatório de Pipeline de Previsão de Vendas"
    body = f"""
    Pipeline de Previsão de Vendas completado com sucesso.
    
    Resultado do treinamento: {training_result}
    Avaliação de drift: {drift_result}
    
    Para mais detalhes, acesse o MLflow UI.
    """
    
    send_email(
        to=default_args['email'],
        subject=subject,
        html_content=body
    )
    
    return "Email enviado com sucesso"

# Definição do DAG
with DAG(
    'sales_forecast_pipeline',
    default_args=default_args,
    description='Pipeline de previsão de vendas com MLOps',
    schedule_interval='0 1 * * 1',  # Toda segunda-feira à 1:00 AM
    catchup=False,
    tags=['mlops', 'forecast', 'sales'],
) as dag:
    
    # Task para carregar dados
    load_data_task = PythonOperator(
        task_id='load_data',
        python_callable=load_data_task,
    )
    
    # Task para pré-processar dados
    preprocess_data_task = PythonOperator(
        task_id='preprocess_data',
        python_callable=preprocess_data_task,
    )
    
    # Task para treinar modelo
    train_model_task = PythonOperator(
        task_id='train_model',
        python_callable=train_model_task,
    )
    
    # Task para avaliar drift
    evaluate_drift_task = PythonOperator(
        task_id='evaluate_drift',
        python_callable=evaluate_drift_task,
    )
    
    # Task para iniciar a API
    deploy_api_task = BashOperator(
        task_id='deploy_api',
        bash_command='cd {{ var.value.MLPROJECT_DIR }} && python src/api/main.py > api.log 2>&1 &',
    )
    
    # Task para enviar relatório por email
    send_report_task = PythonOperator(
        task_id='send_report',
        python_callable=send_report_email,
        provide_context=True,
    )
    
    # Definir dependências entre as tasks
    load_data_task >> preprocess_data_task >> train_model_task >> evaluate_drift_task >> deploy_api_task >> send_report_task 