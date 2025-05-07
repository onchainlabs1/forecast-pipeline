"""
DAG to orchestrate the complete sales forecast pipeline.
This DAG executes the entire project workflow, from data ingestion to model deployment.
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

# Add project directory to sys.path
project_dir = Variable.get("MLPROJECT_DIR", default_var="/app")
sys.path.append(project_dir)

# Import project modules
from src.data.load_data import main as load_data
from src.data.preprocess import main as preprocess_data
from src.training_pipeline import AdvancedTrainingPipeline

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)

# Default configuration for the DAG
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

# Function to load data
def load_data_task():
    logger.info("Starting data loading")
    load_data()
    return "Data loaded successfully"

# Function to preprocess data
def preprocess_data_task():
    logger.info("Starting data preprocessing")
    preprocess_data()
    return "Data preprocessed successfully"

# Function to train the model
def train_model_task():
    logger.info("Starting model training")
    pipeline = AdvancedTrainingPipeline()
    pipeline.run_pipeline()
    return "Model trained successfully"

# Function to evaluate drift and performance
def evaluate_drift_task():
    logger.info("Evaluating model drift")
    pipeline = AdvancedTrainingPipeline()
    model_monitor = pipeline.model_monitor
    
    needs_retraining, reason = model_monitor.needs_retraining()
    if needs_retraining:
        logger.warning(f"Model needs retraining: {reason}")
        return "drift_detected"
    else:
        logger.info("Model stable, no retraining required")
        return "model_stable"

# Function to send report by email
def send_report_email(**context):
    ti = context['ti']
    training_result = ti.xcom_pull(task_ids='train_model')
    drift_result = ti.xcom_pull(task_ids='evaluate_drift')
    
    subject = "Sales Forecast Pipeline Report"
    body = f"""
    Sales Forecast Pipeline completed successfully.
    
    Training result: {training_result}
    Drift evaluation: {drift_result}
    
    For more details, access the MLflow UI.
    """
    
    send_email(
        to=default_args['email'],
        subject=subject,
        html_content=body
    )
    
    return "Email sent successfully"

# DAG definition
with DAG(
    'sales_forecast_pipeline',
    default_args=default_args,
    description='Sales forecast pipeline with MLOps',
    schedule_interval='0 1 * * 1',  # Every Monday at 1:00 AM
    catchup=False,
    tags=['mlops', 'forecast', 'sales'],
) as dag:
    
    # Task to load data
    load_data_task = PythonOperator(
        task_id='load_data',
        python_callable=load_data_task,
    )
    
    # Task to preprocess data
    preprocess_data_task = PythonOperator(
        task_id='preprocess_data',
        python_callable=preprocess_data_task,
    )
    
    # Task to train model
    train_model_task = PythonOperator(
        task_id='train_model',
        python_callable=train_model_task,
    )
    
    # Task to evaluate drift
    evaluate_drift_task = PythonOperator(
        task_id='evaluate_drift',
        python_callable=evaluate_drift_task,
    )
    
    # Task to start the API
    deploy_api_task = BashOperator(
        task_id='deploy_api',
        bash_command='cd {{ var.value.MLPROJECT_DIR }} && python src/api/main.py > api.log 2>&1 &',
    )
    
    # Task to send report by email
    send_report_task = PythonOperator(
        task_id='send_report',
        python_callable=send_report_email,
        provide_context=True,
    )
    
    # Define dependencies between tasks
    load_data_task >> preprocess_data_task >> train_model_task >> evaluate_drift_task >> deploy_api_task >> send_report_task 