#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Utility functions for MLflow tracking and model registry.
"""

import os
import logging
from pathlib import Path

import mlflow
from mlflow.tracking import MlflowClient

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)

# Define paths and constants
PROJECT_DIR = Path(__file__).resolve().parents[2]
MLFLOW_TRACKING_URI = os.getenv("MLFLOW_TRACKING_URI", "mlruns")
EXPERIMENT_NAME = "store-sales-forecasting"


def setup_mlflow(disable_for_production=False):
    """
    Set up MLflow tracking.
    
    Parameters
    ----------
    disable_for_production : bool, optional (default=False)
        If True, MLflow will be disabled in production environment.
    
    Returns
    -------
    str
        The active experiment ID or None if MLflow is disabled.
    """
    try:
        # Check if MLflow should be disabled in production
        if disable_for_production and os.getenv("ENVIRONMENT") == "production":
            logger.info("MLflow disabled in production environment")
            return None
            
        # Set the tracking URI
        mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
        logger.info(f"MLflow tracking URI set to: {MLFLOW_TRACKING_URI}")
        
        # Get or create the experiment
        experiment = mlflow.get_experiment_by_name(EXPERIMENT_NAME)
        if experiment is None:
            experiment_id = mlflow.create_experiment(EXPERIMENT_NAME)
            logger.info(f"Created new experiment: {EXPERIMENT_NAME} (ID: {experiment_id})")
        else:
            experiment_id = experiment.experiment_id
            logger.info(f"Using existing experiment: {EXPERIMENT_NAME} (ID: {experiment_id})")
        
        # Set the active experiment
        mlflow.set_experiment(EXPERIMENT_NAME)
        
        return experiment_id
    
    except Exception as e:
        logger.error(f"Error setting up MLflow: {e}")
        return None


def log_model_params(params):
    """
    Log model parameters to MLflow.
    
    Parameters
    ----------
    params : dict
        Dictionary of model parameters.
    """
    try:
        for key, value in params.items():
            mlflow.log_param(key, value)
        logger.info("Model parameters logged to MLflow")
    
    except Exception as e:
        logger.error(f"Error logging parameters to MLflow: {e}")


def log_model_metrics(metrics):
    """
    Log model metrics to MLflow.
    
    Parameters
    ----------
    metrics : dict
        Dictionary of model metrics.
    """
    try:
        for key, value in metrics.items():
            mlflow.log_metric(key, value)
        logger.info("Model metrics logged to MLflow")
    
    except Exception as e:
        logger.error(f"Error logging metrics to MLflow: {e}")


def log_model(model, model_path, model_name="lightgbm_model", registered_model_name=None):
    """
    Log a model to MLflow and optionally register it.
    
    Parameters
    ----------
    model : object
        The trained model to log.
    model_path : str
        Local path where the model is saved.
    model_name : str, optional (default="lightgbm_model")
        Name of the model artifact in MLflow.
    registered_model_name : str, optional
        If provided, register the model with this name.
    
    Returns
    -------
    str
        The run ID of the MLflow run.
    """
    try:
        # Log the model artifact
        mlflow.lightgbm.log_model(
            model,
            model_name,
            registered_model_name=registered_model_name
        )
        
        # Log the model file as an artifact
        mlflow.log_artifact(model_path, model_name)
        
        logger.info(f"Model logged to MLflow as '{model_name}'")
        
        # If registration is requested
        if registered_model_name:
            logger.info(f"Model registered as '{registered_model_name}'")
        
        # Get the run ID
        run_id = mlflow.active_run().info.run_id
        return run_id
    
    except Exception as e:
        logger.error(f"Error logging model to MLflow: {e}")
        return None


def get_latest_model_version(model_name):
    """
    Get the latest version of a registered model.
    
    Parameters
    ----------
    model_name : str
        Name of the registered model.
    
    Returns
    -------
    model_version : str
        The latest version of the model.
    """
    try:
        client = MlflowClient()
        latest_version = client.get_latest_versions(model_name, stages=["None"])[0].version
        logger.info(f"Latest version of model '{model_name}' is {latest_version}")
        return latest_version
    
    except Exception as e:
        logger.error(f"Error getting latest model version: {e}")
        return None


def transition_model_stage(model_name, version, stage="Production"):
    """
    Transition a model version to a different stage.
    
    Parameters
    ----------
    model_name : str
        Name of the registered model.
    version : str
        Version of the model to transition.
    stage : str, optional (default="Production")
        Target stage for the model version.
    
    Returns
    -------
    bool
        True if successful, False otherwise.
    """
    try:
        client = MlflowClient()
        client.transition_model_version_stage(
            name=model_name,
            version=version,
            stage=stage
        )
        logger.info(f"Model '{model_name}' version {version} transitioned to stage '{stage}'")
        return True
    
    except Exception as e:
        logger.error(f"Error transitioning model stage: {e}")
        return False