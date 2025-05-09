#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Script to run monitoring checks periodically.
Can be scheduled or run as a service in Docker.
"""

import os
import sys
import time
import logging
import argparse
import yaml
import json
import schedule
from pathlib import Path
from datetime import datetime, timedelta
from typing import Dict, Any, Optional

import pandas as pd
import joblib

# Check if MLflow should be disabled
DISABLE_MLFLOW = os.getenv("DISABLE_MLFLOW", "false").lower() == "true"

# Conditional import of MLflow
if not DISABLE_MLFLOW:
    try:
        import mlflow
        MLFLOW_AVAILABLE = True
    except ImportError:
        logging.warning("MLflow not installed, MLflow functionality will be disabled")
        MLFLOW_AVAILABLE = False
else:
    logging.info("MLflow disabled by environment variable")
    MLFLOW_AVAILABLE = False

# Add project root to path
PROJECT_DIR = Path(__file__).resolve().parents[2]
sys.path.append(str(PROJECT_DIR))

# Import project modules
from src.monitoring.model_monitoring import ModelMonitor
from src.features.feature_store import FeatureStore

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler(PROJECT_DIR / "logs" / "monitoring.log")
    ]
)
logger = logging.getLogger(__name__)

# Create logs directory if it doesn't exist
os.makedirs(PROJECT_DIR / "logs", exist_ok=True)


def load_config() -> Dict[str, Any]:
    """Load configuration from YAML file."""
    config_path = PROJECT_DIR / "config.yaml"
    
    if not config_path.exists():
        logger.error(f"Configuration file not found: {config_path}")
        return {}
    
    with open(config_path, "r") as f:
        config = yaml.safe_load(f)
    
    return config


def load_latest_data() -> Optional[pd.DataFrame]:
    """Load the latest data for monitoring checks."""
    try:
        # In a real system, this would fetch the latest production data
        # For demo purposes, we'll use the test data
        data_path = PROJECT_DIR / "data" / "processed" / "test_features.csv"
        
        if not data_path.exists():
            logger.error(f"Data file not found: {data_path}")
            return None
        
        data = pd.read_csv(data_path)
        
        # Simulate drift by modifying some values (for demo purposes)
        # In a real system, this would be actual new data
        if "sales" in data.columns:
            # Add some random drift to numeric columns
            numeric_cols = data.select_dtypes(include="number").columns
            for col in numeric_cols:
                if col != "sales":  # Don't modify the target
                    # Add a small drift factor (5% shift)
                    drift_factor = 0.05
                    data[col] = data[col] * (1 + drift_factor)
        
        return data
    
    except Exception as e:
        logger.error(f"Error loading data: {e}")
        return None


def get_latest_metrics() -> Dict[str, float]:
    """Get the latest performance metrics for the model."""
    try:
        # In a real system, this would fetch actual metrics from production
        metrics_path = PROJECT_DIR / "reports" / "metrics.json"
        
        if not metrics_path.exists():
            logger.error(f"Metrics file not found: {metrics_path}")
            return {}
        
        with open(metrics_path, "r") as f:
            metrics = json.load(f)
        
        # Simulate performance degradation (for demo purposes)
        # In a real system, these would be actual metrics
        for key, value in metrics.items():
            if "rmse" in key or "mae" in key:
                # Add 10% degradation to error metrics
                metrics[key] = value * 1.1
        
        return metrics
    
    except Exception as e:
        logger.error(f"Error getting latest metrics: {e}")
        return {}


def check_model_health(model_name: str, config: Dict[str, Any]) -> bool:
    """
    Check the health of the model.
    
    Parameters
    ----------
    model_name : str
        Name of the model to check
    config : Dict[str, Any]
        Monitoring configuration
        
    Returns
    -------
    bool
        Whether the model is healthy
    """
    logger.info(f"Starting health check for model: {model_name}")
    
    # Initialize monitoring
    monitor = ModelMonitor(model_name)
    
    # Load latest data
    latest_data = load_latest_data()
    if latest_data is None:
        logger.error("Failed to load latest data for monitoring")
        return False
    
    # Determine if MLflow is available for logging
    log_to_mlflow = MLFLOW_AVAILABLE and not DISABLE_MLFLOW
    
    # Check for data drift
    logger.info("Checking for data drift")
    drift_metrics = monitor.check_data_drift(latest_data, log_to_mlflow=log_to_mlflow)
    
    # Get latest performance metrics
    latest_metrics = get_latest_metrics()
    if not latest_metrics:
        logger.error("Failed to get latest performance metrics")
        return False
    
    # Check for performance drift
    logger.info("Checking for performance drift")
    performance_drift = monitor.check_performance_drift(latest_metrics, log_to_mlflow=log_to_mlflow)
    
    # Check if model needs retraining
    needs_retraining, reason = monitor.needs_retraining()
    if needs_retraining:
        logger.warning(f"Model needs retraining: {reason}")
        
        # In a real system, this could trigger an automatic retraining
        # or send an alert to ops team
        if config.get("scheduler", {}).get("retraining", {}).get("trigger_on_drift", False):
            logger.info("Triggering automatic retraining")
            
            # In a real system, this would actually trigger retraining
            # For demo purposes, we just log that it would happen
            logger.info("Automatic retraining would be triggered here")
        
        return False
    
    logger.info("Model is healthy")
    return True


def run_monitoring():
    """Main function to run monitoring checks."""
    # Load configuration
    config = load_config()
    if not config:
        logger.error("Failed to load configuration, exiting")
        return
    
    # Check if monitoring is enabled
    if not config.get("monitoring", {}).get("enabled", False):
        logger.info("Monitoring is disabled in configuration, exiting")
        return
    
    # Get model name from config
    model_name = config.get("mlflow", {}).get("model_name", "store-sales-forecaster")
    
    # Run monitoring check
    is_healthy = check_model_health(model_name, config)
    
    logger.info(f"Monitoring check completed. Model health: {'Good' if is_healthy else 'Needs attention'}")


def schedule_monitoring(config: Dict[str, Any]):
    """
    Schedule monitoring to run at specified intervals.
    
    Parameters
    ----------
    config : Dict[str, Any]
        Configuration containing schedule settings
    """
    check_frequency = config.get("monitoring", {}).get("check_frequency", "daily")
    
    if check_frequency == "hourly":
        logger.info("Scheduling monitoring to run hourly")
        schedule.every().hour.do(run_monitoring)
    elif check_frequency == "daily":
        logger.info("Scheduling monitoring to run daily")
        schedule.every().day.at("02:00").do(run_monitoring)  # Run at 2 AM
    elif check_frequency == "weekly":
        logger.info("Scheduling monitoring to run weekly")
        schedule.every().monday.at("02:00").do(run_monitoring)  # Run at 2 AM on Mondays
    else:
        logger.warning(f"Unknown check frequency: {check_frequency}, defaulting to daily")
        schedule.every().day.at("02:00").do(run_monitoring)  # Run at 2 AM
    
    # Run the monitoring loop
    logger.info("Starting monitoring scheduler")
    while True:
        schedule.run_pending()
        time.sleep(60)  # Check every minute


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run model monitoring checks")
    parser.add_argument("--schedule", action="store_true", help="Run monitoring on a schedule")
    parser.add_argument("--run-now", action="store_true", help="Run monitoring once immediately")
    args = parser.parse_args()
    
    # Load configuration
    config = load_config()
    
    if args.schedule:
        # Run on a schedule
        schedule_monitoring(config)
    elif args.run_now:
        # Run once immediately
        run_monitoring()
    else:
        # Default to running once
        run_monitoring() 