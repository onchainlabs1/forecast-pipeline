#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Script to train a LightGBM model for store sales forecasting.
"""

import os
import json
import logging
import sys
from pathlib import Path

# Add the project root directory to sys.path for relative imports
PROJECT_DIR = Path(__file__).resolve().parents[1]
sys.path.append(str(PROJECT_DIR))

import numpy as np
import pandas as pd
import lightgbm as lgb
from sklearn.metrics import mean_squared_error, mean_absolute_error
from sklearn.model_selection import TimeSeriesSplit
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

# Import utilities
# Modified import to work with the modified sys.path
from src.utils.mlflow_utils import (
    setup_mlflow,
    log_model_params,
    log_model_metrics,
    log_model,
)

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)

# Define paths
# PROJECT_DIR already defined above
PROCESSED_DATA_DIR = PROJECT_DIR / "data" / "processed"
MODELS_DIR = PROJECT_DIR / "models"
REPORTS_DIR = PROJECT_DIR / "reports"


def create_directories():
    """Create necessary directories if they don't exist."""
    os.makedirs(MODELS_DIR, exist_ok=True)
    os.makedirs(REPORTS_DIR, exist_ok=True)
    logger.info(f"Created directories: {MODELS_DIR}, {REPORTS_DIR}")


def load_processed_data():
    """Load the processed training data."""
    try:
        file_path = PROCESSED_DATA_DIR / "train_features.csv"
        if not file_path.exists():
            logger.error(f"File not found: {file_path}")
            return None
        
        logger.info(f"Loading data from {file_path}")
        data = pd.read_csv(file_path)
        logger.info(f"Loaded data shape: {data.shape}")
        return data
    
    except Exception as e:
        logger.error(f"Error loading processed data: {e}")
        return None


def prepare_features_and_target(data):
    """
    Prepare features and target variable from the processed data.
    
    Parameters
    ----------
    data : pandas.DataFrame
        The processed training data.
        
    Returns
    -------
    X : pandas.DataFrame
        Feature matrix.
    y : pandas.Series
        Target variable (sales).
    """
    if data is None:
        return None, None
    
    try:
        # For demonstration, assuming the target column is 'sales'
        # and dropping any non-feature columns
        drop_cols = ['sales', 'date', 'id']
        drop_cols = [col for col in drop_cols if col in data.columns]
        
        X = data.drop(drop_cols, axis=1)
        y = data['sales'] if 'sales' in data.columns else None
        
        # Checking column types
        object_columns = X.select_dtypes(include=['object']).columns.tolist()
        logger.info(f"Converting {len(object_columns)} object columns to categorical: {object_columns}")
        
        # Converting text columns to numeric categories
        for col in object_columns:
            X[col] = X[col].astype('category').cat.codes.astype('int32')
        
        logger.info(f"Features shape: {X.shape}, Target shape: {y.shape if y is not None else None}")
        return X, y
    
    except Exception as e:
        logger.error(f"Error preparing features and target: {e}")
        return None, None


def train_lightgbm_model(X, y, n_splits=5):
    """
    Train a LightGBM model for sales forecasting.
    
    Parameters
    ----------
    X : pandas.DataFrame
        Feature matrix.
    y : pandas.Series
        Target variable (sales).
    n_splits : int, optional (default=5)
        Number of splits for time series cross-validation.
        
    Returns
    -------
    model : lightgbm.Booster
        Trained LightGBM model.
    metrics : dict
        Performance metrics.
    params : dict
        Model parameters.
    """
    if X is None or y is None:
        logger.error("Cannot train model: features or target is None")
        return None, None, None
    
    try:
        logger.info("Training LightGBM model with time series cross-validation")
        
        # LightGBM parameters
        params = {
            'objective': 'regression',
            'metric': 'rmse',
            'boosting_type': 'gbdt',
            'learning_rate': 0.05,
            'num_leaves': 31,
            'max_depth': -1,
            'n_estimators': 100,
            'random_state': 42,
            'verbose': -1,
        }
        
        # Check if MLflow is available
        experiment_id = None
        if MLFLOW_AVAILABLE and not DISABLE_MLFLOW:
            # Set up MLflow
            experiment_id = setup_mlflow()
            logger.info(f"MLflow experiment ID: {experiment_id}")
        else:
            logger.info("MLflow is disabled or not available, continuing without it")
        
        # Use MLflow if available
        mlflow_run = None
        if experiment_id is not None:
            try:
                # Start MLflow run
                mlflow_run = mlflow.start_run(experiment_id=experiment_id)
                
                # Log model parameters
                log_model_params(params)
                
                # Log dataset info
                mlflow.log_param("n_samples", X.shape[0])
                mlflow.log_param("n_features", X.shape[1])
                mlflow.log_param("cv_folds", n_splits)
            except Exception as e:
                logger.error(f"Error starting MLflow run: {e}")
                mlflow_run = None
        
        # Time series cross-validation
        tscv = TimeSeriesSplit(n_splits=n_splits)
        cv_scores = {
            'rmse': [],
            'mae': [],
        }
        
        # Store feature importances across folds
        feature_importances = pd.DataFrame(0, index=X.columns, columns=['importance'])
        
        for fold, (train_idx, val_idx) in enumerate(tscv.split(X)):
            X_train, X_val = X.iloc[train_idx], X.iloc[val_idx]
            y_train, y_val = y.iloc[train_idx], y.iloc[val_idx]
            
            # Create dataset for LightGBM
            train_data = lgb.Dataset(X_train, label=y_train)
            val_data = lgb.Dataset(X_val, label=y_val, reference=train_data)
            
            # Train model - fixing the training parameters
            callbacks = [lgb.early_stopping(50, verbose=False)]
            model = lgb.train(
                params,
                train_data,
                valid_sets=[train_data, val_data],
                valid_names=['train', 'valid'],
                callbacks=callbacks,
            )
            
            # Get predictions
            val_preds = model.predict(X_val)
            
            # Calculate metrics
            rmse = np.sqrt(mean_squared_error(y_val, val_preds))
            mae = mean_absolute_error(y_val, val_preds)
            cv_scores['rmse'].append(rmse)
            cv_scores['mae'].append(mae)
            
            # Add feature importance from this fold
            fold_importance = pd.Series(model.feature_importance(), index=X.columns)
            feature_importances['importance'] += fold_importance
            
            logger.info(f"Fold {fold+1}/{n_splits} - RMSE: {rmse:.4f}, MAE: {mae:.4f}")
        
        # Average feature importance across folds
        feature_importances['importance'] /= n_splits
        feature_importances = feature_importances.sort_values('importance', ascending=False)
        
        # Calculate average metrics across folds
        metrics = {
            'rmse': np.mean(cv_scores['rmse']),
            'mae': np.mean(cv_scores['mae']),
            'rmse_std': np.std(cv_scores['rmse']),
            'mae_std': np.std(cv_scores['mae']),
        }
        
        # Train final model on full dataset
        logger.info("Training final model on full dataset")
        train_data = lgb.Dataset(X, label=y)
        final_model = lgb.train(params, train_data)
        
        # Log metrics and model if MLflow is available
        if mlflow_run is not None:
            try:
                # Log metrics
                log_model_metrics(metrics)
                
                # Log feature importance
                for feature, importance in feature_importances.iterrows():
                    mlflow.log_metric(f"importance_{feature}", importance['importance'])
                
                # Create and save feature importance plot
                import matplotlib.pyplot as plt
                plt.figure(figsize=(10, 8))
                plt.barh(feature_importances.head(20).index, 
                        feature_importances.head(20)['importance'])
                plt.title('Top 20 Feature Importance')
                plt.tight_layout()
                
                # Save plot
                feature_imp_path = REPORTS_DIR / "feature_importance.png"
                plt.savefig(feature_imp_path)
                plt.close()
                
                # Log plot to MLflow
                mlflow.log_artifact(str(feature_imp_path))
                
                # Log model to MLflow
                model_path = MODELS_DIR / "lightgbm_model.pkl"
                joblib.dump(final_model, model_path)
                log_model(final_model, str(model_path), "lightgbm_model", "store-sales-forecaster")
                
            except Exception as e:
                logger.error(f"Error logging to MLflow: {e}")
            finally:
                if mlflow_run:
                    mlflow.end_run()
        else:
            # If MLflow is not available, just save the model
            model_path = MODELS_DIR / "lightgbm_model.pkl"
            joblib.dump(final_model, model_path)
            logger.info(f"Model saved to {model_path}")
        
        logger.info(f"Cross-validation results - RMSE: {metrics['rmse']:.4f} (±{metrics['rmse_std']:.4f}), MAE: {metrics['mae']:.4f} (±{metrics['mae_std']:.4f})")
        
        return final_model, metrics, params
    
    except Exception as e:
        logger.error(f"Error training model: {e}")
        return None, None, None


def save_model_and_metrics(model, metrics, params):
    """
    Save the trained model and performance metrics.
    
    Parameters
    ----------
    model : lightgbm.Booster
        Trained LightGBM model.
    metrics : dict
        Performance metrics.
    params : dict
        Model parameters.
    
    Returns
    -------
    str
        Path to the saved model file.
    """
    if model is None or metrics is None:
        logger.error("Cannot save model or metrics: they are None")
        return None
    
    try:
        # Save model
        model_path = MODELS_DIR / "lightgbm_model.pkl"
        joblib.dump(model, model_path)
        logger.info(f"Model saved to {model_path}")
        
        # Save metrics
        metrics_path = REPORTS_DIR / "metrics.json"
        with open(metrics_path, 'w') as f:
            json.dump(metrics, f, indent=4)
        logger.info(f"Metrics saved to {metrics_path}")
        
        # Save parameters
        params_path = REPORTS_DIR / "params.json"
        with open(params_path, 'w') as f:
            json.dump(params, f, indent=4)
        logger.info(f"Parameters saved to {params_path}")
        
        return str(model_path)
    
    except Exception as e:
        logger.error(f"Error saving model or metrics: {e}")
        return None


def main():
    """Main function to train the model."""
    create_directories()
    
    # Load data
    data = load_processed_data()
    if data is None:
        logger.error("Failed to load data, cannot proceed with training")
        return
    
    # Prepare features and target
    X, y = prepare_features_and_target(data)
    if X is None or y is None:
        logger.error("Failed to prepare features and target, cannot proceed with training")
        return
    
    # Train model
    model, metrics, params = train_lightgbm_model(X, y)
    if model is None or metrics is None or params is None:
        logger.error("Failed to train model, cannot save results")
        return
    
    # Save model and metrics
    model_path = save_model_and_metrics(model, metrics, params)
    if model_path is None:
        logger.error("Failed to save model or metrics")
        return
    
    # Log model to MLflow
    experiment_id = setup_mlflow()
    with mlflow.start_run(experiment_id=experiment_id) as run:
        log_model(
            model,
            model_path,
            model_name="lightgbm_model",
            registered_model_name="store-sales-forecaster"
        )
    
    logger.info("Model training completed successfully")


if __name__ == "__main__":
    main()