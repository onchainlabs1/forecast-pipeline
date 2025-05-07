#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Advanced training pipeline with MLOps best practices.
Integrates Feature Store, monitoring, and automated model deployment.
"""

import os
import sys
import logging
import json
import time
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Any, Optional, Tuple, Union

import pandas as pd
import numpy as np
import lightgbm as lgb
from sklearn.model_selection import TimeSeriesSplit
from sklearn.metrics import mean_squared_error, mean_absolute_error
import mlflow
import joblib

# Add project directory to path
PROJECT_DIR = Path(__file__).resolve().parents[0]
sys.path.append(str(PROJECT_DIR.parent))

# Import project modules
from src.features.feature_store import FeatureStore
from src.monitoring.model_monitoring import ModelMonitor
from src.utils.mlflow_utils import setup_mlflow, log_model, log_model_params, log_model_metrics

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)

# Define paths
DATA_DIR = PROJECT_DIR.parent / "data"
PROCESSED_DATA_DIR = DATA_DIR / "processed"
MODELS_DIR = PROJECT_DIR.parent / "models"
REPORTS_DIR = PROJECT_DIR.parent / "reports"


class AdvancedTrainingPipeline:
    """
    Advanced training pipeline that implements MLOps best practices:
    - Feature Store for managing and versioning features
    - Model monitoring for data drift and performance
    - Integration with MLflow for experiment tracking
    - Automated model deployment
    """
    
    def __init__(self, model_name: str = "store-sales-forecaster"):
        """
        Initialize the training pipeline.
        
        Parameters
        ----------
        model_name : str
            Name of the model
        """
        self.model_name = model_name
        self.feature_store = FeatureStore()
        self.model_monitor = ModelMonitor(model_name)
        self.experiment_id = None
        
        # Create necessary directories
        os.makedirs(MODELS_DIR, exist_ok=True)
        os.makedirs(REPORTS_DIR, exist_ok=True)
        
        # Initialize MLflow
        self.experiment_id = setup_mlflow()
    
    def register_features(self):
        """Register features for store sales forecasting."""
        # Basic features
        self.feature_store.register_feature(
            feature_name="store_nbr",
            description="Store number",
            feature_type="categorical",
            entity_type="store"
        )
        
        self.feature_store.register_feature(
            feature_name="family",
            description="Product family",
            feature_type="categorical",
            entity_type="product",
            transformation="one_hot"
        )
        
        self.feature_store.register_feature(
            feature_name="onpromotion",
            description="Whether item is on promotion",
            feature_type="binary",
            entity_type="product_store"
        )
        
        # Date-based features
        self.feature_store.register_feature(
            feature_name="day_of_week",
            description="Day of week (0-6)",
            feature_type="categorical",
            entity_type="date"
        )
        
        self.feature_store.register_feature(
            feature_name="day_of_month",
            description="Day of month (1-31)",
            feature_type="numerical",
            entity_type="date"
        )
        
        self.feature_store.register_feature(
            feature_name="month",
            description="Month (1-12)",
            feature_type="categorical",
            entity_type="date"
        )
        
        self.feature_store.register_feature(
            feature_name="year",
            description="Year",
            feature_type="numerical",
            entity_type="date"
        )
        
        self.feature_store.register_feature(
            feature_name="is_weekend",
            description="Whether the day is a weekend",
            feature_type="binary",
            entity_type="date",
            dependencies=["day_of_week"]
        )
        
        self.feature_store.register_feature(
            feature_name="is_month_start",
            description="Whether the day is the start of month",
            feature_type="binary",
            entity_type="date"
        )
        
        self.feature_store.register_feature(
            feature_name="is_month_end",
            description="Whether the day is the end of month",
            feature_type="binary",
            entity_type="date"
        )
        
        # Sales lag features
        for lag in [1, 7, 14, 28]:
            self.feature_store.register_feature(
                feature_name=f"sales_lag_{lag}",
                description=f"Sales lagged by {lag} days",
                feature_type="numerical",
                entity_type="product_store",
                transformation="standardize"
            )
        
        # Rolling window features
        for window in [7, 14, 30]:
            self.feature_store.register_feature(
                feature_name=f"sales_rolling_mean_{window}",
                description=f"Mean sales over {window} days",
                feature_type="numerical",
                entity_type="product_store",
                transformation="standardize"
            )
            
            self.feature_store.register_feature(
                feature_name=f"sales_rolling_std_{window}",
                description=f"Standard deviation of sales over {window} days",
                feature_type="numerical",
                entity_type="product_store",
                transformation="standardize"
            )
        
        # External features
        self.feature_store.register_feature(
            feature_name="dcoilwtico",
            description="Oil price",
            feature_type="numerical",
            entity_type="external",
            transformation="standardize"
        )
        
        self.feature_store.register_feature(
            feature_name="is_holiday",
            description="Whether the day is a holiday",
            feature_type="binary",
            entity_type="date"
        )
        
        # Create feature groups
        self.feature_store.create_feature_group(
            group_name="basic_features",
            feature_list=["store_nbr", "family", "onpromotion"],
            description="Basic store and product features"
        )
        
        self.feature_store.create_feature_group(
            group_name="date_features",
            feature_list=["day_of_week", "day_of_month", "month", "year", 
                          "is_weekend", "is_month_start", "is_month_end"],
            description="Date-based features"
        )
        
        self.feature_store.create_feature_group(
            group_name="lag_features",
            feature_list=[f"sales_lag_{lag}" for lag in [1, 7, 14, 28]],
            description="Lagged sales features"
        )
        
        self.feature_store.create_feature_group(
            group_name="rolling_features",
            feature_list=[f"sales_rolling_mean_{window}" for window in [7, 14, 30]] +
                         [f"sales_rolling_std_{window}" for window in [7, 14, 30]],
            description="Rolling window features"
        )
        
        self.feature_store.create_feature_group(
            group_name="external_features",
            feature_list=["dcoilwtico", "is_holiday"],
            description="External factors like oil price and holidays"
        )
        
        logger.info("All features registered in Feature Store")
    
    def load_and_prepare_data(self) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """
        Load data and prepare features using the Feature Store.
        
        Returns
        -------
        Tuple[pd.DataFrame, pd.DataFrame]
            Training data and test data with prepared features
        """
        # Load processed data
        train_path = PROCESSED_DATA_DIR / "train_features.csv"
        test_path = PROCESSED_DATA_DIR / "test_features.csv"
        
        if not train_path.exists() or not test_path.exists():
            logger.error("Processed data files not found. Run preprocess.py first.")
            return None, None
        
        logger.info("Loading processed data files")
        train_data = pd.read_csv(train_path)
        test_data = pd.read_csv(test_path)
        
        # Get all feature groups
        all_features = []
        for group_name in self.feature_store.list_feature_groups():
            group = self.feature_store.get_feature_group(group_name)
            all_features.extend(group.get("features", []))
        
        # Generate features using Feature Store
        logger.info("Generating features using Feature Store")
        train_features = self.feature_store.generate_features(
            data=train_data,
            feature_list=all_features,
            mode="training"
        )
        
        test_features = self.feature_store.generate_features(
            data=test_data,
            feature_list=all_features,
            mode="inference"
        )
        
        logger.info(f"Prepared training data: {train_features.shape}")
        logger.info(f"Prepared test data: {test_features.shape}")
        
        return train_features, test_features
    
    def train_model(self, X_train: pd.DataFrame, y_train: pd.Series, 
                   X_val: Optional[pd.DataFrame] = None, 
                   y_val: Optional[pd.Series] = None) -> Tuple[Any, Dict[str, float]]:
        """
        Train a LightGBM model for sales forecasting.
        
        Parameters
        ----------
        X_train : pd.DataFrame
            Training features
        y_train : pd.Series
            Training target
        X_val : pd.DataFrame, optional
            Validation features
        y_val : pd.Series, optional
            Validation target
            
        Returns
        -------
        Tuple[Any, Dict[str, float]]
            Trained model and performance metrics
        """
        logger.info("Training LightGBM model")
        
        # Model parameters
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
        
        # Start MLflow run
        with mlflow.start_run(experiment_id=self.experiment_id, run_name=f"{self.model_name}_{datetime.now().strftime('%Y%m%d_%H%M%S')}") as run:
            run_id = run.info.run_id
            
            # Log model parameters
            log_model_params(params)
            mlflow.log_param("n_train_samples", len(X_train))
            
            # Create and train LightGBM dataset
            train_data = lgb.Dataset(X_train, label=y_train)
            
            if X_val is not None and y_val is not None:
                valid_data = lgb.Dataset(X_val, label=y_val, reference=train_data)
                valid_sets = [train_data, valid_data]
                valid_names = ['train', 'valid']
                mlflow.log_param("n_val_samples", len(X_val))
            else:
                valid_sets = [train_data]
                valid_names = ['train']
            
            # Train with early stopping
            callbacks = [lgb.early_stopping(50, verbose=True)]
            model = lgb.train(
                params,
                train_data,
                valid_sets=valid_sets,
                valid_names=valid_names,
                callbacks=callbacks,
            )
            
            # Calculate metrics
            y_train_pred = model.predict(X_train)
            train_rmse = np.sqrt(mean_squared_error(y_train, y_train_pred))
            train_mae = mean_absolute_error(y_train, y_train_pred)
            
            metrics = {
                "train_rmse": train_rmse,
                "train_mae": train_mae
            }
            
            if X_val is not None and y_val is not None:
                y_val_pred = model.predict(X_val)
                val_rmse = np.sqrt(mean_squared_error(y_val, y_val_pred))
                val_mae = mean_absolute_error(y_val, y_val_pred)
                metrics["val_rmse"] = val_rmse
                metrics["val_mae"] = val_mae
            
            # Log metrics
            log_model_metrics(metrics)
            
            # Plot and log feature importance
            feature_importance = pd.DataFrame({
                'feature': X_train.columns,
                'importance': model.feature_importance()
            }).sort_values('importance', ascending=False)
            
            # Log top 20 feature importances
            for feature, importance in feature_importance.head(20).iterrows():
                mlflow.log_metric(f"importance_{importance['feature']}", importance['importance'])
            
            # Log feature importance plot
            import matplotlib.pyplot as plt
            plt.figure(figsize=(10, 8))
            plt.barh(feature_importance.head(20)['feature'], feature_importance.head(20)['importance'])
            plt.title('Feature Importance')
            plt.tight_layout()
            
            # Save plot
            feature_imp_path = REPORTS_DIR / "feature_importance.png"
            plt.savefig(feature_imp_path)
            plt.close()
            
            # Log plot to MLflow
            mlflow.log_artifact(str(feature_imp_path), "plots")
            
            # Log model
            model_path = MODELS_DIR / f"{self.model_name}.pkl"
            joblib.dump(model, model_path)
            
            # Register model in MLflow
            log_model(
                model=model,
                artifact_path=str(model_path),
                model_name=self.model_name,
                registered_model_name=self.model_name
            )
            
            logger.info(f"Model trained and registered: {self.model_name}")
            logger.info(f"Train RMSE: {train_rmse:.4f}, Train MAE: {train_mae:.4f}")
            if X_val is not None:
                logger.info(f"Validation RMSE: {val_rmse:.4f}, Validation MAE: {val_mae:.4f}")
            
            return model, metrics
    
    def evaluate_model(self, model: Any, X_test: pd.DataFrame, y_test: pd.Series) -> Dict[str, float]:
        """
        Evaluate model on test data.
        
        Parameters
        ----------
        model : Any
            Trained model
        X_test : pd.DataFrame
            Test features
        y_test : pd.Series
            Test target
            
        Returns
        -------
        Dict[str, float]
            Evaluation metrics
        """
        logger.info("Evaluating model on test data")
        
        # Generate predictions
        y_pred = model.predict(X_test)
        
        # Calculate metrics
        test_rmse = np.sqrt(mean_squared_error(y_test, y_pred))
        test_mae = mean_absolute_error(y_test, y_pred)
        
        metrics = {
            "test_rmse": test_rmse,
            "test_mae": test_mae
        }
        
        # Save metrics
        metrics_path = REPORTS_DIR / "test_metrics.json"
        with open(metrics_path, "w") as f:
            json.dump(metrics, f, indent=2)
        
        # Plot actual vs predicted
        import matplotlib.pyplot as plt
        plt.figure(figsize=(12, 6))
        plt.plot(y_test.values[:100], label='Actual', marker='o')
        plt.plot(y_pred[:100], label='Predicted', marker='x')
        plt.title('Actual vs Predicted Sales (First 100 Samples)')
        plt.legend()
        plt.tight_layout()
        
        # Save plot
        pred_plot_path = REPORTS_DIR / "actual_vs_predicted.png"
        plt.savefig(pred_plot_path)
        plt.close()
        
        # Log to MLflow
        with mlflow.start_run(experiment_id=self.experiment_id) as run:
            mlflow.log_metrics(metrics)
            mlflow.log_artifact(str(pred_plot_path), "plots")
            mlflow.log_artifact(str(metrics_path), "metrics")
        
        logger.info(f"Test RMSE: {test_rmse:.4f}, Test MAE: {test_mae:.4f}")
        
        return metrics
    
    def setup_monitoring(self, X_train: pd.DataFrame, metrics: Dict[str, float]):
        """
        Set up model monitoring with baseline data.
        
        Parameters
        ----------
        X_train : pd.DataFrame
            Training features for baseline distribution
        metrics : Dict[str, float]
            Performance metrics for baseline
        """
        logger.info("Setting up model monitoring")
        
        # Set numerical and categorical columns
        num_cols = X_train.select_dtypes(include=np.number).columns.tolist()
        cat_cols = X_train.select_dtypes(include=["object", "category"]).columns.tolist()
        
        # Set up monitoring baseline
        self.model_monitor.set_baseline(
            data=X_train,
            performance_metrics=metrics,
            numerical_columns=num_cols,
            categorical_columns=cat_cols
        )
        
        logger.info("Model monitoring baseline established")
    
    def run_pipeline(self, setup_features: bool = True):
        """
        Run the complete training pipeline.
        
        Parameters
        ----------
        setup_features : bool
            Whether to set up the feature registry first
        """
        logger.info(f"Starting advanced training pipeline for {self.model_name}")
        
        start_time = time.time()
        
        # Step 1: Set up feature registry
        if setup_features:
            logger.info("Step 1: Setting up feature registry")
            self.register_features()
        
        # Step 2: Load and prepare data
        logger.info("Step 2: Loading and preparing data")
        train_data, test_data = self.load_and_prepare_data()
        
        if train_data is None or test_data is None:
            logger.error("Failed to load data, stopping pipeline")
            return
        
        # Step 3: Split data for training
        logger.info("Step 3: Splitting data for training")
        
        # Define features and target
        target_col = "sales"
        if target_col not in train_data.columns:
            logger.error(f"Target column {target_col} not found in data")
            return
        
        # Split data
        X_train = train_data.drop(target_col, axis=1)
        y_train = train_data[target_col]
        
        X_test = test_data.drop(target_col, axis=1) if target_col in test_data.columns else test_data
        y_test = test_data[target_col] if target_col in test_data.columns else None
        
        # Step 4: Train model
        logger.info("Step 4: Training model")
        # Use time series split for validation
        tscv = TimeSeriesSplit(n_splits=5)
        for train_idx, val_idx in tscv.split(X_train):
            X_train_fold, X_val_fold = X_train.iloc[train_idx], X_train.iloc[val_idx]
            y_train_fold, y_val_fold = y_train.iloc[train_idx], y_train.iloc[val_idx]
            
            # We'll just use the last fold for validation
            pass
        
        model, metrics = self.train_model(X_train, y_train, X_val_fold, y_val_fold)
        
        # Step 5: Evaluate model
        if y_test is not None:
            logger.info("Step 5: Evaluating model on test data")
            test_metrics = self.evaluate_model(model, X_test, y_test)
        else:
            logger.info("Skipping test evaluation as test labels are not available")
            test_metrics = {}
        
        # Step 6: Set up monitoring
        logger.info("Step 6: Setting up model monitoring")
        self.setup_monitoring(X_train, metrics)
        
        # Step 7: Check if model needs retraining (for existing models)
        logger.info("Step 7: Checking if model needs retraining")
        needs_retraining, reason = self.model_monitor.needs_retraining()
        if needs_retraining:
            logger.info(f"Model needs retraining: {reason}")
        else:
            logger.info("Model does not need retraining")
        
        # Finish timing and report
        elapsed_time = time.time() - start_time
        logger.info(f"Pipeline completed in {elapsed_time:.2f} seconds")
        
        return model, metrics


if __name__ == "__main__":
    # Run the pipeline
    pipeline = AdvancedTrainingPipeline()
    pipeline.run_pipeline() 