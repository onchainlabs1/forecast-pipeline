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

# Add project directory to path
PROJECT_DIR = Path(__file__).resolve().parents[0]
sys.path.append(str(PROJECT_DIR.parent))

# Import project modules
from src.features.feature_store import FeatureStore
from src.monitoring.model_monitoring import ModelMonitor
from src.utils.mlflow_utils import setup_mlflow, log_model, log_model_params, log_model_metrics
from src.features.enhanced_features import EnhancedFeatureGenerator
from src.features.config_loader import FeatureConfig

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
    Advanced training pipeline with enhanced feature generation.
    """
    
    def __init__(
        self,
        model_name: str = "store-sales-forecaster",
        data_dir: Optional[Path] = None,
        config_dir: Optional[Path] = None
    ):
        """
        Initialize training pipeline.
        
        Parameters
        ----------
        model_name : str
            Name of the model for MLflow tracking
        data_dir : Path, optional
            Directory containing data files. If not provided,
            will use default directory relative to project root.
        config_dir : Path, optional
            Directory containing configuration files. If not provided,
            will use default directory relative to project root.
        """
        self.model_name = model_name
        
        # Set up directories
        if data_dir is None:
            data_dir = Path(__file__).parent.parent / "data"
        if config_dir is None:
            config_dir = Path(__file__).parent.parent / "config"
        
        self.data_dir = Path(data_dir)
        self.processed_data_dir = self.data_dir / "processed"
        
        # Initialize components
        self.feature_store = FeatureStore()
        self.feature_config = FeatureConfig(config_dir / "feature_config.yaml")
        self.enhanced_features = EnhancedFeatureGenerator()
        self.model_monitor = ModelMonitor(model_name=model_name)
        
        # Set up MLflow if available
        try:
            setup_mlflow()
            self.mlflow_enabled = True
        except Exception as e:
            logger.warning(f"MLflow setup failed: {e}")
            self.mlflow_enabled = False
        
        # Create necessary directories
        os.makedirs(MODELS_DIR, exist_ok=True)
        os.makedirs(REPORTS_DIR, exist_ok=True)
        
        # Initialize MLflow if available
        if self.mlflow_enabled:
            self.experiment_id = setup_mlflow()
            logger.info(f"MLflow experiment ID: {self.experiment_id}")
        else:
            logger.info("MLflow is disabled or not available")
    
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
        Load and prepare data with enhanced features.
        
        Returns
        -------
        Tuple[pd.DataFrame, pd.DataFrame]
            Training data and validation data with prepared features
        """
        # Load processed data
        train_path = self.processed_data_dir / "train_features.csv"
        val_path = self.processed_data_dir / "val_features.csv"
        
        if not train_path.exists() or not val_path.exists():
            logger.error("Processed data files not found. Run preprocess.py first.")
            return None, None
        
        logger.info("Loading processed data files")
        train_data = pd.read_csv(train_path)
        val_data = pd.read_csv(val_path)
        
        # Convert date columns
        date_cols = ['date']
        for df in [train_data, val_data]:
            for col in date_cols:
                if col in df.columns:
                    df[col] = pd.to_datetime(df[col])
        
        # Generate base features using Feature Store
        logger.info("Generating base features")
        all_features = []
        for group_name in self.feature_store.list_feature_groups():
            group = self.feature_store.get_feature_group(group_name)
            all_features.extend(group.get("features", []))
        
        train_features = self.feature_store.generate_features(
            data=train_data,
            feature_list=all_features,
            mode="training"
        )
        
        val_features = self.feature_store.generate_features(
            data=val_data,
            feature_list=all_features,
            mode="inference"
        )
        
        # Add enhanced features if enabled
        if self.feature_config.temporal_decomposition_enabled:
            logger.info("Adding temporal decomposition features")
            train_features = self.enhanced_features.add_temporal_decomposition(
                train_features,
                group_cols=self.feature_config.config["temporal_decomposition"].get("group_by")
            )
            val_features = self.enhanced_features.add_temporal_decomposition(
                val_features,
                group_cols=self.feature_config.config["temporal_decomposition"].get("group_by")
            )
        
        if self.feature_config.promotion_features_enabled:
            logger.info("Adding enhanced promotion features")
            train_features = self.enhanced_features.add_promotion_features(train_features)
            val_features = self.enhanced_features.add_promotion_features(val_features)
        
        if self.feature_config.weather_features_enabled:
            logger.info("Adding weather features")
            # This would require store location data
            store_locations = self._load_store_locations()
            if store_locations:
                train_features = self.enhanced_features.add_weather_features(
                    train_features,
                    store_locations
                )
                val_features = self.enhanced_features.add_weather_features(
                    val_features,
                    store_locations
                )
        
        # Perform feature selection if enabled
        if self.feature_config.feature_selection_enabled:
            logger.info("Performing feature selection")
            selected_features = self._select_features(
                train_features,
                train_data['sales'],  # Target variable
                **self.feature_config.get_feature_selection_params()
            )
            
            # Ensure we only use features that exist in both datasets
            common_features = list(set(selected_features) & set(train_features.columns) & set(val_features.columns))
            logger.info(f"Using {len(common_features)} common features after selection")
            
            train_features = train_features[common_features]
            val_features = val_features[common_features]
        
        logger.info(f"Final training data shape: {train_features.shape}")
        logger.info(f"Final validation data shape: {val_features.shape}")
        
        return train_features, val_features
    
    def _select_features(
        self,
        X: pd.DataFrame,
        y: pd.Series,
        method: str = "importance",
        threshold: float = 0.01,
        max_features: int = 100
    ) -> List[str]:
        """
        Select features based on importance or correlation.
        
        Parameters
        ----------
        X : pd.DataFrame
            Feature matrix
        y : pd.Series
            Target variable
        method : str
            Feature selection method ('importance', 'correlation', or 'variance')
        threshold : float
            Minimum importance/correlation to keep feature
        max_features : int
            Maximum number of features to keep
            
        Returns
        -------
        List[str]
            List of selected feature names
        """
        # Create a copy of the data for feature selection
        X_select = X.copy()
        
        # Pre-process datetime columns
        datetime_cols = X_select.select_dtypes(include=['datetime64']).columns
        for col in datetime_cols:
            X_select[f"{col}_year"] = X_select[col].dt.year
            X_select[f"{col}_month"] = X_select[col].dt.month
            X_select[f"{col}_day"] = X_select[col].dt.day
            X_select[f"{col}_dayofweek"] = X_select[col].dt.dayofweek
            X_select = X_select.drop(col, axis=1)
        
        # Pre-process categorical columns
        categorical_cols = X_select.select_dtypes(include=['object']).columns
        for col in categorical_cols:
            # Convert to categorical codes
            X_select[col] = pd.Categorical(X_select[col]).codes
        
        if method == "importance":
            # Use LightGBM for feature importance
            train_data = lgb.Dataset(X_select, label=y)
            params = {
                'objective': 'regression',
                'metric': 'rmse',
                'verbosity': -1,
                'num_leaves': 31,
                'max_depth': -1
            }
            
            model = lgb.train(params, train_data, num_boost_round=100)
            importance = pd.Series(
                model.feature_importance(),
                index=X_select.columns
            )
            
            # Select features above threshold
            selected = importance[importance >= threshold].index.tolist()
            
        elif method == "correlation":
            # Use absolute correlation with target
            correlations = X_select.apply(lambda x: abs(x.corr(y)))
            selected = correlations[correlations >= threshold].index.tolist()
            
        elif method == "variance":
            # Use variance threshold
            from sklearn.feature_selection import VarianceThreshold
            
            selector = VarianceThreshold(threshold=threshold)
            selector.fit(X_select)
            selected = X_select.columns[selector.get_support()].tolist()
        
        else:
            raise ValueError(f"Unknown feature selection method: {method}")
        
        # Limit number of features if needed
        if len(selected) > max_features:
            if method == "importance":
                selected = importance.nlargest(max_features).index.tolist()
            elif method == "correlation":
                selected = correlations.nlargest(max_features).index.tolist()
            else:
                selected = selected[:max_features]
        
        # Map back to original feature names for datetime columns
        final_selected = []
        for feature in selected:
            parts = feature.split('_')
            if len(parts) > 1 and parts[0] in [col.split('_')[0] for col in datetime_cols]:
                # This is a derived datetime feature, keep the original datetime column
                original_col = parts[0]
                if original_col not in final_selected:
                    final_selected.append(original_col)
            else:
                final_selected.append(feature)
        
        logger.info(f"Selected {len(final_selected)} features using {method} method")
        return final_selected
    
    def _load_store_locations(self) -> Optional[Dict[int, Dict[str, float]]]:
        """Load store locations from configuration."""
        try:
            store_locations_path = self.data_dir / "store_locations.json"
            if store_locations_path.exists():
                with open(store_locations_path, "r") as f:
                    return json.load(f)
            return None
        except Exception as e:
            logger.error(f"Error loading store locations: {e}")
            return None
    
    def train_model(self, X_train: pd.DataFrame, y_train: pd.Series, 
                   X_val: Optional[pd.DataFrame] = None, 
                   y_val: Optional[pd.Series] = None) -> Tuple[Any, Dict[str, float]]:
        """
        Train a machine learning model.
        
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
        model : Any
            Trained model
        metrics : Dict[str, float]
            Model performance metrics
        """
        logger.info(f"Training model with {X_train.shape[0]} samples and {X_train.shape[1]} features")
        
        # Check for validation set
        use_validation = X_val is not None and y_val is not None
        
        # Model hyperparameters
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
        
        # Log model parameters to MLflow
        if self.mlflow_enabled:
            try:
                with mlflow.start_run(experiment_id=self.experiment_id) as run:
                    run_id = run.info.run_id
                    logger.info(f"MLflow run ID: {run_id}")
                    
                    # Log parameters
                    log_model_params(params)
                    mlflow.log_param("n_samples", X_train.shape[0])
                    mlflow.log_param("n_features", X_train.shape[1])
            except Exception as e:
                logger.error(f"Error starting MLflow run: {e}")
        
        # Create dataset for LightGBM
        train_data = lgb.Dataset(X_train, label=y_train)
        
        if use_validation:
            val_data = lgb.Dataset(X_val, label=y_val, reference=train_data)
            valid_sets = [train_data, val_data]
            valid_names = ['train', 'valid']
        else:
            valid_sets = [train_data]
            valid_names = ['train']
        
        # Train the model
        callbacks = [lgb.early_stopping(50, verbose=False)]
        model = lgb.train(
            params,
            train_data,
            num_boost_round=1000,
            valid_sets=valid_sets,
            valid_names=valid_names,
            callbacks=callbacks,
        )
        
        # Get metrics
        metrics = {}
        
        # Predictions on training data
        train_preds = model.predict(X_train)
        metrics['train_rmse'] = float(np.sqrt(mean_squared_error(y_train, train_preds)))
        metrics['train_mae'] = float(mean_absolute_error(y_train, train_preds))
        
        if use_validation:
            # Predictions on validation data
            val_preds = model.predict(X_val)
            metrics['val_rmse'] = float(np.sqrt(mean_squared_error(y_val, val_preds)))
            metrics['val_mae'] = float(mean_absolute_error(y_val, val_preds))
        
        # Log metrics to MLflow
        if self.mlflow_enabled:
            try:
                log_model_metrics(metrics)
                
                # Log feature importance
                importance = model.feature_importance(importance_type='gain')
                feature_importance = pd.DataFrame({
                    'Feature': X_train.columns,
                    'Importance': importance
                }).sort_values(by='Importance', ascending=False)
                
                # Create feature importance plot
                plt_path = Path(REPORTS_DIR) / f"feature_importance_{datetime.now().strftime('%Y%m%d_%H%M%S')}.png"
                feature_importance.plot(x='Feature', y='Importance', kind='bar', figsize=(12, 6))
                plt.tight_layout()
                plt.savefig(plt_path)
                plt.close()
                
                # Log feature importance plot
                mlflow.log_artifact(str(plt_path))
                
                # Log feature importance as a JSON file
                importance_path = Path(REPORTS_DIR) / f"feature_importance_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
                feature_importance.to_json(importance_path, orient='records')
                mlflow.log_artifact(str(importance_path))
                
                # Log model to MLflow
                model_path = str(MODELS_DIR / f"{self.model_name}.pkl")
                joblib.dump(model, model_path)
                log_model(model, model_path, registered_model_name=self.model_name)
                
            except Exception as e:
                logger.error(f"Error logging to MLflow: {e}")
        
        # Save metrics to JSON
        metrics_path = REPORTS_DIR / "metrics.json"
        try:
            with open(metrics_path, 'w') as f:
                json.dump(metrics, f, indent=4)
            logger.info(f"Metrics saved to {metrics_path}")
        except Exception as e:
            logger.error(f"Error saving metrics: {e}")
        
        logger.info(f"Model training completed with metrics: {metrics}")
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