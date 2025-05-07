#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Model Monitoring System.
Monitors model performance and data drift to alert when retraining is needed.
"""

import os
import json
import logging
import pickle
from pathlib import Path
from datetime import datetime, timedelta
from typing import Dict, List, Any, Optional, Tuple, Union

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import ks_2samp
from sklearn.metrics import mean_squared_error, mean_absolute_error
import mlflow

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)

# Define paths
PROJECT_DIR = Path(__file__).resolve().parents[2]
MONITORING_DIR = PROJECT_DIR / "monitoring"


class ModelMonitor:
    """
    Model monitoring system to detect data drift and performance degradation.
    
    This class provides functionality to:
    - Compare new data distributions to baseline data
    - Detect concept drift in model performance
    - Generate alerts when drift thresholds are exceeded
    - Visualize drift metrics over time
    """
    
    def __init__(self, model_name: str):
        """
        Initialize the model monitor.
        
        Parameters
        ----------
        model_name : str
            Name of the model to monitor
        """
        self.model_name = model_name
        self.baseline_data = None
        self.baseline_stats = None
        self.baseline_performance = None
        self.drift_thresholds = {
            "ks_statistic": 0.1,  # Kolmogorov-Smirnov threshold
            "performance_degradation": 0.1  # 10% degradation in metrics
        }
        
        # Create monitoring directory if it doesn't exist
        os.makedirs(MONITORING_DIR, exist_ok=True)
        model_dir = MONITORING_DIR / model_name
        os.makedirs(model_dir, exist_ok=True)
        
        # Create subdirectories
        os.makedirs(model_dir / "data_drift", exist_ok=True)
        os.makedirs(model_dir / "performance", exist_ok=True)
        os.makedirs(model_dir / "alerts", exist_ok=True)
        os.makedirs(model_dir / "visualizations", exist_ok=True)
        
        # Try to load existing baseline data
        self._load_baseline()
    
    def _load_baseline(self):
        """Load baseline data and stats if they exist."""
        baseline_path = MONITORING_DIR / self.model_name / "baseline_stats.json"
        
        if baseline_path.exists():
            with open(baseline_path, "r") as f:
                self.baseline_stats = json.load(f)
            logger.info(f"Loaded baseline statistics for {self.model_name}")
            
            # Load baseline performance
            performance_path = MONITORING_DIR / self.model_name / "baseline_performance.json"
            if performance_path.exists():
                with open(performance_path, "r") as f:
                    self.baseline_performance = json.load(f)
                logger.info(f"Loaded baseline performance for {self.model_name}")
    
    def set_baseline(self, 
                    data: pd.DataFrame, 
                    performance_metrics: Dict[str, float],
                    categorical_columns: Optional[List[str]] = None,
                    numerical_columns: Optional[List[str]] = None):
        """
        Set baseline data statistics for drift comparison.
        
        Parameters
        ----------
        data : pd.DataFrame
            Baseline training or validation data
        performance_metrics : Dict[str, float]
            Baseline model performance metrics
        categorical_columns : List[str], optional
            List of categorical column names
        numerical_columns : List[str], optional
            List of numerical column names
        """
        if numerical_columns is None:
            numerical_columns = data.select_dtypes(include=np.number).columns.tolist()
        
        if categorical_columns is None:
            categorical_columns = data.select_dtypes(include=["object", "category"]).columns.tolist()
        
        # Compute baseline statistics
        num_stats = {}
        for col in numerical_columns:
            if col in data.columns:
                num_stats[col] = {
                    "mean": float(data[col].mean()),
                    "median": float(data[col].median()),
                    "std": float(data[col].std()),
                    "min": float(data[col].min()),
                    "max": float(data[col].max()),
                    "p25": float(data[col].quantile(0.25)),
                    "p75": float(data[col].quantile(0.75))
                }
        
        cat_stats = {}
        for col in categorical_columns:
            if col in data.columns:
                value_counts = data[col].value_counts(normalize=True).to_dict()
                cat_stats[col] = {str(k): float(v) for k, v in value_counts.items()}
        
        # Save baseline statistics
        self.baseline_stats = {
            "numerical": num_stats,
            "categorical": cat_stats,
            "timestamp": datetime.now().isoformat(),
            "n_rows": len(data)
        }
        
        baseline_path = MONITORING_DIR / self.model_name / "baseline_stats.json"
        with open(baseline_path, "w") as f:
            json.dump(self.baseline_stats, f, indent=2)
        
        # Save baseline performance
        self.baseline_performance = performance_metrics
        performance_path = MONITORING_DIR / self.model_name / "baseline_performance.json"
        with open(performance_path, "w") as f:
            json.dump(self.baseline_performance, f, indent=2)
        
        # Store a sample of baseline data
        sample_size = min(1000, len(data))
        self.baseline_data = data.sample(sample_size)
        sample_path = MONITORING_DIR / self.model_name / "baseline_sample.pkl"
        with open(sample_path, "wb") as f:
            pickle.dump(self.baseline_data, f)
        
        logger.info(f"Baseline set for {self.model_name} with {len(data)} rows")
    
    def check_data_drift(self, 
                        new_data: pd.DataFrame,
                        log_to_mlflow: bool = True) -> Dict[str, Any]:
        """
        Check for drift between baseline and new data.
        
        Parameters
        ----------
        new_data : pd.DataFrame
            New data to compare against baseline
        log_to_mlflow : bool
            Whether to log drift metrics to MLflow
            
        Returns
        -------
        Dict[str, Any]
            Dictionary with drift metrics
        """
        if self.baseline_stats is None:
            logger.warning("Baseline stats not set. Call set_baseline first.")
            return {}
        
        # Get numerical and categorical columns from baseline
        num_cols = list(self.baseline_stats["numerical"].keys())
        cat_cols = list(self.baseline_stats["categorical"].keys())
        
        drift_metrics = {
            "timestamp": datetime.now().isoformat(),
            "n_rows": len(new_data),
            "numerical_drift": {},
            "categorical_drift": {},
            "overall_drift_score": 0.0,
            "drift_detected": False
        }
        
        # Check numerical columns for drift using KS test
        ks_results = []
        for col in num_cols:
            if col in new_data.columns:
                # Get baseline stats
                baseline_mean = self.baseline_stats["numerical"][col]["mean"]
                baseline_std = self.baseline_stats["numerical"][col]["std"]
                
                # Get new stats
                new_mean = float(new_data[col].mean())
                new_std = float(new_data[col].std())
                
                # Perform KS test
                if self.baseline_data is not None and col in self.baseline_data:
                    ks_stat, p_value = ks_2samp(
                        self.baseline_data[col].dropna(), 
                        new_data[col].dropna()
                    )
                else:
                    # If baseline data not available, use simplified test
                    ks_stat = abs(baseline_mean - new_mean) / max(baseline_std, 1e-6)
                    p_value = 0.05 if ks_stat > 2.0 else 0.5
                
                drift_metrics["numerical_drift"][col] = {
                    "ks_statistic": float(ks_stat),
                    "p_value": float(p_value),
                    "mean_diff": float(abs(baseline_mean - new_mean)),
                    "std_diff": float(abs(baseline_std - new_std)),
                    "drift_detected": ks_stat > self.drift_thresholds["ks_statistic"]
                }
                
                ks_results.append(ks_stat)
        
        # Check categorical columns for drift using distribution difference
        for col in cat_cols:
            if col in new_data.columns:
                baseline_dist = self.baseline_stats["categorical"][col]
                
                # Calculate new distribution
                new_dist = new_data[col].value_counts(normalize=True).to_dict()
                
                # Calculate JS divergence (simplified)
                js_div = 0.0
                for category, baseline_prob in baseline_dist.items():
                    baseline_p = float(baseline_prob)
                    new_p = float(new_dist.get(category, 0.0))
                    js_div += abs(baseline_p - new_p)
                
                js_div = min(1.0, js_div / 2.0)  # Normalize
                
                drift_metrics["categorical_drift"][col] = {
                    "js_divergence": float(js_div),
                    "drift_detected": js_div > self.drift_thresholds["ks_statistic"]
                }
        
        # Calculate overall drift score
        num_drifts = sum(1 for col_metrics in drift_metrics["numerical_drift"].values()
                        if col_metrics["drift_detected"])
        cat_drifts = sum(1 for col_metrics in drift_metrics["categorical_drift"].values()
                        if col_metrics["drift_detected"])
        
        total_cols = len(num_cols) + len(cat_cols)
        if total_cols > 0:
            drift_score = (num_drifts + cat_drifts) / total_cols
        else:
            drift_score = 0.0
        
        drift_metrics["overall_drift_score"] = float(drift_score)
        drift_metrics["drift_detected"] = drift_score > 0.3  # If > 30% of features have drift
        
        # Save drift metrics
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        metrics_path = MONITORING_DIR / self.model_name / "data_drift" / f"drift_{timestamp}.json"
        with open(metrics_path, "w") as f:
            json.dump(drift_metrics, f, indent=2)
        
        # Log to MLflow if requested
        if log_to_mlflow:
            try:
                with mlflow.start_run(run_name=f"drift_monitoring_{timestamp}"):
                    mlflow.log_metric("overall_drift_score", drift_metrics["overall_drift_score"])
                    mlflow.log_metric("drift_detected", int(drift_metrics["drift_detected"]))
                    
                    for col, metrics in drift_metrics["numerical_drift"].items():
                        mlflow.log_metric(f"drift_{col}_ks", metrics["ks_statistic"])
            except Exception as e:
                logger.warning(f"Failed to log drift metrics to MLflow: {e}")
        
        # Generate alert if drift detected
        if drift_metrics["drift_detected"]:
            self._generate_alert("data_drift", drift_metrics)
            self._visualize_drift(new_data, timestamp)
        
        return drift_metrics
    
    def check_performance_drift(self, 
                              current_metrics: Dict[str, float],
                              log_to_mlflow: bool = True) -> Dict[str, Any]:
        """
        Check for drift in model performance.
        
        Parameters
        ----------
        current_metrics : Dict[str, float]
            Current model performance metrics
        log_to_mlflow : bool
            Whether to log performance drift to MLflow
            
        Returns
        -------
        Dict[str, Any]
            Dictionary with performance drift metrics
        """
        if self.baseline_performance is None:
            logger.warning("Baseline performance not set. Call set_baseline first.")
            return {}
        
        performance_drift = {
            "timestamp": datetime.now().isoformat(),
            "metrics_diff": {},
            "drift_detected": False
        }
        
        # Calculate differences for each metric
        for metric, baseline_value in self.baseline_performance.items():
            if metric in current_metrics:
                current_value = current_metrics[metric]
                
                # Calculate relative difference
                if baseline_value != 0:
                    rel_diff = abs(current_value - baseline_value) / abs(baseline_value)
                else:
                    rel_diff = abs(current_value - baseline_value)
                
                performance_drift["metrics_diff"][metric] = {
                    "baseline": float(baseline_value),
                    "current": float(current_value),
                    "abs_diff": float(abs(current_value - baseline_value)),
                    "rel_diff": float(rel_diff),
                    "degraded": bool(rel_diff > self.drift_thresholds["performance_degradation"] and
                                   current_value > baseline_value)  # For error metrics, higher is worse
                }
        
        # Determine if performance has degraded overall
        degraded_metrics = sum(1 for metric in performance_drift["metrics_diff"].values()
                              if metric["degraded"])
        performance_drift["drift_detected"] = degraded_metrics > 0
        
        # Save performance drift metrics
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        metrics_path = MONITORING_DIR / self.model_name / "performance" / f"perf_{timestamp}.json"
        with open(metrics_path, "w") as f:
            json.dump(performance_drift, f, indent=2)
        
        # Log to MLflow if requested
        if log_to_mlflow:
            try:
                with mlflow.start_run(run_name=f"performance_monitoring_{timestamp}"):
                    mlflow.log_metric("performance_drift_detected", int(performance_drift["drift_detected"]))
                    
                    for metric, values in performance_drift["metrics_diff"].items():
                        mlflow.log_metric(f"{metric}_rel_diff", values["rel_diff"])
                        mlflow.log_metric(f"{metric}_current", values["current"])
            except Exception as e:
                logger.warning(f"Failed to log performance metrics to MLflow: {e}")
        
        # Generate alert if drift detected
        if performance_drift["drift_detected"]:
            self._generate_alert("performance_drift", performance_drift)
        
        return performance_drift
    
    def _generate_alert(self, alert_type: str, alert_data: Dict[str, Any]):
        """
        Generate an alert for drift detection.
        
        Parameters
        ----------
        alert_type : str
            Type of alert ('data_drift' or 'performance_drift')
        alert_data : Dict[str, Any]
            Alert data
        """
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        alert = {
            "timestamp": datetime.now().isoformat(),
            "type": alert_type,
            "model": self.model_name,
            "data": alert_data,
            "message": f"{alert_type.replace('_', ' ').title()} detected for model {self.model_name}"
        }
        
        # Save alert
        alert_path = MONITORING_DIR / self.model_name / "alerts" / f"alert_{timestamp}.json"
        with open(alert_path, "w") as f:
            json.dump(alert, f, indent=2)
        
        logger.warning(f"ALERT: {alert['message']}")
    
    def _visualize_drift(self, new_data: pd.DataFrame, timestamp: str):
        """
        Create visualizations for data drift.
        
        Parameters
        ----------
        new_data : pd.DataFrame
            New data for visualization
        timestamp : str
            Timestamp for file naming
        """
        if self.baseline_data is None:
            logger.warning("Baseline data not available for visualization")
            return
        
        try:
            # Select numerical columns that exist in both datasets
            num_cols = list(set(self.baseline_stats["numerical"].keys()) & 
                           set(new_data.columns))
            
            for col in num_cols[:5]:  # Limit to 5 cols to avoid too many plots
                plt.figure(figsize=(10, 6))
                sns.kdeplot(self.baseline_data[col].dropna(), label="Baseline")
                sns.kdeplot(new_data[col].dropna(), label="Current")
                plt.title(f"Distribution Drift: {col}")
                plt.legend()
                plt.tight_layout()
                
                # Save plot
                plot_path = MONITORING_DIR / self.model_name / "visualizations" / f"drift_{col}_{timestamp}.png"
                plt.savefig(plot_path)
                plt.close()
            
            # Create a summary plot of all drifts
            drift_values = []
            cols = []
            
            # Get drift values from latest metrics
            latest_drift_file = sorted(list((MONITORING_DIR / self.model_name / "data_drift").glob("drift_*.json")))[-1]
            with open(latest_drift_file, "r") as f:
                drift_data = json.load(f)
            
            for col, metrics in drift_data["numerical_drift"].items():
                drift_values.append(metrics["ks_statistic"])
                cols.append(col)
            
            for col, metrics in drift_data["categorical_drift"].items():
                drift_values.append(metrics["js_divergence"])
                cols.append(col)
            
            # Create bar plot
            plt.figure(figsize=(12, 8))
            bars = plt.bar(cols, drift_values)
            plt.axhline(y=self.drift_thresholds["ks_statistic"], color='r', linestyle='-', label="Drift Threshold")
            plt.title("Feature Drift Summary")
            plt.xlabel("Feature")
            plt.ylabel("Drift Metric")
            plt.xticks(rotation=45, ha="right")
            plt.tight_layout()
            
            # Color bars exceeding threshold
            for i, bar in enumerate(bars):
                if drift_values[i] > self.drift_thresholds["ks_statistic"]:
                    bar.set_color("red")
            
            # Save plot
            plot_path = MONITORING_DIR / self.model_name / "visualizations" / f"drift_summary_{timestamp}.png"
            plt.savefig(plot_path)
            plt.close()
        
        except Exception as e:
            logger.error(f"Error creating drift visualizations: {e}")
    
    def get_alerts(self, days: int = 7) -> List[Dict[str, Any]]:
        """
        Get recent alerts.
        
        Parameters
        ----------
        days : int
            Number of days to look back
            
        Returns
        -------
        List[Dict[str, Any]]
            List of recent alerts
        """
        alerts_dir = MONITORING_DIR / self.model_name / "alerts"
        if not alerts_dir.exists():
            return []
        
        cutoff_date = datetime.now() - timedelta(days=days)
        alerts = []
        
        for alert_file in alerts_dir.glob("alert_*.json"):
            with open(alert_file, "r") as f:
                alert = json.load(f)
            
            alert_date = datetime.fromisoformat(alert["timestamp"])
            if alert_date >= cutoff_date:
                alerts.append(alert)
        
        return sorted(alerts, key=lambda x: x["timestamp"], reverse=True)
    
    def needs_retraining(self) -> Tuple[bool, str]:
        """
        Check if model needs retraining based on recent alerts.
        
        Returns
        -------
        Tuple[bool, str]
            Whether retraining is needed and reason
        """
        recent_alerts = self.get_alerts(days=7)
        
        if not recent_alerts:
            return False, "No alerts detected recently"
        
        # Check for severe data drift
        data_drift_alerts = [a for a in recent_alerts if a["type"] == "data_drift"]
        if data_drift_alerts:
            latest_drift = data_drift_alerts[0]
            if latest_drift["data"]["overall_drift_score"] > 0.5:
                return True, f"Severe data drift detected: {latest_drift['data']['overall_drift_score']:.2f}"
        
        # Check for performance degradation
        perf_alerts = [a for a in recent_alerts if a["type"] == "performance_drift"]
        if perf_alerts:
            return True, "Performance degradation detected"
        
        return False, "No significant issues detected" 