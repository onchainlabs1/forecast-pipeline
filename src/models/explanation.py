"""
Module for model explainability using SHAP.
"""

import logging
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import shap
from typing import Dict, List, Any, Optional, Union
from pathlib import Path
import joblib

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)

# Define paths
PROJECT_DIR = Path(__file__).resolve().parents[2]
MODELS_DIR = PROJECT_DIR / "models"
REPORTS_DIR = PROJECT_DIR / "reports"
SHAP_PLOTS_DIR = REPORTS_DIR / "shap_plots"


class ModelExplainer:
    """
    Class to generate model explanations using SHAP.
    
    This class encapsulates the functionality to generate SHAP explanations
    for machine learning models, especially for LightGBM.
    """
    
    def __init__(self, model=None, model_path=None, feature_names=None):
        """
        Initializes the model explainer.
        
        Parameters
        ----------
        model : object, optional
            Trained model to explain.
        model_path : str or Path, optional
            Path to the model file.
        feature_names : list, optional
            Names of the features used by the model.
        """
        self.model = model
        self.feature_names = feature_names
        self.explainer = None
        
        # Create directory for plots
        SHAP_PLOTS_DIR.mkdir(parents=True, exist_ok=True)
        
        # Load model if a path is provided
        if model_path is not None:
            self.load_model(model_path)
    
    def load_model(self, model_path):
        """
        Loads a model from a file.
        
        Parameters
        ----------
        model_path : str or Path
            Path to the model file.
        """
        try:
            logger.info(f"Loading model from {model_path}")
            self.model = joblib.load(model_path)
            logger.info("Model loaded successfully")
        except Exception as e:
            logger.error(f"Error loading model: {e}")
            raise e
    
    def create_explainer(self, data=None):
        """
        Creates a SHAP explainer for the model.
        
        Parameters
        ----------
        data : pandas.DataFrame, optional
            Background data for the explainer. If None, the model summary will be used.
        
        Returns
        -------
        explainer : shap.Explainer
            SHAP explainer for the model.
        """
        try:
            logger.info("Creating SHAP explainer")
            
            # For tree-based models (like LightGBM)
            if hasattr(self.model, 'predict') and hasattr(self.model, 'feature_importances_'):
                if data is not None:
                    self.explainer = shap.TreeExplainer(self.model, data)
                else:
                    self.explainer = shap.TreeExplainer(self.model)
            else:
                # For other model types
                if data is not None:
                    self.explainer = shap.Explainer(self.model.predict, data)
                else:
                    logger.warning("Background data needed for non-tree-based model")
                    raise ValueError("Background data needed for non-tree-based model")
            
            logger.info("SHAP explainer created successfully")
            return self.explainer
        
        except Exception as e:
            logger.error(f"Error creating SHAP explainer: {e}")
            raise e
    
    def compute_shap_values(self, X):
        """
        Calculates the SHAP values for a dataset.
        
        Parameters
        ----------
        X : pandas.DataFrame
            Data for which to calculate SHAP values.
        
        Returns
        -------
        shap_values : numpy.ndarray
            SHAP values for the provided data.
        """
        if self.explainer is None:
            self.create_explainer()
        
        logger.info("Calculating SHAP values")
        shap_values = self.explainer(X)
        logger.info("SHAP values calculated successfully")
        
        return shap_values
    
    def plot_summary(self, shap_values, X, output_file=None):
        """
        Generates a SHAP summary plot.
        
        Parameters
        ----------
        shap_values : numpy.ndarray
            Calculated SHAP values.
        X : pandas.DataFrame
            Data for which the SHAP values were calculated.
        output_file : str or Path, optional
            Path to save the plot. If None, the plot will be displayed.
        
        Returns
        -------
        fig : matplotlib.figure.Figure
            Figure of the generated plot.
        """
        logger.info("Generating SHAP summary plot")
        
        plt.figure(figsize=(10, 8))
        shap.summary_plot(shap_values, X, show=False)
        
        if output_file:
            plt.savefig(output_file, bbox_inches='tight', dpi=300)
            logger.info(f"SHAP summary plot saved to {output_file}")
        
        fig = plt.gcf()
        plt.close()
        
        return fig
    
    def plot_feature_importance(self, shap_values, X, output_file=None):
        """
        Generates a feature importance plot based on SHAP.
        
        Parameters
        ----------
        shap_values : numpy.ndarray
            Calculated SHAP values.
        X : pandas.DataFrame
            Data for which the SHAP values were calculated.
        output_file : str or Path, optional
            Path to save the plot. If None, the plot will be displayed.
        
        Returns
        -------
        fig : matplotlib.figure.Figure
            Figure of the generated plot.
        """
        logger.info("Generating feature importance plot")
        
        plt.figure(figsize=(10, 8))
        shap.plots.bar(shap_values, show=False)
        
        if output_file:
            plt.savefig(output_file, bbox_inches='tight', dpi=300)
            logger.info(f"Feature importance plot saved to {output_file}")
        
        fig = plt.gcf()
        plt.close()
        
        return fig
    
    def explain_prediction(self, instance, feature_names=None):
        """
        Explains a specific prediction.
        
        Parameters
        ----------
        instance : pandas.Series or numpy.ndarray
            Instance for which to explain the prediction.
        feature_names : list, optional
            Feature names. If None, the default names will be used.
        
        Returns
        -------
        explanation : dict
            Dictionary with the prediction explanations.
        """
        if self.explainer is None:
            logger.warning("SHAP explainer not initialized")
            raise ValueError("SHAP explainer not initialized. Call create_explainer first.")
        
        logger.info("Explaining prediction for specific instance")
        
        # Convert to DataFrame if needed
        if isinstance(instance, pd.Series):
            instance = instance.to_frame().T
        elif isinstance(instance, np.ndarray):
            if instance.ndim == 1:
                instance = pd.DataFrame([instance], columns=feature_names or self.feature_names)
            else:
                instance = pd.DataFrame(instance, columns=feature_names or self.feature_names)
        
        # Calculate SHAP values
        shap_values = self.explainer(instance)
        
        # Generate explanation
        feature_names = instance.columns
        base_value = shap_values.base_values[0]
        shap_values_array = shap_values.values[0]
        
        # Sort by magnitude
        idx = np.argsort(np.abs(shap_values_array))[::-1]
        
        # Create dictionary with explanations
        explanation = {
            "base_value": float(base_value),
            "prediction": float(base_value + np.sum(shap_values_array)),
            "feature_contributions": [
                {
                    "feature": feature_names[i],
                    "contribution": float(shap_values_array[i]),
                    "value": float(instance.iloc[0, i])
                }
                for i in idx
            ]
        }
        
        logger.info("Prediction explanation generated successfully")
        return explanation
    
    def generate_explanation_report(self, X, output_dir=None):
        """
        Generates a complete model explanation report.
        
        Parameters
        ----------
        X : pandas.DataFrame
            Data to generate explanations.
        output_dir : str or Path, optional
            Directory to save the plots. If None, SHAP_PLOTS_DIR will be used.
        
        Returns
        -------
        report : dict
            Dictionary with the explanation report.
        """
        if output_dir is None:
            output_dir = SHAP_PLOTS_DIR
        else:
            output_dir = Path(output_dir)
            output_dir.mkdir(parents=True, exist_ok=True)
        
        logger.info(f"Generating model explanation report in {output_dir}")
        
        # Calculate SHAP values
        shap_values = self.compute_shap_values(X)
        
        # Generate plots
        summary_plot_path = output_dir / "shap_summary.png"
        importance_plot_path = output_dir / "shap_importance.png"
        
        self.plot_summary(shap_values, X, summary_plot_path)
        self.plot_feature_importance(shap_values, X, importance_plot_path)
        
        # Calculate average importance
        if isinstance(shap_values, shap._explanation.Explanation):
            shap_values_array = shap_values.values
            mean_abs_shap = np.mean(np.abs(shap_values_array), axis=0)
        else:
            mean_abs_shap = np.mean(np.abs(shap_values), axis=0)
        
        # Sort features by importance
        if X.columns is not None:
            feature_importance = sorted(
                zip(X.columns, mean_abs_shap),
                key=lambda x: x[1],
                reverse=True
            )
            feature_importance = [{"feature": feature, "importance": float(imp)} for feature, imp in feature_importance]
        else:
            feature_importance = [{"feature": f"feature_{i}", "importance": float(imp)} for i, imp in enumerate(mean_abs_shap)]
        
        # Create report
        report = {
            "model_type": type(self.model).__name__,
            "feature_importance": feature_importance,
            "plots": {
                "summary_plot": str(summary_plot_path),
                "importance_plot": str(importance_plot_path)
            }
        }
        
        # Save report
        report_path = output_dir / "explanation_report.json"
        pd.Series(report).to_json(report_path, orient="index")
        
        logger.info(f"Model explanation report saved to {report_path}")
        return report 