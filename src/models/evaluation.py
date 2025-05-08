#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Model evaluation utilities for the store sales forecasting project.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import mean_squared_error, mean_absolute_error, mean_absolute_percentage_error


def calculate_metrics(y_true, y_pred):
    """
    Calculate regression metrics for model evaluation.
    
    Parameters
    ----------
    y_true : array-like
        True target values
    y_pred : array-like
        Predicted target values
        
    Returns
    -------
    dict
        Dictionary containing the calculated metrics
    """
    # Ensure y_pred has no negative values (sales can't be negative)
    y_pred = np.maximum(y_pred, 0)
    
    # Calculate metrics
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    mae = mean_absolute_error(y_true, y_pred)
    
    # Handle zero values for MAPE calculation
    mask = y_true != 0
    y_true_masked = y_true[mask]
    y_pred_masked = y_pred[mask]
    
    if len(y_true_masked) > 0:
        mape = mean_absolute_percentage_error(y_true_masked, y_pred_masked) * 100
    else:
        mape = np.nan
    
    return {
        "rmse": rmse,
        "mae": mae,
        "mape": mape
    }


def calculate_metrics_by_group(df, y_true_col, y_pred_col, group_cols):
    """
    Calculate metrics grouped by specified columns.
    
    Parameters
    ----------
    df : pandas.DataFrame
        DataFrame containing true and predicted values
    y_true_col : str
        Column name for true values
    y_pred_col : str
        Column name for predicted values
    group_cols : list
        List of column names to group by
        
    Returns
    -------
    pandas.DataFrame
        DataFrame with metrics for each group
    """
    result = []
    
    # Group by the specified columns
    for name, group in df.groupby(group_cols):
        # If group_cols is a single string, name will be a scalar
        if not isinstance(name, tuple):
            name = (name,)
        
        # Calculate metrics for the group
        metrics = calculate_metrics(group[y_true_col].values, group[y_pred_col].values)
        
        # Create a row for the group
        row = dict(zip(group_cols, name))
        row.update(metrics)
        row["count"] = len(group)
        
        result.append(row)
    
    # Convert to DataFrame
    result_df = pd.DataFrame(result)
    
    # Sort by RMSE (ascending)
    if "rmse" in result_df.columns:
        result_df = result_df.sort_values("rmse")
    
    return result_df


def plot_predictions_vs_actual(y_true, y_pred, title="Predictions vs Actual"):
    """
    Plot predicted values against actual values.
    
    Parameters
    ----------
    y_true : array-like
        True target values
    y_pred : array-like
        Predicted target values
    title : str, optional
        Plot title
        
    Returns
    -------
    matplotlib.figure.Figure
        The created figure
    """
    fig, ax = plt.subplots(figsize=(10, 8))
    
    ax.scatter(y_true, y_pred, alpha=0.3)
    
    # Add perfect prediction line
    min_val = min(np.min(y_true), np.min(y_pred))
    max_val = max(np.max(y_true), np.max(y_pred))
    ax.plot([min_val, max_val], [min_val, max_val], 'r--')
    
    ax.set_xlabel("Actual")
    ax.set_ylabel("Predicted")
    ax.set_title(title)
    
    # Calculate metrics for the title
    metrics = calculate_metrics(y_true, y_pred)
    metrics_text = f"RMSE: {metrics['rmse']:.4f}, MAE: {metrics['mae']:.4f}, MAPE: {metrics['mape']:.2f}%"
    ax.text(0.05, 0.95, metrics_text, transform=ax.transAxes, verticalalignment='top')
    
    plt.tight_layout()
    return fig


def plot_prediction_distribution(y_true, y_pred, title="Prediction Error Distribution"):
    """
    Plot the distribution of prediction errors.
    
    Parameters
    ----------
    y_true : array-like
        True target values
    y_pred : array-like
        Predicted target values
    title : str, optional
        Plot title
        
    Returns
    -------
    matplotlib.figure.Figure
        The created figure
    """
    errors = y_pred - y_true
    
    fig, ax = plt.subplots(figsize=(10, 6))
    
    sns.histplot(errors, kde=True, ax=ax)
    
    ax.axvline(0, color='r', linestyle='--')
    ax.set_xlabel("Prediction Error (Predicted - Actual)")
    ax.set_ylabel("Frequency")
    ax.set_title(title)
    
    # Calculate error statistics
    mean_error = np.mean(errors)
    std_error = np.std(errors)
    error_stats = f"Mean Error: {mean_error:.4f}, Std Dev: {std_error:.4f}"
    ax.text(0.05, 0.95, error_stats, transform=ax.transAxes, verticalalignment='top')
    
    plt.tight_layout()
    return fig


def plot_time_series_predictions(df, date_col, y_true_col, y_pred_col, title="Time Series Predictions"):
    """
    Plot time series of actual and predicted values.
    
    Parameters
    ----------
    df : pandas.DataFrame
        DataFrame containing the date, true values, and predicted values
    date_col : str
        Column name for dates
    y_true_col : str
        Column name for true values
    y_pred_col : str
        Column name for predicted values
    title : str, optional
        Plot title
        
    Returns
    -------
    matplotlib.figure.Figure
        The created figure
    """
    # Sort by date
    df = df.sort_values(date_col)
    
    fig, ax = plt.subplots(figsize=(12, 6))
    
    ax.plot(df[date_col], df[y_true_col], label="Actual")
    ax.plot(df[date_col], df[y_pred_col], label="Predicted")
    
    ax.set_xlabel("Date")
    ax.set_ylabel("Value")
    ax.set_title(title)
    ax.legend()
    
    plt.xticks(rotation=45)
    plt.tight_layout()
    return fig


def plot_feature_importance(model, feature_names, top_n=20, title="Feature Importance"):
    """
    Plot feature importance from a trained model.
    
    Parameters
    ----------
    model : object
        Trained model with feature_importance_ attribute (e.g., LightGBM, XGBoost)
    feature_names : list
        List of feature names
    top_n : int, optional
        Number of top features to show
    title : str, optional
        Plot title
        
    Returns
    -------
    matplotlib.figure.Figure
        The created figure
    """
    # Get feature importances
    try:
        if hasattr(model, 'feature_importances_'):
            importances = model.feature_importances_
        elif hasattr(model, 'feature_importance'):
            importances = model.feature_importance()
        else:
            raise AttributeError("Model does not have feature_importances_ or feature_importance method")
        
        # Create DataFrame with feature names and importances
        importance_df = pd.DataFrame({
            'feature': feature_names,
            'importance': importances
        })
        
        # Sort by importance and get top N
        importance_df = importance_df.sort_values('importance', ascending=False).head(top_n)
        
        # Plot
        fig, ax = plt.subplots(figsize=(10, 8))
        sns.barplot(x='importance', y='feature', data=importance_df, ax=ax)
        
        ax.set_title(title)
        ax.set_xlabel('Importance')
        ax.set_ylabel('Feature')
        
        plt.tight_layout()
        return fig
        
    except Exception as e:
        print(f"Error plotting feature importance: {e}")
        return None


def evaluate_model(model, X_test, y_test, feature_names=None):
    """
    Comprehensive model evaluation with metrics and plots.
    
    Parameters
    ----------
    model : object
        Trained model with predict method
    X_test : pandas.DataFrame
        Test features
    y_test : array-like
        Test target values
    feature_names : list, optional
        List of feature names for importance plot
        
    Returns
    -------
    tuple
        Metrics dictionary and list of figures
    """
    # Make predictions
    y_pred = model.predict(X_test)
    
    # Ensure predictions are non-negative
    y_pred = np.maximum(y_pred, 0)
    
    # Calculate metrics
    metrics = calculate_metrics(y_test, y_pred)
    
    # Create plots
    figures = []
    
    # Predictions vs Actual plot
    fig1 = plot_predictions_vs_actual(y_test, y_pred)
    figures.append(fig1)
    
    # Error distribution plot
    fig2 = plot_prediction_distribution(y_test, y_pred)
    figures.append(fig2)
    
    # Feature importance plot (if feature names are provided)
    if feature_names is not None:
        try:
            fig3 = plot_feature_importance(model, feature_names)
            if fig3 is not None:
                figures.append(fig3)
        except Exception:
            pass
    
    return metrics, figures 