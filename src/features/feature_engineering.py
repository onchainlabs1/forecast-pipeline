#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Feature engineering module for the store sales forecasting project.
Contains functions to create features from raw data.
"""

import logging
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from pathlib import Path
from typing import List, Dict, Optional

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)


def create_time_features(df: pd.DataFrame, date_col: str = "date") -> pd.DataFrame:
    """
    Create time-based features from a date column.
    
    Parameters
    ----------
    df : pd.DataFrame
        DataFrame containing the date column
    date_col : str, optional (default="date")
        Name of the date column
        
    Returns
    -------
    pd.DataFrame
        DataFrame with additional time features
    """
    logger.info("Creating time features")
    
    # Ensure the date column is datetime
    df[date_col] = pd.to_datetime(df[date_col])
    
    # Extract basic date components
    df["day_of_week"] = df[date_col].dt.dayofweek
    df["day_of_month"] = df[date_col].dt.day
    df["month"] = df[date_col].dt.month
    df["year"] = df[date_col].dt.year
    df["quarter"] = df[date_col].dt.quarter
    
    # Create binary indicators
    df["is_weekend"] = df["day_of_week"].isin([5, 6]).astype(int)
    df["is_month_start"] = df[date_col].dt.is_month_start.astype(int)
    df["is_month_end"] = df[date_col].dt.is_month_end.astype(int)
    df["is_quarter_start"] = df[date_col].dt.is_quarter_start.astype(int)
    df["is_quarter_end"] = df[date_col].dt.is_quarter_end.astype(int)
    df["is_year_start"] = df[date_col].dt.is_year_start.astype(int)
    df["is_year_end"] = df[date_col].dt.is_year_end.astype(int)
    
    # Create cyclical features for day of week, day of month, and month
    # This preserves the cyclical nature of these features
    df["day_of_week_sin"] = np.sin(2 * np.pi * df["day_of_week"] / 7)
    df["day_of_week_cos"] = np.cos(2 * np.pi * df["day_of_week"] / 7)
    df["day_of_month_sin"] = np.sin(2 * np.pi * df["day_of_month"] / 31)
    df["day_of_month_cos"] = np.cos(2 * np.pi * df["day_of_month"] / 31)
    df["month_sin"] = np.sin(2 * np.pi * df["month"] / 12)
    df["month_cos"] = np.cos(2 * np.pi * df["month"] / 12)
    
    return df


def create_lag_features(
    df: pd.DataFrame, 
    target_col: str, 
    groupby_cols: List[str], 
    lag_periods: List[int]
) -> pd.DataFrame:
    """
    Create lag features for a target column.
    
    Parameters
    ----------
    df : pd.DataFrame
        DataFrame containing the target column
    target_col : str
        Name of the target column
    groupby_cols : List[str]
        Columns to group by before shifting
    lag_periods : List[int]
        List of periods to shift by
        
    Returns
    -------
    pd.DataFrame
        DataFrame with additional lag features
    """
    logger.info(f"Creating lag features for {target_col}")
    
    # Sort by date and groupby columns to ensure correct shifting
    sort_cols = ["date"] + groupby_cols
    df = df.sort_values(sort_cols).copy()
    
    # Create lag features
    for lag in lag_periods:
        col_name = f"{target_col}_lag_{lag}"
        df[col_name] = df.groupby(groupby_cols)[target_col].shift(lag)
    
    return df


def create_rolling_features(
    df: pd.DataFrame, 
    target_col: str, 
    groupby_cols: List[str], 
    windows: List[int],
    min_periods: int = 1
) -> pd.DataFrame:
    """
    Create rolling window features for a target column.
    
    Parameters
    ----------
    df : pd.DataFrame
        DataFrame containing the target column
    target_col : str
        Name of the target column
    groupby_cols : List[str]
        Columns to group by before applying rolling windows
    windows : List[int]
        List of window sizes
    min_periods : int, optional (default=1)
        Minimum number of observations required for calculating statistics
        
    Returns
    -------
    pd.DataFrame
        DataFrame with additional rolling features
    """
    logger.info(f"Creating rolling features for {target_col}")
    
    # Sort by date and groupby columns
    sort_cols = ["date"] + groupby_cols
    df = df.sort_values(sort_cols).copy()
    
    # Create rolling features for each window size
    for window in windows:
        # Rolling mean
        df[f"{target_col}_roll_{window}_mean"] = df.groupby(groupby_cols)[target_col].transform(
            lambda x: x.rolling(window, min_periods=min_periods).mean()
        )
        
        # Rolling standard deviation
        df[f"{target_col}_roll_{window}_std"] = df.groupby(groupby_cols)[target_col].transform(
            lambda x: x.rolling(window, min_periods=min_periods).std()
        )
        
        # Rolling max
        df[f"{target_col}_roll_{window}_max"] = df.groupby(groupby_cols)[target_col].transform(
            lambda x: x.rolling(window, min_periods=min_periods).max()
        )
        
        # Rolling min
        df[f"{target_col}_roll_{window}_min"] = df.groupby(groupby_cols)[target_col].transform(
            lambda x: x.rolling(window, min_periods=min_periods).min()
        )
    
    return df


def create_expanding_features(
    df: pd.DataFrame, 
    target_col: str, 
    groupby_cols: List[str],
    min_periods: int = 1
) -> pd.DataFrame:
    """
    Create expanding window features for a target column.
    
    Parameters
    ----------
    df : pd.DataFrame
        DataFrame containing the target column
    target_col : str
        Name of the target column
    groupby_cols : List[str]
        Columns to group by before applying expanding windows
    min_periods : int, optional (default=1)
        Minimum number of observations required for calculating statistics
        
    Returns
    -------
    pd.DataFrame
        DataFrame with additional expanding features
    """
    logger.info(f"Creating expanding features for {target_col}")
    
    # Sort by date and groupby columns
    sort_cols = ["date"] + groupby_cols
    df = df.sort_values(sort_cols).copy()
    
    # Expanding mean
    df[f"{target_col}_exp_mean"] = df.groupby(groupby_cols)[target_col].transform(
        lambda x: x.expanding(min_periods=min_periods).mean()
    )
    
    # Expanding standard deviation
    df[f"{target_col}_exp_std"] = df.groupby(groupby_cols)[target_col].transform(
        lambda x: x.expanding(min_periods=min_periods).std()
    )
    
    # Expanding max
    df[f"{target_col}_exp_max"] = df.groupby(groupby_cols)[target_col].transform(
        lambda x: x.expanding(min_periods=min_periods).max()
    )
    
    # Expanding min
    df[f"{target_col}_exp_min"] = df.groupby(groupby_cols)[target_col].transform(
        lambda x: x.expanding(min_periods=min_periods).min()
    )
    
    return df


def create_target_encoding_features(
    df: pd.DataFrame,
    target_col: str,
    categorical_cols: List[str],
    train_indices: Optional[List[int]] = None
) -> pd.DataFrame:
    """
    Create target encoding features for categorical variables.
    If train_indices is provided, use only those indices for calculating the encoding.
    
    Parameters
    ----------
    df : pd.DataFrame
        DataFrame containing the target and categorical columns
    target_col : str
        Name of the target column
    categorical_cols : List[str]
        List of categorical columns to encode
    train_indices : Optional[List[int]]
        Indices of training data to use for encoding
        
    Returns
    -------
    pd.DataFrame
        DataFrame with additional target encoding features
    """
    logger.info(f"Creating target encoding features for {categorical_cols}")
    
    result = df.copy()
    
    # Get training data if train_indices is provided
    train_df = df.iloc[train_indices].copy() if train_indices is not None else df.copy()
    
    # Calculate the global mean
    global_mean = train_df[target_col].mean()
    
    # Create encoding for each categorical column
    for col in categorical_cols:
        # Calculate mean target value for each category
        encoding = train_df.groupby(col)[target_col].agg(["mean", "count"]).reset_index()
        encoding.columns = [col, f"{col}_{target_col}_mean", f"{col}_{target_col}_count"]
        
        # Merge encoding with original data
        result = result.merge(encoding, on=col, how="left")
        
        # Fill missing values with global mean
        result[f"{col}_{target_col}_mean"] = result[f"{col}_{target_col}_mean"].fillna(global_mean)
        result[f"{col}_{target_col}_count"] = result[f"{col}_{target_col}_count"].fillna(0)
    
    return result


def add_interaction_features(
    df: pd.DataFrame,
    feature_cols: List[str]
) -> pd.DataFrame:
    """
    Create interaction features by multiplying pairs of numerical features.
    
    Parameters
    ----------
    df : pd.DataFrame
        DataFrame containing the feature columns
    feature_cols : List[str]
        List of numerical feature columns to create interactions for
        
    Returns
    -------
    pd.DataFrame
        DataFrame with additional interaction features
    """
    logger.info("Creating interaction features")
    
    result = df.copy()
    
    # Create interactions for pairs of features
    for i, col1 in enumerate(feature_cols):
        for col2 in feature_cols[i+1:]:
            interaction_name = f"{col1}_x_{col2}"
            result[interaction_name] = result[col1] * result[col2]
    
    return result


def add_promotion_features(
    df: pd.DataFrame,
    promotion_col: str = "onpromotion"
) -> pd.DataFrame:
    """
    Create features related to promotions, including historical promotion patterns.
    
    Parameters
    ----------
    df : pd.DataFrame
        DataFrame containing the promotion column
    promotion_col : str, optional (default="onpromotion")
        Name of the promotion column
        
    Returns
    -------
    pd.DataFrame
        DataFrame with additional promotion features
    """
    logger.info("Creating promotion features")
    
    result = df.copy()
    
    # Ensure promotion column is numeric
    result[promotion_col] = result[promotion_col].astype(int)
    
    # Calculate rolling promotion statistics by store and family
    groupby_cols = ["store_nbr", "family"]
    
    # Create 7-day, 14-day, and 30-day rolling promotion counts
    for window in [7, 14, 30]:
        col_name = f"{promotion_col}_roll_{window}_count"
        result[col_name] = result.groupby(groupby_cols)[promotion_col].transform(
            lambda x: x.rolling(window, min_periods=1).sum()
        )
    
    # Create promotion lag features (was on promotion N days ago)
    for lag in [1, 7, 14, 28]:
        col_name = f"{promotion_col}_lag_{lag}"
        result[col_name] = result.groupby(groupby_cols)[promotion_col].shift(lag)
    
    # Fill missing values with 0
    promotion_feature_cols = [col for col in result.columns if promotion_col in col and col != promotion_col]
    result[promotion_feature_cols] = result[promotion_feature_cols].fillna(0)
    
    return result


def process_holiday_features(
    df: pd.DataFrame,
    holidays_df: pd.DataFrame
) -> pd.DataFrame:
    """
    Process holidays and merge holiday features into the main dataframe.
    
    Parameters
    ----------
    df : pd.DataFrame
        Main dataframe with date column
    holidays_df : pd.DataFrame
        Dataframe with holiday information
        
    Returns
    -------
    pd.DataFrame
        DataFrame with additional holiday features
    """
    logger.info("Processing holiday features")
    
    result = df.copy()
    
    # Convert date columns to datetime
    result["date"] = pd.to_datetime(result["date"])
    holidays_df["date"] = pd.to_datetime(holidays_df["date"])
    
    # Keep only non-transferred holidays
    holidays_df = holidays_df[holidays_df["transferred"] == False].copy()
    
    # Create holiday type indicators
    holidays_df["is_holiday"] = 1
    
    # Create separate indicators by locale and type
    for locale in ["National", "Regional", "Local"]:
        locale_holidays = holidays_df[holidays_df["locale"] == locale][["date", "is_holiday"]].copy()
        locale_holidays.rename(columns={"is_holiday": f"is_{locale.lower()}_holiday"}, inplace=True)
        result = result.merge(locale_holidays, on="date", how="left")
    
    # Fill missing values with 0
    holiday_cols = [col for col in result.columns if "holiday" in col]
    result[holiday_cols] = result[holiday_cols].fillna(0)
    
    # Add days-until-next-holiday and days-since-last-holiday features
    all_dates = pd.DataFrame({"date": pd.date_range(result["date"].min(), result["date"].max(), freq="D")})
    holiday_dates = holidays_df[holidays_df["locale"] == "National"]["date"].unique()
    
    all_dates["is_holiday"] = all_dates["date"].isin(holiday_dates).astype(int)
    
    # Calculate days until next holiday
    all_dates["next_holiday"] = all_dates["date"].shift(-1)
    all_dates.loc[all_dates["is_holiday"] == 1, "next_holiday"] = all_dates.loc[all_dates["is_holiday"] == 1, "date"]
    all_dates["next_holiday"] = all_dates["next_holiday"].fillna(method="bfill")
    all_dates["days_to_next_holiday"] = (all_dates["next_holiday"] - all_dates["date"]).dt.days
    
    # Calculate days since last holiday
    all_dates["last_holiday"] = all_dates["date"].shift(1)
    all_dates.loc[all_dates["is_holiday"] == 1, "last_holiday"] = all_dates.loc[all_dates["is_holiday"] == 1, "date"]
    all_dates["last_holiday"] = all_dates["last_holiday"].fillna(method="ffill")
    all_dates["days_since_last_holiday"] = (all_dates["date"] - all_dates["last_holiday"]).dt.days
    
    # Merge holiday distance features
    result = result.merge(
        all_dates[["date", "days_to_next_holiday", "days_since_last_holiday"]],
        on="date",
        how="left"
    )
    
    return result 