#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Script to preprocess the Favorita Store Sales dataset and create features for modeling.
"""

import os
import logging
from pathlib import Path
import pandas as pd
import numpy as np
from datetime import datetime

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)

# Define paths
PROJECT_DIR = Path(__file__).resolve().parents[2]
RAW_DATA_DIR = PROJECT_DIR / "data" / "raw"
PROCESSED_DATA_DIR = PROJECT_DIR / "data" / "processed"


def create_directories():
    """Create necessary directories if they don't exist."""
    os.makedirs(PROCESSED_DATA_DIR, exist_ok=True)
    logger.info(f"Created directory: {PROCESSED_DATA_DIR}")


def load_raw_data():
    """
    Load the raw Favorita Store Sales datasets.
    Returns a dictionary with all dataframes.
    """
    datasets = {}
    files = [
        "train.csv",
        "test.csv",
        "holidays_events.csv",
        "oil.csv",
        "stores.csv",
        "transactions.csv",
    ]
    
    for file in files:
        file_path = RAW_DATA_DIR / file
        if file_path.exists():
            logger.info(f"Loading {file}")
            datasets[file.split(".")[0]] = pd.read_csv(file_path)
        else:
            logger.warning(f"File not found: {file_path}")
    
    return datasets


def preprocess_dates(df):
    """
    Process date columns and extract date features.
    
    Parameters
    ----------
    df : pandas.DataFrame
        DataFrame with a 'date' column
        
    Returns
    -------
    pandas.DataFrame
        DataFrame with additional date features
    """
    logger.info("Processing date features")
    
    # Convert date column to datetime
    df["date"] = pd.to_datetime(df["date"])
    
    # Extract date features
    df["day_of_week"] = df["date"].dt.dayofweek
    df["day_of_month"] = df["date"].dt.day
    df["month"] = df["date"].dt.month
    df["year"] = df["date"].dt.year
    df["quarter"] = df["date"].dt.quarter
    df["is_weekend"] = df["day_of_week"].isin([5, 6]).astype(int)
    df["is_month_start"] = df["date"].dt.is_month_start.astype(int)
    df["is_month_end"] = df["date"].dt.is_month_end.astype(int)
    
    return df


def preprocess_store_data(df, stores_df):
    """
    Merge store information into the main dataframe.
    
    Parameters
    ----------
    df : pandas.DataFrame
        Main dataframe with store_nbr column
    stores_df : pandas.DataFrame
        Stores metadata
        
    Returns
    -------
    pandas.DataFrame
        DataFrame with additional store features
    """
    logger.info("Processing store features")
    
    # Merge store information
    df = df.merge(stores_df, on="store_nbr", how="left")
    
    # Convert categorical variables to dummy variables
    df = pd.get_dummies(df, columns=["city", "state", "type"], drop_first=True)
    
    # Fill missing values in store-related columns
    store_cols = [col for col in df.columns if col.startswith(("city_", "state_", "type_"))]
    df[store_cols] = df[store_cols].fillna(0)
    
    return df


def preprocess_oil_data(df, oil_df):
    """
    Merge oil price information into the main dataframe.
    
    Parameters
    ----------
    df : pandas.DataFrame
        Main dataframe with date column
    oil_df : pandas.DataFrame
        Oil price data
        
    Returns
    -------
    pandas.DataFrame
        DataFrame with additional oil price features
    """
    logger.info("Processing oil price features")
    
    # Convert date column in oil dataframe
    oil_df["date"] = pd.to_datetime(oil_df["date"])
    
    # Fill missing oil prices using forward fill
    oil_df = oil_df.sort_values("date")
    oil_df["dcoilwtico"] = oil_df["dcoilwtico"].fillna(method="ffill")
    
    # Add 7-day and 30-day rolling averages
    oil_df["oil_price_7d_avg"] = oil_df["dcoilwtico"].rolling(7, min_periods=1).mean()
    oil_df["oil_price_30d_avg"] = oil_df["dcoilwtico"].rolling(30, min_periods=1).mean()
    
    # Merge with main dataframe
    df = df.merge(oil_df, on="date", how="left")
    
    # Fill any remaining missing values
    oil_cols = ["dcoilwtico", "oil_price_7d_avg", "oil_price_30d_avg"]
    df[oil_cols] = df[oil_cols].fillna(method="ffill").fillna(method="bfill")
    
    return df


def preprocess_holidays(df, holidays_df):
    """
    Process holidays and events information.
    
    Parameters
    ----------
    df : pandas.DataFrame
        Main dataframe with date column
    holidays_df : pandas.DataFrame
        Holidays and events data
        
    Returns
    -------
    pandas.DataFrame
        DataFrame with additional holiday features
    """
    logger.info("Processing holidays and events features")
    
    # Convert date column in holidays dataframe
    holidays_df["date"] = pd.to_datetime(holidays_df["date"])
    
    # Keep only relevant columns and active holidays (not transferred)
    holidays_df = holidays_df[holidays_df["transferred"] == False]
    holidays_df = holidays_df[["date", "type", "locale", "description"]]
    
    # Create binary indicators for holidays
    holidays_df["is_holiday"] = 1
    
    # Create separate indicators for national, regional, and local holidays
    holidays_national = holidays_df[holidays_df["locale"] == "National"][["date", "is_holiday"]].copy()
    holidays_national.rename(columns={"is_holiday": "is_national_holiday"}, inplace=True)
    
    holidays_regional = holidays_df[holidays_df["locale"] == "Regional"][["date", "is_holiday"]].copy()
    holidays_regional.rename(columns={"is_holiday": "is_regional_holiday"}, inplace=True)
    
    holidays_local = holidays_df[holidays_df["locale"] == "Local"][["date", "is_holiday"]].copy()
    holidays_local.rename(columns={"is_holiday": "is_local_holiday"}, inplace=True)
    
    # Merge with main dataframe
    df = df.merge(holidays_national, on="date", how="left")
    df = df.merge(holidays_regional, on="date", how="left")
    df = df.merge(holidays_local, on="date", how="left")
    
    # Fill missing values with 0 (no holiday)
    holiday_cols = ["is_national_holiday", "is_regional_holiday", "is_local_holiday"]
    df[holiday_cols] = df[holiday_cols].fillna(0)
    
    return df


def add_lag_features(df, target_col, groupby_cols, lag_days):
    """
    Add lag features for a target column.
    
    Parameters
    ----------
    df : pandas.DataFrame
        DataFrame with the target column
    target_col : str
        Column to create lags for
    groupby_cols : list
        Columns to group by before creating lags
    lag_days : list
        List of lag days to create
        
    Returns
    -------
    pandas.DataFrame
        DataFrame with additional lag features
    """
    logger.info(f"Creating lag features for {target_col}")
    
    # Sort by date and groupby columns
    df = df.sort_values(["date"] + groupby_cols)
    
    # Create a copy to avoid SettingWithCopyWarning
    result = df.copy()
    
    # Create lag features
    for lag in lag_days:
        lag_col_name = f"{target_col}_lag_{lag}"
        result[lag_col_name] = result.groupby(groupby_cols)[target_col].shift(lag)
    
    return result


def add_rolling_features(df, target_col, groupby_cols, windows, min_periods=1):
    """
    Add rolling window features for a target column.
    
    Parameters
    ----------
    df : pandas.DataFrame
        DataFrame with the target column
    target_col : str
        Column to create rolling features for
    groupby_cols : list
        Columns to group by before creating rolling features
    windows : list
        List of window sizes to create
    min_periods : int
        Minimum number of observations required
        
    Returns
    -------
    pandas.DataFrame
        DataFrame with additional rolling features
    """
    logger.info(f"Creating rolling features for {target_col}")
    
    # Sort by date and groupby columns
    df = df.sort_values(["date"] + groupby_cols)
    
    # Create a copy to avoid SettingWithCopyWarning
    result = df.copy()
    
    # Group by the specified columns
    for window in windows:
        # Rolling mean
        mean_col_name = f"{target_col}_roll_{window}_mean"
        result[mean_col_name] = result.groupby(groupby_cols)[target_col].transform(
            lambda x: x.rolling(window, min_periods=min_periods).mean()
        )
        
        # Rolling standard deviation
        std_col_name = f"{target_col}_roll_{window}_std"
        result[std_col_name] = result.groupby(groupby_cols)[target_col].transform(
            lambda x: x.rolling(window, min_periods=min_periods).std()
        )
        
        # Rolling max
        max_col_name = f"{target_col}_roll_{window}_max"
        result[max_col_name] = result.groupby(groupby_cols)[target_col].transform(
            lambda x: x.rolling(window, min_periods=min_periods).max()
        )
        
        # Rolling min
        min_col_name = f"{target_col}_roll_{window}_min"
        result[min_col_name] = result.groupby(groupby_cols)[target_col].transform(
            lambda x: x.rolling(window, min_periods=min_periods).min()
        )
    
    return result


def add_target_encoding(df, target_col, categorical_cols):
    """
    Add target encoding features for categorical columns.
    
    Parameters
    ----------
    df : pandas.DataFrame
        DataFrame with the target column and categorical columns
    target_col : str
        Target column to use for encoding
    categorical_cols : list
        List of categorical columns to encode
        
    Returns
    -------
    pandas.DataFrame
        DataFrame with additional target encoding features
    """
    logger.info(f"Creating target encoding features for {categorical_cols}")
    
    # Create a copy to avoid SettingWithCopyWarning
    result = df.copy()
    
    # Apply target encoding to each categorical column
    for col in categorical_cols:
        # Calculate the global mean
        global_mean = df[target_col].mean()
        
        # Calculate the grouped means
        grouped_means = df.groupby(col)[target_col].mean().reset_index()
        grouped_means.rename(columns={target_col: f"{col}_{target_col}_mean"}, inplace=True)
        
        # Merge the grouped means
        result = result.merge(grouped_means, on=col, how="left")
        
        # Fill missing values with the global mean
        result[f"{col}_{target_col}_mean"] = result[f"{col}_{target_col}_mean"].fillna(global_mean)
    
    return result


def process_train_data(datasets):
    """
    Process the training data.
    
    Parameters
    ----------
    datasets : dict
        Dictionary containing all the loaded dataframes
        
    Returns
    -------
    pandas.DataFrame
        Processed training data
    """
    logger.info("Processing training data")
    
    # Get relevant datasets
    train_df = datasets.get("train", None)
    stores_df = datasets.get("stores", None)
    oil_df = datasets.get("oil", None)
    holidays_df = datasets.get("holidays_events", None)
    transactions_df = datasets.get("transactions", None)
    
    if train_df is None:
        logger.error("Training data not found")
        return None
    
    # Initial checks and cleaning
    logger.info(f"Original training data shape: {train_df.shape}")
    
    # Drop rows with missing values in key columns
    cols_to_check = ["date", "store_nbr", "family", "sales", "onpromotion"]
    train_df = train_df.dropna(subset=cols_to_check)
    logger.info(f"Training data shape after dropping missing values: {train_df.shape}")
    
    # Add date features
    train_df = preprocess_dates(train_df)
    
    # Add store features if available
    if stores_df is not None:
        train_df = preprocess_store_data(train_df, stores_df)
    
    # Add oil price features if available
    if oil_df is not None:
        train_df = preprocess_oil_data(train_df, oil_df)
    
    # Add holiday features if available
    if holidays_df is not None:
        train_df = preprocess_holidays(train_df, holidays_df)
    
    # Add transaction features if available
    if transactions_df is not None:
        # Convert date column
        transactions_df["date"] = pd.to_datetime(transactions_df["date"])
        # Merge transactions data
        train_df = train_df.merge(transactions_df, on=["date", "store_nbr"], how="left")
        # Fill missing values
        train_df["transactions"] = train_df["transactions"].fillna(0)
    
    # Get categorical columns for encoding
    categorical_cols = ["family", "store_nbr"]
    
    # Add lag features for sales and transactions
    lag_days = [1, 7, 14, 28]
    train_df = add_lag_features(train_df, "sales", ["store_nbr", "family"], lag_days)
    if "transactions" in train_df.columns:
        train_df = add_lag_features(train_df, "transactions", ["store_nbr"], lag_days)
    
    # Add rolling window features for sales
    windows = [7, 14, 30]
    train_df = add_rolling_features(train_df, "sales", ["store_nbr", "family"], windows)
    
    # Add target encoding for categorical columns
    train_df = add_target_encoding(train_df, "sales", categorical_cols)
    
    # Handle missing values in lag and rolling features
    # These will be missing for the first rows of each group
    num_features = [col for col in train_df.columns if col.startswith(("sales_lag", "sales_roll", "transactions_lag"))]
    for col in num_features:
        train_df[col] = train_df[col].fillna(0)  # Simply fill with zeros as they are early data
    
    # Remove rows with date < 2015-01-28 (first 28 days) as they won't have all lag features
    min_date = pd.to_datetime("2015-01-28")
    train_filtered = train_df[train_df["date"] >= min_date].reset_index(drop=True)
    logger.info(f"Final training data shape: {train_filtered.shape}")
    
    return train_filtered


def process_test_data(datasets):
    """
    Process the test data similarly to the training data.
    
    Parameters
    ----------
    datasets : dict
        Dictionary containing all the loaded dataframes
        
    Returns
    -------
    pandas.DataFrame
        Processed test data
    """
    logger.info("Processing test data")
    
    # Get relevant datasets
    test_df = datasets.get("test", None)
    train_df = datasets.get("train", None)  # Need training data for lag features
    stores_df = datasets.get("stores", None)
    oil_df = datasets.get("oil", None)
    holidays_df = datasets.get("holidays_events", None)
    transactions_df = datasets.get("transactions", None)
    
    if test_df is None or train_df is None:
        logger.error("Test or training data not found")
        return None
    
    # Initial checks and cleaning
    logger.info(f"Original test data shape: {test_df.shape}")
    
    # Convert date column to datetime
    test_df["date"] = pd.to_datetime(test_df["date"])
    
    # Add date features
    test_df = preprocess_dates(test_df)
    
    # Add store features if available
    if stores_df is not None:
        test_df = preprocess_store_data(test_df, stores_df)
    
    # Add oil price features if available
    if oil_df is not None:
        test_df = preprocess_oil_data(test_df, oil_df)
    
    # Add holiday features if available
    if holidays_df is not None:
        test_df = preprocess_holidays(test_df, holidays_df)
    
    # Add transaction features if available (will have missing values for test period)
    if transactions_df is not None:
        # Convert date column
        transactions_df["date"] = pd.to_datetime(transactions_df["date"])
        # Merge transactions data
        test_df = test_df.merge(transactions_df, on=["date", "store_nbr"], how="left")
        # Fill missing values with the latest available value for each store
        last_transactions = transactions_df.groupby("store_nbr")["transactions"].last().reset_index()
        test_df = test_df.merge(
            last_transactions,
            on="store_nbr",
            how="left",
            suffixes=("", "_last")
        )
        test_df["transactions"] = test_df["transactions"].fillna(test_df["transactions_last"])
        test_df.drop("transactions_last", axis=1, inplace=True)
    
    # Combine train and test for generating lag features
    train_df["date"] = pd.to_datetime(train_df["date"])
    combined_df = pd.concat([train_df, test_df], sort=False)
    combined_df = combined_df.sort_values(["date", "store_nbr", "family"]).reset_index(drop=True)
    
    # Get categorical columns for encoding
    categorical_cols = ["family", "store_nbr"]
    
    # Add lag features for sales (and transactions if available)
    lag_days = [1, 7, 14, 28]
    combined_df = add_lag_features(combined_df, "sales", ["store_nbr", "family"], lag_days)
    if "transactions" in combined_df.columns:
        combined_df = add_lag_features(combined_df, "transactions", ["store_nbr"], lag_days)
    
    # Add rolling window features for sales
    windows = [7, 14, 30]
    combined_df = add_rolling_features(combined_df, "sales", ["store_nbr", "family"], windows)
    
    # Add target encoding for categorical columns
    # Use only training data for encoding to avoid leakage
    encoding_df = train_df.copy()
    for col in categorical_cols:
        # Calculate the global mean
        global_mean = encoding_df["sales"].mean()
        
        # Calculate the grouped means
        grouped_means = encoding_df.groupby(col)["sales"].mean().reset_index()
        grouped_means.rename(columns={"sales": f"{col}_sales_mean"}, inplace=True)
        
        # Merge the grouped means
        combined_df = combined_df.merge(grouped_means, on=col, how="left")
        
        # Fill missing values with the global mean
        combined_df[f"{col}_sales_mean"] = combined_df[f"{col}_sales_mean"].fillna(global_mean)
    
    # Extract only the test data from the combined dataframe
    test_processed = combined_df[combined_df["date"].isin(test_df["date"])].reset_index(drop=True)
    
    # Handle missing values in lag and rolling features
    num_features = [col for col in test_processed.columns if col.startswith(("sales_lag", "sales_roll", "transactions_lag"))]
    for col in num_features:
        test_processed[col] = test_processed[col].fillna(0)
    
    logger.info(f"Final test data shape: {test_processed.shape}")
    
    return test_processed


def main():
    """Main function to execute the preprocessing."""
    # Create output directory
    create_directories()
    
    # Load raw data
    datasets = load_raw_data()
    if not datasets:
        logger.error("Failed to load raw data")
        return
    
    # Process training data
    train_features = process_train_data(datasets)
    if train_features is not None:
        # Save processed training data
        train_output_path = PROCESSED_DATA_DIR / "train_features.csv"
        train_features.to_csv(train_output_path, index=False)
        logger.info(f"Processed training data saved to {train_output_path}")
    
    # Process test data
    test_features = process_test_data(datasets)
    if test_features is not None:
        # Save processed test data
        test_output_path = PROCESSED_DATA_DIR / "test_features.csv"
        test_features.to_csv(test_output_path, index=False)
        logger.info(f"Processed test data saved to {test_output_path}")


if __name__ == "__main__":
    main() 