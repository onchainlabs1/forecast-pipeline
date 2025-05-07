#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Script to download and load the Favorita Store Sales dataset from Kaggle.
"""

import os
import logging
from pathlib import Path
import subprocess
import zipfile

import pandas as pd
from dotenv import load_dotenv

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)

# Load environment variables
load_dotenv()

# Define paths
PROJECT_DIR = Path(__file__).resolve().parents[2]
RAW_DATA_DIR = PROJECT_DIR / "data" / "raw"
KAGGLE_COMPETITION = "store-sales-time-series-forecasting"


def create_directories():
    """Create necessary directories if they don't exist."""
    os.makedirs(RAW_DATA_DIR, exist_ok=True)
    logger.info(f"Created directory: {RAW_DATA_DIR}")


def download_kaggle_dataset():
    """
    Download dataset from Kaggle using the Kaggle API.
    Requires Kaggle API credentials to be set up.
    """
    try:
        logger.info(f"Downloading dataset from Kaggle competition: {KAGGLE_COMPETITION}")
        
        # Check if Kaggle credentials are available
        kaggle_cred_path = Path.home() / ".kaggle" / "kaggle.json"
        if not kaggle_cred_path.exists():
            logger.warning(
                "Kaggle credentials not found. Please set up your Kaggle API credentials "
                "by creating a token at https://www.kaggle.com/account and placing it in "
                f"{kaggle_cred_path}"
            )
            return False
        
        # Download the dataset
        subprocess.run(
            [
                "kaggle",
                "competitions",
                "download",
                "-c",
                KAGGLE_COMPETITION,
                "-p",
                str(RAW_DATA_DIR),
            ],
            check=True,
        )
        
        # Extract the ZIP file
        zip_path = RAW_DATA_DIR / f"{KAGGLE_COMPETITION}.zip"
        with zipfile.ZipFile(zip_path, "r") as zip_ref:
            zip_ref.extractall(RAW_DATA_DIR)
        
        # Remove the ZIP file
        os.remove(zip_path)
        logger.info("Dataset downloaded and extracted successfully")
        return True
    
    except Exception as e:
        logger.error(f"Failed to download dataset: {e}")
        return False


def load_data():
    """
    Load the Favorita Store Sales dataset.
    Returns a dictionary with all dataframes.
    """
    datasets = {}
    
    try:
        # List of files to load
        files = [
            "train.csv",
            "test.csv",
            "holidays_events.csv",
            "oil.csv",
            "stores.csv",
            "transactions.csv",
        ]
        
        # Load each file
        for file in files:
            file_path = RAW_DATA_DIR / file
            if file_path.exists():
                logger.info(f"Loading {file}")
                datasets[file.split('.')[0]] = pd.read_csv(file_path)
            else:
                logger.warning(f"File not found: {file_path}")
        
        # Display basic information
        for name, df in datasets.items():
            logger.info(f"{name}: {df.shape} rows x columns")
        
        return datasets
    
    except Exception as e:
        logger.error(f"Error loading data: {e}")
        return {}


def main():
    """Main function to execute the data loading process."""
    create_directories()
    
    # Check if files already exist
    if (RAW_DATA_DIR / "train.csv").exists():
        logger.info("Dataset files already exist, skipping download")
    else:
        success = download_kaggle_dataset()
        if not success:
            logger.error("Failed to download the dataset, please download it manually")
            logger.info(
                f"Place the dataset files in the {RAW_DATA_DIR} directory "
                "after downloading from Kaggle"
            )
    
    # Load and display data summary
    datasets = load_data()
    if datasets:
        logger.info("Data loaded successfully")
    else:
        logger.error("Failed to load data")


if __name__ == "__main__":
    main() 