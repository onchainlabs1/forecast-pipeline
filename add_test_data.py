#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Script to add historical sales data and matching predictions for testing forecast accuracy.
This allows us to verify that the forecast accuracy calculation is working correctly.
"""

import sys
import logging
from pathlib import Path
from datetime import datetime, timedelta
import random
import numpy as np

# Add the project root to the path
project_root = Path(__file__).parents[0]
sys.path.insert(0, str(project_root))

from src.database.database import db_session
from src.database.repository import (
    StoreRepository,
    ProductFamilyRepository,
    HistoricalSalesRepository,
    PredictionRepository
)

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)

def add_test_data_for_accuracy():
    """
    Add historical sales data and matching predictions for the past 30 days
    with a known accuracy level for testing.
    """
    logger.info("Adding test data for forecast accuracy verification")
    
    with db_session() as session:
        # Get stores and families
        stores = StoreRepository.get_all(session)
        families = ProductFamilyRepository.get_all(session)
        
        if not stores or not families:
            logger.error("No stores or families found")
            return
        
        # Use a subset of stores and families to limit data volume
        stores = stores[:3]  # First 3 stores
        families = families[:2]  # First 2 families
        
        # Generate data for the past 30 days
        end_date = datetime.now().date()
        start_date = end_date - timedelta(days=30)
        
        # Set an expected accuracy level we want to achieve
        target_accuracy = 80.0  # 80% accuracy (= 20% MAPE)
        
        # Track the total error to ensure we hit our target accuracy
        total_absolute_error = 0
        total_squared_error = 0
        total_error = 0
        sum_actuals = 0
        count = 0
        
        # List to store all data points for detailed logging
        all_data_points = []
        
        logger.info(f"Generating test data with target accuracy: {target_accuracy}%")
        
        # Generate data for each day, store, and family
        for days_ago in range(30, 0, -1):
            current_date = end_date - timedelta(days=days_ago)
            current_datetime = datetime.combine(current_date, datetime.min.time())
            
            for store in stores:
                for family in families:
                    # Generate a base sales value (simulating actual sales)
                    base_sales = 100 + (store.id * 50) + (family.id * 20)
                    
                    # Add some seasonal variation
                    day_factor = 0.8 + (current_date.weekday() * 0.05)  # Weekday effect
                    month_factor = 1.0 + 0.1 * np.sin(2 * np.pi * current_date.month / 12)  # Month effect
                    
                    # Calculate the actual sales
                    actual_sales = base_sales * day_factor * month_factor
                    # Add some random noise to make it realistic
                    actual_sales *= random.uniform(0.9, 1.1)
                    
                    # Ensure actual sales is positive
                    actual_sales = max(1.0, actual_sales) # Prevent division by zero issues
                    
                    # Calculate a predicted sales value with controlled error
                    # To achieve target MAPE (100 - accuracy), we'll introduce errors
                    # of approximately that magnitude
                    error_factor = (100 - target_accuracy) / 100  # e.g., 20% error
                    
                    # Generate an error that's sometimes positive, sometimes negative
                    # but with absolute value around our target error
                    error_direction = 1 if random.random() > 0.5 else -1
                    error_size = error_factor * random.uniform(0.8, 1.2)  # Vary the error by ±20%
                    
                    # Calculate the predicted sales value
                    predicted_sales = actual_sales * (1 + (error_direction * error_size))
                    
                    # Calculate error metrics for this data point
                    error = predicted_sales - actual_sales
                    absolute_error = abs(error)
                    squared_error = error ** 2
                    percentage_error = (absolute_error / actual_sales) * 100
                    
                    # Store detailed data for logging
                    all_data_points.append((float(predicted_sales), float(actual_sales), 
                                          float(error), float(percentage_error)))
                    
                    # Track the metrics to validate we're hitting the target accuracy
                    total_error += error
                    total_absolute_error += absolute_error
                    total_squared_error += squared_error
                    sum_actuals += actual_sales
                    count += 1
                    
                    # Create the historical sales record
                    sales_data = {
                        "store_id": store.id,
                        "family_id": family.id,
                        "date": current_datetime,
                        "sales": actual_sales,
                        "onpromotion": random.random() < 0.2  # 20% chance of promotion
                    }
                    
                    # Check if sales data already exists for this store/family/date
                    existing_sales = HistoricalSalesRepository.get_for_store_family_date(
                        session, store.id, family.id, current_datetime
                    )
                    
                    if existing_sales:
                        # Update existing record
                        existing_sales.sales = actual_sales
                        session.commit()
                    else:
                        # Create new record
                        HistoricalSalesRepository.create(session, sales_data)
                    
                    # Create a matching prediction record for the same date
                    prediction_data = {
                        "store_id": store.id,
                        "family_id": family.id,
                        "prediction_date": current_datetime - timedelta(days=1),  # Predicted 1 day before
                        "target_date": current_datetime,
                        "onpromotion": sales_data["onpromotion"],
                        "predicted_sales": predicted_sales,
                        "prediction_interval_lower": predicted_sales * 0.9,
                        "prediction_interval_upper": predicted_sales * 1.1,
                        "model_version": "1.0.0",
                        "feature_values": {
                            "day_of_week": current_date.weekday(),
                            "is_weekend": 1 if current_date.weekday() >= 5 else 0,
                            "onpromotion": 1 if sales_data["onpromotion"] else 0
                        }
                    }
                    
                    # Create the prediction record
                    PredictionRepository.create(session, prediction_data)
        
        # Calculate the actual accuracy we achieved
        if count > 0:
            # Calculate the same way as in the API endpoint
            avg_actual = sum_actuals / count
            mape = (total_absolute_error / count) / avg_actual * 100
            mae = total_absolute_error / count
            rmse = (total_squared_error / count) ** 0.5
            mean_error = total_error / count
            forecast_accuracy = max(0, 100 - mape)
            
            logger.info(f"Added {count} test data points with actual accuracy: {forecast_accuracy:.2f}%")
            logger.info(f"MAPE: {mape:.2f}%, MAE: {mae:.2f}, RMSE: {rmse:.2f}, Mean Error: {mean_error:.2f}")
            
            # Log a few sample data points
            for i, (pred, actual, error, pct_error) in enumerate(all_data_points[:5]):
                logger.info(f"Sample {i+1}: Predicted={pred:.2f}, Actual={actual:.2f}, Error={error:.2f} ({pct_error:.2f}%)")
        else:
            logger.warning("No test data was added")

if __name__ == "__main__":
    add_test_data_for_accuracy()
    print("Test data for forecast accuracy verification added successfully.") 