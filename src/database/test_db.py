#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Script to test database operations.
"""

import sys
import os
import logging
from pathlib import Path
from datetime import datetime, timedelta
import random
import numpy as np

# Add root directory to path
project_root = Path(__file__).parents[2]
sys.path.insert(0, str(project_root))

from src.database.database import db_session
from src.database.repository import (
    StoreRepository,
    ProductFamilyRepository,
    HistoricalSalesRepository,
    PredictionRepository,
    ModelMetricRepository,
    FeatureImportanceRepository
)

# Logging configuration
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)


def test_add_prediction():
    """Test for adding a prediction to the database."""
    logger.info("Testing addition of prediction to the database")
    
    with db_session() as session:
        # Get a store and family
        store = StoreRepository.get_by_store_nbr(session, 1)
        family = ProductFamilyRepository.get_by_name(session, "GROCERY I")
        
        if not store or not family:
            logger.error("Store or family not found")
            return
        
        # Create prediction data
        prediction_date = datetime.now()
        target_date = prediction_date + timedelta(days=1)
        
        prediction_data = {
            "store_id": store.id,
            "family_id": family.id,
            "prediction_date": prediction_date,
            "target_date": target_date,
            "onpromotion": False,
            "predicted_sales": 250.75,
            "prediction_interval_lower": 230.5,
            "prediction_interval_upper": 270.0,
            "model_version": "1.0.0",
            "feature_values": {"day_of_week": 2, "is_weekend": 0, "onpromotion": 0}
        }
        
        # Save prediction
        prediction = PredictionRepository.create(session, prediction_data)
        logger.info(f"Prediction created with ID: {prediction.id}")
        
        # Retrieve prediction
        saved_prediction = PredictionRepository.get_by_id(session, prediction.id)
        logger.info(f"Retrieved prediction: {saved_prediction}")
        
        return saved_prediction.id


def test_add_metrics():
    """Test for adding model metrics to the database."""
    logger.info("Testing addition of metrics to the database")
    
    with db_session() as session:
        # Create metrics data
        metrics_data = [
            {
                "model_name": "store-sales-forecaster",
                "model_version": "1.0.0",
                "metric_name": "rmse",
                "metric_value": 0.45,
                "timestamp": datetime.now()
            },
            {
                "model_name": "store-sales-forecaster",
                "model_version": "1.0.0",
                "metric_name": "mae",
                "metric_value": 0.32,
                "timestamp": datetime.now()
            },
            {
                "model_name": "store-sales-forecaster",
                "model_version": "1.0.0",
                "metric_name": "mape",
                "metric_value": 12.5,
                "timestamp": datetime.now()
            },
            {
                "model_name": "store-sales-forecaster",
                "model_version": "1.0.0",
                "metric_name": "r2",
                "metric_value": 0.87,
                "timestamp": datetime.now()
            }
        ]
        
        # Save metrics
        metrics = ModelMetricRepository.create_many(session, metrics_data)
        logger.info(f"Metrics created: {len(metrics)}")
        
        # Retrieve metrics
        latest_metrics = ModelMetricRepository.get_latest_metrics(
            session, "store-sales-forecaster"
        )
        logger.info(f"Retrieved metrics: {latest_metrics}")
        
        return latest_metrics


def test_add_feature_importance():
    """Test for adding feature importance to the database."""
    logger.info("Testing addition of feature importance to the database")
    
    with db_session() as session:
        # Create feature importance data
        features = [
            "day_of_week", "month", "day_of_month", "onpromotion",
            "is_weekend", "is_month_start", "is_month_end",
            "store_nbr", "family_GROCERY I", "family_BEVERAGES"
        ]
        
        importance_data_list = []
        total_importance = 0
        
        # Generate random importance values
        raw_values = [random.random() for _ in range(len(features))]
        total = sum(raw_values)
        normalized_values = [value / total for value in raw_values]
        
        # Create importance records
        for i, feature in enumerate(features):
            importance_data_list.append({
                "model_name": "store-sales-forecaster",
                "model_version": "1.0.0",
                "feature_name": feature,
                "importance_value": normalized_values[i],
                "timestamp": datetime.now()
            })
        
        # Save feature importance
        importance_records = FeatureImportanceRepository.create_many(
            session, importance_data_list
        )
        logger.info(f"Feature importance records created: {len(importance_records)}")
        
        # Retrieve feature importance
        importance_df = FeatureImportanceRepository.get_feature_importance_as_dataframe(
            session, "store-sales-forecaster", "1.0.0"
        )
        logger.info(f"Feature importance records retrieved: {len(importance_df)}")
        
        return importance_df


def test_query_historical_sales():
    """Test for querying historical sales from the database."""
    logger.info("Testing query of historical sales from the database")
    
    with db_session() as session:
        # Get a store and family
        store = StoreRepository.get_by_store_nbr(session, 1)
        family = ProductFamilyRepository.get_by_name(session, "GROCERY I")
        
        if not store or not family:
            logger.error("Store or family not found")
            return
        
        # Query sales history
        sales_df = HistoricalSalesRepository.get_sales_history_as_dataframe(
            session, store_id=store.id, family_id=family.id, days=30
        )
        
        logger.info(f"Historical sales retrieved: {len(sales_df)}")
        if not sales_df.empty:
            logger.info(f"First rows:\n{sales_df.head()}")
        
        return sales_df


def test_add_more_historical_sales():
    """Test for adding more historical sales to the database."""
    logger.info("Testing addition of more historical sales to the database")
    
    with db_session() as session:
        # Get stores and families
        stores = StoreRepository.get_all(session)
        families = ProductFamilyRepository.get_all(session)
        
        if not stores or not families:
            logger.error("Stores or families not found")
            return
        
        # Limit to 5 stores and 3 families to avoid creating too much data
        stores = stores[:5]
        families = families[:3]
        
        # Generate historical sales for the last 30 days
        end_date = datetime.now().date()
        start_date = end_date - timedelta(days=30)
        
        total_records = 0
        
        # Seasonality factors
        day_of_week_factors = {0: 0.8, 1: 0.9, 2: 1.0, 3: 1.0, 4: 1.2, 5: 1.5, 6: 0.7}  # Mon-Sun
        
        for store in stores:
            for family in families:
                sales_data_list = []
                
                for days_ago in range(30, 0, -1):
                    current_date = end_date - timedelta(days=days_ago)
                    current_datetime = datetime.combine(current_date, datetime.min.time())
                    
                    # Day factor
                    day_factor = day_of_week_factors[current_date.weekday()]
                    
                    # Monthly factor (seasonal effect)
                    month_factor = 1.0 + 0.1 * np.sin(2 * np.pi * current_date.month / 12)
                    
                    # Store factor (stores with different volumes of sales have different factors)
                    store_factor = 0.5 + 1.5 * store.id / len(stores)
                    
                    # Family factor (products with different volumes of sales have different factors)
                    family_factor = 0.5 + 1.5 * family.id / len(families)
                    
                    # Random promotion
                    onpromotion = random.random() < 0.2
                    promotion_factor = 1.3 if onpromotion else 1.0
                    
                    # Base sales with seasonal effect and factors
                    base_sales = 100 * day_factor * month_factor * store_factor * family_factor * promotion_factor
                    
                    # Add random noise
                    noise = random.uniform(0.8, 1.2)
                    sales = base_sales * noise
                    
                    # Create sales data
                    sales_data = {
                        "store_id": store.id,
                        "family_id": family.id,
                        "date": current_datetime,
                        "sales": sales,
                        "onpromotion": onpromotion
                    }
                    
                    sales_data_list.append(sales_data)
                
                # Insert batch
                if sales_data_list:
                    HistoricalSalesRepository.create_many(session, sales_data_list)
                    total_records += len(sales_data_list)
                    logger.info(f"Inserted {len(sales_data_list)} historical records for store {store.store_nbr}, family {family.name}")
        
        logger.info(f"Total historical records inserted: {total_records}")
        return total_records


def test_add_future_predictions():
    """Test for adding future predictions to the database."""
    logger.info("Testing addition of future predictions to the database")
    
    with db_session() as session:
        # Get stores and families
        stores = StoreRepository.get_all(session)
        families = ProductFamilyRepository.get_all(session)
        
        if not stores or not families:
            logger.error("Stores or families not found")
            return
        
        # Limit to 5 stores and 3 families to avoid creating too much data
        stores = stores[:5]
        families = families[:3]
        
        # Generate predictions for the next 14 days
        prediction_date = datetime.now()
        
        total_records = 0
        
        # Seasonality factors
        day_of_week_factors = {0: 0.8, 1: 0.9, 2: 1.0, 3: 1.0, 4: 1.2, 5: 1.5, 6: 0.7}  # Mon-Sun
        
        for store in stores:
            for family in families:
                prediction_data_list = []
                
                for days_ahead in range(1, 15):
                    target_date = prediction_date + timedelta(days=days_ahead)
                    
                    # Day factor
                    day_factor = day_of_week_factors[target_date.weekday()]
                    
                    # Monthly factor (seasonal effect)
                    month_factor = 1.0 + 0.1 * np.sin(2 * np.pi * target_date.month / 12)
                    
                    # Store factor (stores with different volumes of sales have different factors)
                    store_factor = 0.5 + 1.5 * store.id / len(stores)
                    
                    # Family factor (products with different volumes of sales have different factors)
                    family_factor = 0.5 + 1.5 * family.id / len(families)
                    
                    # Random promotion
                    onpromotion = random.random() < 0.2
                    promotion_factor = 1.3 if onpromotion else 1.0
                    
                    # Base sales with seasonal effect and factors
                    base_sales = 100 * day_factor * month_factor * store_factor * family_factor * promotion_factor
                    
                    # Add random noise
                    noise = random.uniform(0.8, 1.2)
                    predicted_sales = base_sales * noise
                    
                    # Calculate prediction intervals
                    lower_bound = predicted_sales * 0.9
                    upper_bound = predicted_sales * 1.1
                    
                    # Create prediction data
                    prediction_data = {
                        "store_id": store.id,
                        "family_id": family.id,
                        "prediction_date": prediction_date,
                        "target_date": target_date,
                        "onpromotion": onpromotion,
                        "predicted_sales": predicted_sales,
                        "prediction_interval_lower": lower_bound,
                        "prediction_interval_upper": upper_bound,
                        "model_version": "1.0.0",
                        "feature_values": {
                            "day_of_week": target_date.weekday(),
                            "is_weekend": 1 if target_date.weekday() >= 5 else 0,
                            "onpromotion": 1 if onpromotion else 0
                        }
                    }
                    
                    # Save prediction
                    prediction = PredictionRepository.create(session, prediction_data)
                    total_records += 1
                
                logger.info(f"Inserted {days_ahead} predictions for store {store.store_nbr}, family {family.name}")
        
        logger.info(f"Total predictions inserted: {total_records}")
        return total_records


if __name__ == "__main__":
    """Run database tests."""
    logger.info("Starting database tests")
    
    try:
        prediction_id = test_add_prediction()
        logger.info(f"Prediction created with ID: {prediction_id}")
        
        metrics = test_add_metrics()
        logger.info(f"Metrics added: {metrics}")
        
        importance_df = test_add_feature_importance()
        logger.info(f"Feature importance added")
        
        # Add more tests
        historical_count = test_add_more_historical_sales()
        logger.info(f"Inserted {historical_count} historical sales records")
        
        predictions_count = test_add_future_predictions()
        logger.info(f"Inserted {predictions_count} future predictions")
        
        sales_df = test_query_historical_sales()
        logger.info("Historical sales query completed")
        
        logger.info("All tests completed successfully")
    
    except Exception as e:
        logger.error(f"Error during tests: {e}")
        sys.exit(1) 