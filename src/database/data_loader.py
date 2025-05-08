#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Load initial data into the database.
"""

import os
import logging
from pathlib import Path
from datetime import datetime, timedelta
import random
from typing import Dict, List, Any, Optional

import pandas as pd
import numpy as np
from sqlalchemy.orm import Session

from src.database.models import Store, ProductFamily, HistoricalSales
from src.database.repository import (
    StoreRepository, ProductFamilyRepository, HistoricalSalesRepository
)

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)

# Define paths
PROJECT_DIR = Path(__file__).resolve().parents[2]
DATA_DIR = PROJECT_DIR / "data"
RAW_DATA_DIR = DATA_DIR / "raw"


def load_stores(db: Session) -> None:
    """
    Load store data into the database.
    """
    try:
        # Check if stores.csv exists
        stores_path = RAW_DATA_DIR / "stores.csv"
        if not stores_path.exists():
            logger.warning(f"Stores file not found: {stores_path}")
            _create_demo_stores(db)
            return
        
        # Load stores from CSV
        stores_df = pd.read_csv(stores_path)
        logger.info(f"Loaded {len(stores_df)} stores from CSV")
        
        # Insert stores into database
        for _, row in stores_df.iterrows():
            store_data = {
                "store_nbr": int(row["store_nbr"]),
                "city": row.get("city", None),
                "state": row.get("state", None),
                "type": row.get("type", None),
                "cluster": row.get("cluster", None)
            }
            
            # Check if store already exists
            existing_store = StoreRepository.get_by_store_nbr(db, store_data["store_nbr"])
            if not existing_store:
                StoreRepository.create(db, store_data)
        
        logger.info("Stores loaded successfully")
    
    except Exception as e:
        logger.error(f"Error loading stores: {e}")
        # Create demo stores as fallback
        _create_demo_stores(db)


def _create_demo_stores(db: Session) -> None:
    """Create demo stores for testing."""
    logger.info("Creating demo stores")
    
    # List of demo stores
    demo_stores = [
        {"store_nbr": 1, "city": "Quito", "state": "Pichincha", "type": "A", "cluster": 1},
        {"store_nbr": 2, "city": "Guayaquil", "state": "Guayas", "type": "B", "cluster": 2},
        {"store_nbr": 3, "city": "Cuenca", "state": "Azuay", "type": "A", "cluster": 1},
        {"store_nbr": 4, "city": "Ambato", "state": "Tungurahua", "type": "C", "cluster": 3},
        {"store_nbr": 5, "city": "Machala", "state": "El Oro", "type": "B", "cluster": 2},
        {"store_nbr": 6, "city": "Quito", "state": "Pichincha", "type": "D", "cluster": 4},
        {"store_nbr": 7, "city": "Manta", "state": "Manabi", "type": "B", "cluster": 2},
        {"store_nbr": 8, "city": "Guayaquil", "state": "Guayas", "type": "A", "cluster": 1},
        {"store_nbr": 9, "city": "Loja", "state": "Loja", "type": "C", "cluster": 3},
        {"store_nbr": 10, "city": "Ibarra", "state": "Imbabura", "type": "D", "cluster": 4}
    ]
    
    # Insert stores into database
    for store_data in demo_stores:
        # Check if store already exists
        existing_store = StoreRepository.get_by_store_nbr(db, store_data["store_nbr"])
        if not existing_store:
            StoreRepository.create(db, store_data)
    
    logger.info(f"Created {len(demo_stores)} demo stores")


def load_product_families(db: Session) -> None:
    """
    Load product family data into the database.
    """
    try:
        # Check if we have a file with product families
        train_path = RAW_DATA_DIR / "train.csv"
        if train_path.exists():
            # Extract unique families from train data
            logger.info("Extracting product families from training data")
            
            # Memory-efficient reading of CSV to extract unique families
            chunks = pd.read_csv(train_path, usecols=["family"], chunksize=100000)
            families = set()
            for chunk in chunks:
                families.update(chunk["family"].unique())
            
            logger.info(f"Found {len(families)} unique product families")
            
            # Insert families into database
            for family_name in families:
                family_data = {
                    "name": family_name,
                    "description": f"Product family for {family_name} products"
                }
                
                # Check if family already exists
                existing_family = ProductFamilyRepository.get_by_name(db, family_name)
                if not existing_family:
                    ProductFamilyRepository.create(db, family_data)
            
            logger.info("Product families loaded successfully")
        
        else:
            logger.warning(f"Training file not found: {train_path}")
            _create_demo_families(db)
    
    except Exception as e:
        logger.error(f"Error loading product families: {e}")
        # Create demo families as fallback
        _create_demo_families(db)


def _create_demo_families(db: Session) -> None:
    """Create demo product families for testing."""
    logger.info("Creating demo product families")
    
    # List of demo product families
    demo_families = [
        {"name": "GROCERY I", "description": "Basic grocery items"},
        {"name": "BEVERAGES", "description": "Drinks and beverages"},
        {"name": "PRODUCE", "description": "Fresh produce"},
        {"name": "CLEANING", "description": "Cleaning supplies"},
        {"name": "DAIRY", "description": "Dairy products"},
        {"name": "BREAD/BAKERY", "description": "Bread and bakery items"},
        {"name": "POULTRY", "description": "Poultry products"},
        {"name": "MEATS", "description": "Meat products"},
        {"name": "SEAFOOD", "description": "Seafood products"},
        {"name": "PERSONAL CARE", "description": "Personal care items"},
        {"name": "GROCERY II", "description": "Specialty grocery items"},
        {"name": "FROZEN FOODS", "description": "Frozen food items"},
        {"name": "DELI", "description": "Deli products"},
        {"name": "HOME AND KITCHEN", "description": "Home and kitchen supplies"},
        {"name": "PREPARED FOODS", "description": "Ready-to-eat foods"}
    ]
    
    # Insert families into database
    for family_data in demo_families:
        # Check if family already exists
        existing_family = ProductFamilyRepository.get_by_name(db, family_data["name"])
        if not existing_family:
            ProductFamilyRepository.create(db, family_data)
    
    logger.info(f"Created {len(demo_families)} demo product families")


def load_historical_sales(db: Session, days: int = 90) -> None:
    """
    Load historical sales data into the database.
    
    Parameters
    ----------
    db : Session
        Database session.
    days : int, optional (default=90)
        Number of days of historical data to load.
    """
    try:
        # Check if we have a train.csv file
        train_path = RAW_DATA_DIR / "train.csv"
        if train_path.exists():
            logger.info("Loading historical sales from training data")
            
            # Get end date (last date in the dataset)
            # Read a small sample to get date range
            sample_df = pd.read_csv(train_path, nrows=1000)
            sample_df["date"] = pd.to_datetime(sample_df["date"])
            
            # If the sample doesn't have dates, use current date
            if "date" not in sample_df.columns or sample_df["date"].isna().all():
                end_date = datetime.now().date()
            else:
                end_date = sample_df["date"].max().date()
            
            # Calculate start date
            start_date = end_date - timedelta(days=days)
            
            logger.info(f"Loading sales data from {start_date} to {end_date}")
            
            # Read the dataset in chunks
            chunks = pd.read_csv(
                train_path,
                parse_dates=["date"],
                chunksize=100000
            )
            
            total_records = 0
            for chunk in chunks:
                # Filter by date range
                filtered_df = chunk[(chunk["date"].dt.date >= start_date) & 
                                   (chunk["date"].dt.date <= end_date)]
                
                if filtered_df.empty:
                    continue
                
                # Process each row
                sales_data_list = []
                for _, row in filtered_df.iterrows():
                    store_nbr = int(row["store_nbr"])
                    family_name = row["family"]
                    date = row["date"]
                    sales = float(row["sales"])
                    onpromotion = bool(row["onpromotion"]) if "onpromotion" in row else False
                    
                    # Get store and family IDs
                    store = StoreRepository.get_or_create_by_store_nbr(db, store_nbr)
                    family = ProductFamilyRepository.get_or_create_by_name(db, family_name)
                    
                    # Create sales data
                    sales_data = {
                        "store_id": store.id,
                        "family_id": family.id,
                        "date": date,
                        "sales": sales,
                        "onpromotion": onpromotion
                    }
                    
                    sales_data_list.append(sales_data)
                
                # Batch insert
                if sales_data_list:
                    HistoricalSalesRepository.create_many(db, sales_data_list)
                    total_records += len(sales_data_list)
                    logger.info(f"Inserted {len(sales_data_list)} historical sales records")
            
            logger.info(f"Total historical sales records loaded: {total_records}")
        
        else:
            logger.warning(f"Training file not found: {train_path}")
            _create_demo_sales(db, days)
    
    except Exception as e:
        logger.error(f"Error loading historical sales: {e}")
        # Create demo sales as fallback
        _create_demo_sales(db, days)


def _create_demo_sales(db: Session, days: int = 90) -> None:
    """
    Create demo historical sales data for testing.
    
    Parameters
    ----------
    db : Session
        Database session.
    days : int, optional (default=90)
        Number of days of historical data to create.
    """
    logger.info(f"Creating {days} days of demo historical sales data")
    
    # Get all stores and families
    stores = StoreRepository.get_all(db)
    families = ProductFamilyRepository.get_all(db)
    
    if not stores or not families:
        logger.warning("No stores or families found. Loading demo data first.")
        _create_demo_stores(db)
        _create_demo_families(db)
        stores = StoreRepository.get_all(db)
        families = ProductFamilyRepository.get_all(db)
    
    # Generate sales data
    sales_data_list = []
    end_date = datetime.now().date()
    
    # Define seasonality and trend factors for realistic data
    day_of_week_factors = {0: 0.8, 1: 0.9, 2: 1.0, 3: 1.0, 4: 1.2, 5: 1.5, 6: 0.7}  # Mon-Sun
    
    # Generate sales for each day
    for day_offset in range(days, 0, -1):
        current_date = end_date - timedelta(days=day_offset)
        current_datetime = datetime.combine(current_date, datetime.min.time())
        
        # Day of week factor
        day_factor = day_of_week_factors[current_date.weekday()]
        
        # Month factor (seasonal effect)
        month_factor = 1.0 + 0.1 * np.sin(2 * np.pi * current_date.month / 12)
        
        # Process each store and family
        for store in stores:
            # Store factor (different stores have different sales volumes)
            store_factor = 0.5 + 1.5 * store.id / len(stores)
            
            for family in families:
                # Family factor (different products have different sales volumes)
                family_factor = 0.5 + 1.5 * family.id / len(families)
                
                # Random promotion
                onpromotion = random.random() < 0.2
                promotion_factor = 1.3 if onpromotion else 1.0
                
                # Base sales with seasonality, trend, store, and family factors
                base_sales = 50 * day_factor * month_factor * store_factor * family_factor * promotion_factor
                
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
        
        # Batch insert every 10 days
        if len(sales_data_list) >= 1000 or day_offset == 1:
            HistoricalSalesRepository.create_many(db, sales_data_list)
            logger.info(f"Inserted {len(sales_data_list)} demo sales records")
            sales_data_list = []
    
    logger.info("Demo historical sales data created successfully")


def load_initial_data(db: Session) -> None:
    """
    Load initial data into the database.
    
    Parameters
    ----------
    db : Session
        Database session.
    """
    # Check if we already have data
    existing_stores = db.query(Store).count()
    existing_families = db.query(ProductFamily).count()
    existing_sales = db.query(HistoricalSales).count()
    
    if existing_stores > 0 and existing_families > 0 and existing_sales > 0:
        logger.info(f"Database already contains data: {existing_stores} stores, "
                   f"{existing_families} families, {existing_sales} sales records")
        return
    
    logger.info("Loading initial data into the database")
    
    # Load stores
    if existing_stores == 0:
        load_stores(db)
    
    # Load product families
    if existing_families == 0:
        load_product_families(db)
    
    # Load historical sales data
    if existing_sales == 0:
        load_historical_sales(db, days=90)
    
    logger.info("Initial data loaded successfully") 