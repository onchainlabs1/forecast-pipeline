#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
FastAPI application for serving sales forecasting predictions.
"""

import os
import sys
import logging
from pathlib import Path
from typing import List, Dict, Any, Optional
from datetime import datetime

import pandas as pd
import numpy as np
import joblib
from fastapi import FastAPI, HTTPException, Query
from pydantic import BaseModel, Field
import mlflow
import uvicorn

# Add project root to sys.path
project_root = Path(__file__).parents[2]
sys.path.insert(0, str(project_root))

# Import utilities
from src.utils.mlflow_utils import setup_mlflow

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)

# Define paths
PROJECT_DIR = Path(__file__).resolve().parents[2]
MODELS_DIR = PROJECT_DIR / "models"
MODEL_PATH = MODELS_DIR / "lightgbm_model.pkl"

# FastAPI app
app = FastAPI(
    title="Store Sales Forecasting API",
    description="API for predicting store sales for Favorita stores",
    version="1.0.0",
)

# Pydantic models for request and response
class StoreItem(BaseModel):
    store_nbr: int = Field(..., description="Store number")
    family: str = Field(..., description="Product family")
    onpromotion: bool = Field(..., description="Whether the item is on promotion")
    date: str = Field(..., description="Prediction date (YYYY-MM-DD)")
    
    class Config:
        schema_extra = {
            "example": {
                "store_nbr": 1,
                "family": "GROCERY I",
                "onpromotion": True,
                "date": "2017-08-16"
            }
        }


class PredictionRequest(BaseModel):
    items: List[StoreItem] = Field(..., description="List of items to predict sales for")


class PredictionItem(BaseModel):
    store_nbr: int
    family: str
    date: str
    predicted_sales: float


class PredictionResponse(BaseModel):
    predictions: List[PredictionItem]
    model_version: str
    prediction_time: str


# Model loading
def load_model():
    """
    Load the trained model.
    
    Returns
    -------
    model : object
        The trained model.
    """
    try:
        # Check if model file exists
        if MODEL_PATH.exists():
            logger.info(f"Loading model from {MODEL_PATH}")
            return joblib.load(MODEL_PATH)
        
        # If model file doesn't exist, try loading from MLflow
        logger.info("Model file not found, loading from MLflow")
        setup_mlflow()
        
        try:
            model = mlflow.pyfunc.load_model(
                model_uri="models:/store-sales-forecaster/Production"
            )
            logger.info("Model loaded from MLflow registry")
            return model
        except Exception as e:
            logger.error(f"Error loading model from MLflow: {e}")
            raise FileNotFoundError("Model not found in local file or MLflow registry")
    
    except Exception as e:
        logger.error(f"Error loading model: {e}")
        raise e


# Prepare features for prediction
def prepare_features(items: List[StoreItem]) -> pd.DataFrame:
    """
    Prepare features for prediction.
    
    Parameters
    ----------
    items : List[StoreItem]
        List of items to predict sales for.
        
    Returns
    -------
    pd.DataFrame
        DataFrame with prepared features.
    """
    try:
        # Convert items to dataframe
        df = pd.DataFrame([item.dict() for item in items])
        
        # Convert date to datetime
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
        
        # Convert onpromotion to int
        df["onpromotion"] = df["onpromotion"].astype(int)
        
        # One-hot encode family
        df = pd.get_dummies(df, columns=["family"], prefix=["family"])
        
        # For demonstration, add placeholder values for missing features
        # In a real scenario, you would load the same preprocessing pipeline used during training
        # or use the full feature engineering pipeline
        
        # Example lag features (placeholder values)
        for lag in [1, 7, 14, 28]:
            df[f"sales_lag_{lag}"] = 0
        
        # Example rolling features (placeholder values)
        for window in [7, 14, 30]:
            df[f"sales_roll_{window}_mean"] = 0
            df[f"sales_roll_{window}_std"] = 0
            df[f"sales_roll_{window}_max"] = 0
            df[f"sales_roll_{window}_min"] = 0
        
        # Example target encoding features (placeholder values)
        df["store_nbr_sales_mean"] = 0
        
        # Important: Remove date column before prediction
        df = df.drop("date", axis=1)
        
        return df
    
    except Exception as e:
        logger.error(f"Error preparing features: {e}")
        raise e


# Load model at startup
model = None

@app.on_event("startup")
async def startup_event():
    global model
    try:
        model = load_model()
        logger.info("Model loaded successfully")
    except Exception as e:
        logger.error(f"Error loading model at startup: {e}")
        # We'll retry loading the model on the first request


@app.get("/")
async def root():
    """Root endpoint with API information."""
    return {
        "message": "Store Sales Forecasting API",
        "version": "1.0.0",
        "docs_url": "/docs",
    }


@app.get("/health")
async def health_check():
    """Health check endpoint."""
    global model
    
    if model is None:
        try:
            model = load_model()
            return {"status": "healthy", "model_loaded": True}
        except Exception as e:
            return {"status": "unhealthy", "model_loaded": False, "error": str(e)}
    
    return {"status": "healthy", "model_loaded": True}


@app.post("/predict", response_model=PredictionResponse)
async def predict(request: PredictionRequest):
    """
    Predict sales for a list of store items.
    
    Parameters
    ----------
    request : PredictionRequest
        Request containing list of items to predict sales for.
        
    Returns
    -------
    PredictionResponse
        Response containing sales predictions.
    """
    global model
    
    # Load model if not already loaded
    if model is None:
        try:
            model = load_model()
        except Exception as e:
            raise HTTPException(
                status_code=500,
                detail=f"Model could not be loaded: {str(e)}"
            )
    
    try:
        # Prepare features
        features = prepare_features(request.items)
        
        # Make predictions
        predictions = model.predict(features)
        
        # Ensure predictions are not negative
        predictions = np.maximum(predictions, 0)
        
        # Create response
        prediction_items = []
        for i, item in enumerate(request.items):
            prediction_items.append(
                PredictionItem(
                    store_nbr=item.store_nbr,
                    family=item.family,
                    date=item.date,
                    predicted_sales=float(predictions[i])
                )
            )
        
        return PredictionResponse(
            predictions=prediction_items,
            model_version="1.0.0",  # In production, get this from MLflow
            prediction_time=datetime.now().isoformat()
        )
    
    except Exception as e:
        logger.error(f"Error making predictions: {e}")
        raise HTTPException(
            status_code=500,
            detail=f"Error making predictions: {str(e)}"
        )


@app.get("/predict_single")
async def predict_single(
    store_nbr: int = Query(..., description="Store number"),
    family: str = Query(..., description="Product family"),
    onpromotion: bool = Query(..., description="Whether the item is on promotion"),
    date: str = Query(..., description="Prediction date (YYYY-MM-DD)")
):
    """
    Predict sales for a single store item using query parameters.
    
    Parameters
    ----------
    store_nbr : int
        Store number
    family : str
        Product family
    onpromotion : bool
        Whether the item is on promotion
    date : str
        Prediction date (YYYY-MM-DD)
        
    Returns
    -------
    dict
        Dictionary containing the prediction.
    """
    # Create a request with a single item
    request = PredictionRequest(
        items=[
            StoreItem(
                store_nbr=store_nbr,
                family=family,
                onpromotion=onpromotion,
                date=date
            )
        ]
    )
    
    # Use the predict endpoint
    response = await predict(request)
    
    # Return a simplified response
    return {
        "store_nbr": store_nbr,
        "family": family,
        "date": date,
        "predicted_sales": response.predictions[0].predicted_sales,
        "model_version": response.model_version,
        "prediction_time": response.prediction_time
    }


if __name__ == "__main__":
    # Run the API with uvicorn
    uvicorn.run(
        "main:app",
        host="0.0.0.0",
        port=8000,
        reload=True,
        log_level="info"
    )