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
from datetime import datetime, timedelta

import pandas as pd
import numpy as np
import joblib
from fastapi import FastAPI, HTTPException, Query, Depends, Security, status
from fastapi.security import OAuth2PasswordRequestForm
from pydantic import BaseModel, Field
import mlflow
import uvicorn
from sqlalchemy.orm import Session
from sqlalchemy import func, distinct

# Add project root to sys.path
project_root = Path(__file__).parents[2]
sys.path.insert(0, str(project_root))

# Import utilities
from src.utils.mlflow_utils import setup_mlflow
from src.security.auth import (
    Token, User, authenticate_user, create_access_token, 
    get_current_active_user, validate_scopes, fake_users_db,
    ACCESS_TOKEN_EXPIRE_MINUTES
)
from src.models.explanation import ModelExplainer
from src.database.database import get_db
from src.database.repository import (
    StoreRepository, ProductFamilyRepository, HistoricalSalesRepository,
    PredictionRepository, ModelMetricRepository, FeatureImportanceRepository,
    ModelDriftRepository
)
from src.database.models import Store, ProductFamily, HistoricalSales, Prediction, ModelMetric, FeatureImportance, ModelDrift

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

# Set up Sentry for error monitoring (optional)
SENTRY_DSN = os.getenv("SENTRY_DSN")
try:
    import sentry_sdk
    if SENTRY_DSN:
        sentry_sdk.init(
            dsn=SENTRY_DSN,
            traces_sample_rate=0.1,
            environment=os.getenv("ENVIRONMENT", "development"),
        )
        logger.info("Sentry initialized for error monitoring")
except ImportError:
    logger.warning("Sentry SDK not installed. Error monitoring disabled.")

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


# Authentication endpoint
@app.post("/token", response_model=Token)
async def login_for_access_token(form_data: OAuth2PasswordRequestForm = Depends()):
    user = authenticate_user(fake_users_db, form_data.username, form_data.password)
    if not user:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Incorrect username or password",
            headers={"WWW-Authenticate": "Bearer"},
        )
    
    # Create an access token with the user's scopes (intersection between requested and assigned)
    access_token_expires = timedelta(minutes=ACCESS_TOKEN_EXPIRE_MINUTES)
    scopes = [scope for scope in form_data.scopes if scope in user.scopes]
    
    access_token = create_access_token(
        data={"sub": user.username, "scopes": scopes},
        expires_delta=access_token_expires,
    )
    
    return {"access_token": access_token, "token_type": "bearer"}


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
    """
    Startup event to load model and setup MLflow.
    """
    global model
    
    logger.info("API starting up")
    try:
        # Initialize MLflow
        setup_mlflow()
        
        # Load model
        model = load_model()
        logger.info("Model loaded successfully")
    except Exception as e:
        logger.error(f"Error during startup: {e}")
        model = None
        logger.error(f"Error loading model at startup: {e}")
        # We'll retry loading the model on the first request


@app.get("/")
async def root():
    """
    Root endpoint that provides basic information about the API.
    """
    return {
        "message": "Store Sales Forecasting API",
        "docs_url": "/docs",
        "version": "1.0.0"
    }


@app.get("/health")
async def health_check():
    """
    Health check endpoint for monitoring systems.
    """
    global model
    
    # If model is not loaded, try to load it
    if model is None:
        try:
            model = load_model()
        except Exception as e:
            return {
                "status": "error",
                "message": f"Model is not loaded: {str(e)}",
                "timestamp": datetime.now().isoformat()
            }
    
    return {
        "status": "healthy",
        "model_loaded": model is not None,
        "timestamp": datetime.now().isoformat()
    }


@app.post("/predict", response_model=PredictionResponse)
async def predict(
    request: PredictionRequest,
    current_user: User = Security(validate_scopes(["predictions:read"]))
):
    """
    Batch prediction endpoint.
    
    Parameters
    ----------
    request : PredictionRequest
        Request with a list of items to predict sales for.
    """
    global model
    
    # If model is not loaded, try to load it
    if model is None:
        try:
            model = load_model()
        except Exception as e:
            raise HTTPException(
                status_code=503,
                detail=f"Model is not available: {str(e)}"
            )
    
    try:
        # Prepare features
        features = prepare_features(request.items)
        
        # Make predictions
        predictions = model.predict(features)
        
        # Prepare response
        result = []
        for i, item in enumerate(request.items):
            result.append(
                PredictionItem(
                    store_nbr=item.store_nbr,
                    family=item.family,
                    date=item.date,
                    predicted_sales=float(predictions[i])
                )
            )
        
        # Create response
        response = PredictionResponse(
            predictions=result,
            model_version=getattr(model, "version", "unknown"),
            prediction_time=datetime.now().isoformat()
        )
        
        return response
    
    except Exception as e:
        logger.error(f"Error making predictions: {e}")
        raise HTTPException(
            status_code=500,
            detail=f"Error making predictions: {str(e)}"
        )


@app.get("/predict_single")
async def predict_single(
    store_nbr: int,
    family: str,
    onpromotion: bool,
    date: str,
    current_user: User = Depends(get_current_active_user)
):
    """
    Get a single prediction for a specific store, family, promotion status, and date.
    """
    try:
        # Convert date string to datetime
        prediction_date = datetime.strptime(date, "%Y-%m-%d")
        
        # Generate features for this prediction
        features = generate_features(store_nbr, family, onpromotion, prediction_date)
        
        # Make prediction
        try:
            prediction = model.predict([features])[0]
            
            # Generate unique ID for this prediction
            prediction_id = f"{store_nbr}-{family}-{prediction_date.strftime('%Y-%m-%d')}"
            
            # Save prediction to database (if available)
            save_prediction(store_nbr, family, prediction_date, prediction, current_user.username)
            
            # Return result
            return {
                "prediction": float(prediction),
                "prediction_id": prediction_id,
                "store_nbr": store_nbr,
                "family": family,
                "date": date,
                "onpromotion": onpromotion
            }
        except Exception as model_error:
            if "number of features in data" in str(model_error):
                # Feature dimension mismatch
                logger.error(f"Feature dimension mismatch. Model expects different number of features than generated.")
                return {
                    "prediction": generate_fallback_prediction(store_nbr, family),
                    "prediction_id": f"{store_nbr}-{family}-{prediction_date.strftime('%Y-%m-%d')}",
                    "store_nbr": store_nbr,
                    "family": family,
                    "date": date,
                    "onpromotion": onpromotion,
                    "is_fallback": True
                }
            else:
                # Other model error
                raise model_error
    except Exception as e:
        logger.error(f"Error making predictions: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Error making predictions: {str(e)}")


def generate_fallback_prediction(store_nbr, family):
    """Generate a fallback prediction when the model fails."""
    # Simple fallback logic - use average of historical data or a constant
    # In a production system, this should be more sophisticated
    try:
        # Try to get average sales for this store/family
        query = """
        SELECT AVG(sales) as avg_sales FROM historical_sales 
        WHERE store_nbr = :store_nbr AND family = :family
        """
        result = db.execute(query, {"store_nbr": store_nbr, "family": family}).fetchone()
        if result and result["avg_sales"]:
            return float(result["avg_sales"])
        else:
            # If no historical data, return a default value
            return 10.0  # Default fallback value
    except:
        # If database query fails, return a default value
        return 10.0  # Default fallback value


@app.get("/explain/{prediction_id}")
async def explain_prediction(
    prediction_id: str,
    store_nbr: int,
    family: str,
    onpromotion: bool,
    date: str,
    current_user: User = Depends(get_current_active_user)
):
    """
    Get explanation for a prediction.
    """
    try:
        # Convert date string to datetime
        prediction_date = datetime.strptime(date, "%Y-%m-%d")
        
        # Generate features for this prediction
        features = generate_features(store_nbr, family, onpromotion, prediction_date)
        
        # Get feature names
        feature_names = get_feature_names()
        
        try:
            # Generate SHAP values
            from src.models.explanation import generate_explanation
            
            explanation = generate_explanation(model, features, feature_names)
            
            if explanation:
                return explanation
            else:
                # Handle the case where explanation could not be generated
                logger.warning(f"Could not generate explanation for prediction {prediction_id}")
                return {
                    "message": "Explanation not available for this model type",
                    "feature_contributions": [
                        {"feature": "store_nbr", "value": store_nbr, "contribution": 0.0},
                        {"feature": "family", "value": family, "contribution": 0.0},
                        {"feature": "onpromotion", "value": onpromotion, "contribution": 0.0},
                        {"feature": "date", "value": date, "contribution": 0.0}
                    ]
                }
        except Exception as exp_error:
            logger.error(f"Error generating explanation: {str(exp_error)}")
            # Return a simplified explanation with just the input values
            return {
                "message": f"Error generating explanation: {str(exp_error)}",
                "feature_contributions": [
                    {"feature": "store_nbr", "value": store_nbr, "contribution": 0.0},
                    {"feature": "family", "value": family, "contribution": 0.0},
                    {"feature": "onpromotion", "value": onpromotion, "contribution": 0.0},
                    {"feature": "date", "value": date, "contribution": 0.0}
                ]
            }
    except Exception as e:
        logger.error(f"Error generating explanation: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Error generating explanation: {str(e)}")


@app.get("/users/me", response_model=User)
async def read_users_me(current_user: User = Depends(get_current_active_user)):
    """
    Return information about the current authenticated user.
    """
    return current_user


@app.get("/metrics_summary")
async def get_metrics_summary(
    current_user: User = Security(validate_scopes(["predictions:read"])),
    db: Session = Depends(get_db)
):
    """
    Get summary metrics for the dashboard.
    """
    try:
        # Get counts from database
        total_stores = db.query(func.count(distinct(Store.id))).scalar() or 0
        total_families = db.query(func.count(distinct(ProductFamily.id))).scalar() or 0
        
        # Calculate average sales
        avg_sales_result = db.query(func.avg(HistoricalSales.sales)).scalar()
        avg_sales = float(avg_sales_result) if avg_sales_result is not None else 0
        
        # Get forecast accuracy from model metrics
        # For now, use a placeholder value
        forecast_accuracy = 87.5
        
        # Could be enhanced to get real forecast accuracy from ModelMetricsRepository
        # metrics = ModelMetricRepository.get_latest_metrics(db)
        # forecast_accuracy = metrics.accuracy * 100 if metrics else 85.0
        
        return {
            "total_stores": total_stores,
            "total_families": total_families,
            "avg_sales": avg_sales,
            "forecast_accuracy": forecast_accuracy
        }
    except Exception as e:
        logger.error(f"Error getting metrics summary: {e}")
        raise HTTPException(
            status_code=500,
            detail=f"Error getting metrics summary: {str(e)}"
        )


@app.get("/stores")
async def get_stores(db: Session = Depends(get_db)):
    """
    Get list of stores.
    """
    try:
        # Get stores from database
        stores = StoreRepository.get_all(db)
        
        # Convert to store numbers (integers)
        store_numbers = [store.store_nbr for store in stores]
        
        return sorted(store_numbers)
    except Exception as e:
        logger.error(f"Error getting stores: {e}")
        raise HTTPException(
            status_code=500,
            detail=f"Error getting stores: {str(e)}"
        )


@app.get("/families")
async def get_families(db: Session = Depends(get_db)):
    """
    Get list of product families.
    """
    try:
        # Get product families from database
        families = ProductFamilyRepository.get_all(db)
        
        # Extract family names
        family_names = [family.name for family in families]
        
        return sorted(family_names)
    except Exception as e:
        logger.error(f"Error getting product families: {e}")
        raise HTTPException(
            status_code=500,
            detail=f"Error getting product families: {str(e)}"
        )


@app.get("/sales_history")
async def get_sales_history(
    store_nbr: int = Query(..., description="Store number"),
    family: str = Query(..., description="Product family"),
    days: int = Query(90, description="Number of days of history to return"),
    current_user: User = Security(validate_scopes(["predictions:read"])),
    db: Session = Depends(get_db)
):
    """
    Get historical sales data for a specific store and product family.
    """
    try:
        # Get store and family IDs
        store = StoreRepository.get_by_store_nbr(db, store_nbr)
        if not store:
            raise HTTPException(status_code=404, detail=f"Store {store_nbr} not found")
        
        family_obj = ProductFamilyRepository.get_by_name(db, family)
        if not family_obj:
            raise HTTPException(status_code=404, detail=f"Product family {family} not found")
        
        # Get sales history from database
        sales_history = HistoricalSalesRepository.get_sales_history(
            db, store.id, family_obj.id, days=days
        )
        
        # Format the response
        result = []
        for record in sales_history:
            result.append({
                "date": record.date.strftime("%Y-%m-%d"),
                "sales": float(record.sales),
                "is_promotion": 1 if record.onpromotion else 0
            })
        
        # If no data found, log a warning
        if not result:
            logger.warning(f"No historical sales data found for store {store_nbr}, family {family}")
            
        return result
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error getting sales history: {e}")
        raise HTTPException(
            status_code=500,
            detail=f"Error getting sales history: {str(e)}"
        )


@app.get("/store_comparison")
async def get_store_comparison(
    days: int = Query(30, description="Number of days to analyze"),
    current_user: User = Security(validate_scopes(["predictions:read"])),
    db: Session = Depends(get_db)
):
    """
    Get comparison data for stores.
    """
    try:
        # Get store performance data from database
        stores_df = HistoricalSalesRepository.get_store_comparison(db, days=days)
        
        # Get metrics data for forecast accuracy
        # In a real app, we would get actual forecast accuracy from ModelMetricRepository
        
        result = []
        if not stores_df.empty:
            # Take top 10 stores by sales
            top_stores = stores_df.nlargest(10, 'total_sales')
            
            # Convert DataFrame to list of dictionaries
            for _, row in top_stores.iterrows():
                # Generate random forecast accuracy between 70% and 95%
                forecast_accuracy = np.random.uniform(0.7, 0.95)
                
                result.append({
                    "store": f"Store {int(row['store_nbr'])}",
                    "sales": float(row["total_sales"]),
                    "forecast_accuracy": float(forecast_accuracy)
                })
        
        if not result:
            logger.warning(f"No store comparison data found for the last {days} days")
        
        return result
    except Exception as e:
        logger.error(f"Error getting store comparison: {e}")
        raise HTTPException(
            status_code=500,
            detail=f"Error getting store comparison: {str(e)}"
        )


@app.get("/family_performance")
async def get_family_performance(
    days: int = Query(30, description="Number of days to analyze"),
    current_user: User = Security(validate_scopes(["predictions:read"])),
    db: Session = Depends(get_db)
):
    """
    Get performance data for product families.
    """
    try:
        # Get family performance data from database
        performance_df = HistoricalSalesRepository.get_family_performance(db, days=days)
        
        # Calculate growth rates (comparing first half to second half of the period)
        # A more sophisticated approach would be to use timeseries analysis
        result = []
        
        if not performance_df.empty:
            # Convert DataFrame to list of dictionaries
            for _, row in performance_df.iterrows():
                result.append({
                    "family": row["family"],
                    "sales": float(row["total_sales"]),
                    # Random growth value between -0.2 and +0.3 for demonstration
                    # In a real application, you would calculate actual growth
                    "growth": float(np.random.uniform(-0.2, 0.3))
                })
        
        if not result:
            logger.warning(f"No family performance data found for the last {days} days")
        
        return result
    except Exception as e:
        logger.error(f"Error getting family performance: {e}")
        raise HTTPException(
            status_code=500,
            detail=f"Error getting family performance: {str(e)}"
        )


@app.get("/historical_sales")
async def get_historical_sales(
    store_nbr: int = Query(..., description="Store number"),
    family: str = Query(..., description="Product family"),
    days: int = Query(60, description="Number of days of history to return"),
    current_user: User = Security(validate_scopes(["predictions:read"])),
    db: Session = Depends(get_db)
):
    """
    Get historical sales data for a specific store and product family.
    """
    try:
        # Get store and family IDs
        store = StoreRepository.get_by_store_nbr(db, store_nbr)
        if not store:
            raise HTTPException(status_code=404, detail=f"Store {store_nbr} not found")
        
        family_obj = ProductFamilyRepository.get_by_name(db, family)
        if not family_obj:
            raise HTTPException(status_code=404, detail=f"Product family {family} not found")
        
        # Get sales history from database
        sales_history = HistoricalSalesRepository.get_sales_history(
            db, store.id, family_obj.id, days=days
        )
        
        # Format the response
        result = []
        for record in sales_history:
            result.append({
                "date": record.date.strftime("%Y-%m-%d"),
                "sales": float(record.sales)
            })
        
        # If no data found, log a warning
        if not result:
            logger.warning(f"No historical sales data found for store {store_nbr}, family {family}")
            
        return result
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error getting historical sales: {e}")
        raise HTTPException(
            status_code=500,
            detail=f"Error getting historical sales: {str(e)}"
        )


@app.get("/models")
async def get_models(current_user: User = Security(validate_scopes(["predictions:read"]))):
    """
    Get list of available models.
    """
    try:
        # In a real implementation, you would fetch models from MLflow or a database
        # For now, we'll return a hardcoded list
        return [
            {"name": "LightGBM (Production)", "status": "active", "version": "1.0.0"},
            {"name": "Prophet (Staging)", "status": "staging", "version": "0.9.0"},
            {"name": "ARIMA (Development)", "status": "development", "version": "0.5.0"}
        ]
    except Exception as e:
        logger.error(f"Error getting models: {e}")
        raise HTTPException(
            status_code=500,
            detail=f"Error getting models: {str(e)}"
        )


@app.get("/model_metrics")
async def get_model_metrics(
    model_name: str = Query(..., description="Model name"),
    current_user: User = Security(validate_scopes(["predictions:read"]))
):
    """
    Get performance metrics for a specific model.
    """
    try:
        # In a real implementation, you would fetch metrics from MLflow or a database
        # For now, we'll return hardcoded metrics based on the model name
        metrics = {}
        
        if model_name == "LightGBM (Production)":
            metrics = {
                "rmse": 245.32,
                "mae": 187.44,
                "mape": 14.3,
                "r2": 0.87,
                "rmse_change": "-12.5%",
                "mae_change": "-8.2%",
                "mape_change": "-5.1%",
                "r2_change": "+0.04"
            }
        elif model_name == "Prophet (Staging)":
            metrics = {
                "rmse": 267.89,
                "mae": 201.35,
                "mape": 16.2,
                "r2": 0.82,
                "rmse_change": "+9.2%",
                "mae_change": "+7.4%",
                "mape_change": "+13.3%",
                "r2_change": "-0.06"
            }
        else:  # ARIMA or any other model
            metrics = {
                "rmse": 295.67,
                "mae": 234.12,
                "mape": 18.7,
                "r2": 0.75,
                "rmse_change": "+20.5%",
                "mae_change": "+24.9%",
                "mape_change": "+30.8%",
                "r2_change": "-0.14"
            }
        
        return metrics
    except Exception as e:
        logger.error(f"Error getting model metrics: {e}")
        raise HTTPException(
            status_code=500,
            detail=f"Error getting model metrics: {str(e)}"
        )


@app.get("/feature_importance")
async def get_feature_importance(
    model_name: str = Query(..., description="Model name"),
    current_user: User = Security(validate_scopes(["predictions:read"]))
):
    """
    Get feature importance for a specific model.
    """
    try:
        # In a real implementation, you would fetch feature importance from the model
        # For now, we'll return hardcoded values
        features = [
            "onpromotion", "day_of_week", "store_nbr", "month", 
            "day_of_month", "is_weekend", "family_GROCERY I", 
            "family_BEVERAGES", "family_PRODUCE", "family_CLEANING"
        ]
        
        # Generate random importance values
        np.random.seed(hash(model_name) % 1000)  # Use model name as seed for consistent results
        importance = np.random.uniform(0.01, 0.25, size=len(features))
        importance = importance / importance.sum()
        
        # Create result
        result = []
        for i, feature in enumerate(features):
            result.append({
                "feature": feature,
                "importance": float(importance[i])
            })
        
        return result
    except Exception as e:
        logger.error(f"Error getting feature importance: {e}")
        raise HTTPException(
            status_code=500,
            detail=f"Error getting feature importance: {str(e)}"
        )


@app.get("/model_drift")
async def get_model_drift(
    model_name: str = Query(..., description="Model name"),
    days: int = Query(7, description="Number of days to analyze"),
    current_user: User = Security(validate_scopes(["predictions:read"])),
    db: Session = Depends(get_db)
):
    """
    Get model drift information.
    """
    try:
        # Get model drift records
        drift_data = ModelDriftRepository.get_recent_drift(db, model_name, days)
        
        # Guard against empty data
        if not drift_data or len(drift_data) == 0:
            return {
                "dates": [],
                "rmse": [],
                "mae": [],
                "drift_score": []
            }
        
        # Ensure we don't try to access more days than we have data for
        available_days = min(days, len(drift_data))
        
        # Create response with available data only
        dates = [d.date.strftime("%Y-%m-%d") for d in drift_data[:available_days]]
        rmse = [float(d.rmse) for d in drift_data[:available_days]]
        mae = [float(d.mae) for d in drift_data[:available_days]]
        drift_score = [float(d.drift_detected * 100) for d in drift_data[:available_days]]
        
        return {
            "dates": dates,
            "rmse": rmse,
            "mae": mae,
            "drift_score": drift_score
        }
        
    except Exception as e:
        logger.error(f"Error getting model drift: {e}")
        raise HTTPException(
            status_code=500,
            detail=f"Error getting model drift: {str(e)}"
        )


def generate_features_for_prediction(store_nbr, family, onpromotion, date):
    """
    Generate features for a single prediction.
    
    Parameters
    ----------
    store_nbr : int
        Store number
    family : str
        Product family
    onpromotion : bool
        Whether the item is on promotion
    date : datetime.date
        Date for prediction
    
    Returns
    -------
    X : pandas.DataFrame
        DataFrame with features for prediction
    """
    # Create a dataframe with a single row
    df = pd.DataFrame({
        'store_nbr': [store_nbr],
        'family': [family],
        'onpromotion': [int(onpromotion)],
        'date': [date]
    })
    
    # Ensure date is in datetime format
    if not pd.api.types.is_datetime64_any_dtype(df['date']):
        if isinstance(date, str):
            df['date'] = pd.to_datetime(df['date'])
        # If it's already a datetime.date object, convert to datetime
        elif isinstance(date, datetime.date) and not isinstance(date, datetime.datetime):
            df['date'] = pd.to_datetime(df['date'])
    
    # Add time-based features
    df['year'] = df['date'].dt.year
    df['month'] = df['date'].dt.month
    df['day'] = df['date'].dt.day
    df['dayofweek'] = df['date'].dt.dayofweek
    df['dayofyear'] = df['date'].dt.dayofyear
    df['quarter'] = df['date'].dt.quarter
    df['is_weekend'] = df['dayofweek'].apply(lambda x: 1 if x >= 5 else 0)
    
    # One-hot encode categorical variables
    # Store
    for i in range(1, 55):  # Assuming 54 stores based on the error message
        df[f'store_{i}'] = (df['store_nbr'] == i).astype(int)
    
    # Family - using the most common families
    families = [
        'AUTOMOTIVE', 'BABY CARE', 'BEAUTY', 'BEVERAGES', 'BOOKS', 
        'BREAD/BAKERY', 'CELEBRATION', 'CLEANING', 'DAIRY', 'DELI', 
        'EGGS', 'FROZEN FOODS', 'GROCERY I', 'GROCERY II', 'HARDWARE',
        'HOME AND KITCHEN', 'HOME APPLIANCES', 'HOME CARE', 'LADIESWEAR',
        'LAWN AND GARDEN', 'LINGERIE', 'LIQUOR,WINE,BEER', 'MAGAZINES',
        'MEATS', 'PERSONAL CARE', 'PET SUPPLIES', 'PLAYERS AND ELECTRONICS',
        'POULTRY', 'PREPARED FOODS', 'PRODUCE', 'SCHOOL AND OFFICE SUPPLIES',
        'SEAFOOD'
    ]
    
    for f in families:
        df[f'family_{f.replace(" ", "_")}'] = (df['family'] == f).astype(int)
    
    # Drop original columns
    df = df.drop(['store_nbr', 'family', 'date'], axis=1)
    
    # Ensure all required features are present with default value 0
    # Get the expected features from the model
    if hasattr(model, 'feature_names_'):
        expected_features = model.feature_names_
        
        # Create a DataFrame with all expected features set to 0
        X = pd.DataFrame(0, index=df.index, columns=expected_features)
        
        # Update with the values we have
        for col in df.columns:
            if col in X.columns:
                X[col] = df[col]
    else:
        # If the model doesn't expose feature names, just return what we have
        X = df
    
    return X


def generate_features(store_nbr, family, onpromotion, date):
    """
    Generate features for a single prediction.
    
    This function is a wrapper around generate_features_for_prediction to ensure
    compatibility with the updated code.
    
    Parameters
    ----------
    store_nbr : int
        Store number
    family : str
        Product family
    onpromotion : bool
        Whether the item is on promotion
    date : datetime
        Date for prediction
    
    Returns
    -------
    numpy.ndarray
        Array with features for prediction
    """
    try:
        # Generate features using the existing function
        X = generate_features_for_prediction(store_nbr, family, onpromotion, date)
        
        # Return as numpy array (first row)
        if isinstance(X, pd.DataFrame):
            return X.values[0]
        else:
            return X[0]
    except Exception as e:
        logger.error(f"Error generating features: {e}")
        # Return a minimal feature set (this will likely trigger the fallback prediction)
        return np.zeros(28)  # Minimal feature set


def get_feature_names():
    """
    Get feature names for the current model.
    
    Returns
    -------
    list
        List of feature names
    """
    # Basic time features
    features = [
        'onpromotion', 'year', 'month', 'day', 'dayofweek',
        'dayofyear', 'quarter', 'is_weekend'
    ]
    
    # Store features
    for i in range(1, 55):  # Assuming 54 stores
        features.append(f'store_{i}')
    
    # Family features
    families = [
        'AUTOMOTIVE', 'BABY CARE', 'BEAUTY', 'BEVERAGES', 'BOOKS', 
        'BREAD/BAKERY', 'CELEBRATION', 'CLEANING', 'DAIRY', 'DELI', 
        'EGGS', 'FROZEN FOODS', 'GROCERY I', 'GROCERY II', 'HARDWARE',
        'HOME AND KITCHEN', 'HOME APPLIANCES', 'HOME CARE', 'LADIESWEAR',
        'LAWN AND GARDEN', 'LINGERIE', 'LIQUOR,WINE,BEER', 'MAGAZINES',
        'MEATS', 'PERSONAL CARE', 'PET SUPPLIES', 'PLAYERS AND ELECTRONICS',
        'POULTRY', 'PREPARED FOODS', 'PRODUCE', 'SCHOOL AND OFFICE SUPPLIES',
        'SEAFOOD'
    ]
    
    for f in families:
        features.append(f'family_{f.replace(" ", "_")}')
        
    return features


if __name__ == "__main__":
    # Start the API server
    port = int(os.getenv("PORT", 8000))
    host = os.getenv("HOST", "0.0.0.0")
    
    logger.info(f"Starting API server on {host}:{port}")
    uvicorn.run(app, host=host, port=port)