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
import sentry_sdk

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

# Set up Sentry for error monitoring (opcional)
SENTRY_DSN = os.getenv("SENTRY_DSN")
if SENTRY_DSN:
    sentry_sdk.init(
        dsn=SENTRY_DSN,
        traces_sample_rate=0.1,  # Ajuste conforme necessário
        environment=os.getenv("ENVIRONMENT", "development"),
    )
    logger.info("Sentry initialized for error monitoring")

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


# Endpoint para autenticação
@app.post("/token", response_model=Token)
async def login_for_access_token(form_data: OAuth2PasswordRequestForm = Depends()):
    user = authenticate_user(fake_users_db, form_data.username, form_data.password)
    if not user:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Incorrect username or password",
            headers={"WWW-Authenticate": "Bearer"},
        )
    
    # Criar um token de acesso com os escopos do usuário (interseção entre os solicitados e os atribuídos)
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
    global model
    try:
        model = load_model()
        logger.info("Model loaded successfully")
    except Exception as e:
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
    store_nbr: int = Query(..., description="Store number"),
    family: str = Query(..., description="Product family"),
    onpromotion: bool = Query(..., description="Whether the item is on promotion"),
    date: str = Query(..., description="Prediction date (YYYY-MM-DD)"),
    current_user: User = Security(validate_scopes(["predictions:read"]))
):
    """
    Single prediction endpoint that uses query parameters.
    """
    # Create StoreItem from query parameters
    item = StoreItem(
        store_nbr=store_nbr,
        family=family,
        onpromotion=onpromotion,
        date=date
    )
    
    # Use the batch prediction endpoint
    request = PredictionRequest(items=[item])
    response = await predict(request, current_user)
    
    # Return only the first prediction
    return response.predictions[0]


@app.get("/explain/{prediction_id}")
async def explain_prediction(
    prediction_id: str,
    store_nbr: int = Query(..., description="Store number"),
    family: str = Query(..., description="Product family"),
    onpromotion: bool = Query(..., description="Whether the item is on promotion"),
    date: str = Query(..., description="Prediction date (YYYY-MM-DD)"),
    current_user: User = Security(validate_scopes(["predictions:read"]))
):
    """
    Explicação para uma previsão específica usando SHAP.
    
    Este endpoint gera uma explicação detalhada para uma previsão específica,
    mostrando a contribuição de cada feature para o resultado final.
    """
    global model
    
    # Se o modelo não estiver carregado, tenta carregá-lo
    if model is None:
        try:
            model = load_model()
        except Exception as e:
            raise HTTPException(
                status_code=503,
                detail=f"Model is not available: {str(e)}"
            )
    
    try:
        # Criar StoreItem a partir dos parâmetros
        item = StoreItem(
            store_nbr=store_nbr,
            family=family,
            onpromotion=onpromotion,
            date=date
        )
        
        # Preparar features
        features_df = prepare_features([item])
        
        # Inicializar o explicador de modelos
        explainer = ModelExplainer(model=model)
        
        # Criar o explicador
        explainer.create_explainer()
        
        # Gerar explicação
        explanation = explainer.explain_prediction(features_df.iloc[0])
        
        # Adicionar metadados
        explanation["prediction_id"] = prediction_id
        explanation["item"] = {
            "store_nbr": store_nbr,
            "family": family,
            "date": date,
            "onpromotion": onpromotion
        }
        
        return explanation
    
    except Exception as e:
        logger.error(f"Error generating explanation: {e}")
        raise HTTPException(
            status_code=500,
            detail=f"Error generating explanation: {str(e)}"
        )


@app.get("/users/me", response_model=User)
async def read_users_me(current_user: User = Depends(get_current_active_user)):
    """
    Return information about the current authenticated user.
    """
    return current_user


if __name__ == "__main__":
    # Start the API server
    port = int(os.getenv("PORT", 8000))
    host = os.getenv("HOST", "0.0.0.0")
    
    logger.info(f"Starting API server on {host}:{port}")
    uvicorn.run(app, host=host, port=port)