#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
SQLAlchemy models for the sales forecasting database.
"""

import uuid
from datetime import datetime
from typing import List, Optional

from sqlalchemy import Column, Integer, Float, String, Boolean, DateTime, ForeignKey, Text, Index, JSON
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import relationship

Base = declarative_base()


class Store(Base):
    """Store model representing a physical store location."""
    
    __tablename__ = "stores"
    
    id = Column(Integer, primary_key=True, autoincrement=True)
    store_nbr = Column(Integer, nullable=False, unique=True, index=True)
    city = Column(String(100), nullable=True)
    state = Column(String(100), nullable=True)
    type = Column(String(50), nullable=True)
    cluster = Column(Integer, nullable=True)
    created_at = Column(DateTime, default=datetime.utcnow)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)
    
    # Relationships
    predictions = relationship("Prediction", back_populates="store")
    historical_sales = relationship("HistoricalSales", back_populates="store")
    
    def __repr__(self):
        return f"<Store(store_nbr={self.store_nbr}, city={self.city}, state={self.state})>"


class ProductFamily(Base):
    """Product family model representing a category of products."""
    
    __tablename__ = "product_families"
    
    id = Column(Integer, primary_key=True, autoincrement=True)
    name = Column(String(100), nullable=False, unique=True, index=True)
    description = Column(Text, nullable=True)
    created_at = Column(DateTime, default=datetime.utcnow)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)
    
    # Relationships
    predictions = relationship("Prediction", back_populates="family")
    historical_sales = relationship("HistoricalSales", back_populates="family")
    
    def __repr__(self):
        return f"<ProductFamily(name={self.name})>"


class Prediction(Base):
    """Prediction model for storing model predictions."""
    
    __tablename__ = "predictions"
    
    id = Column(String(36), primary_key=True, default=lambda: str(uuid.uuid4()))
    store_id = Column(Integer, ForeignKey("stores.id"), nullable=False)
    family_id = Column(Integer, ForeignKey("product_families.id"), nullable=False)
    prediction_date = Column(DateTime, nullable=False)
    target_date = Column(DateTime, nullable=False)
    onpromotion = Column(Boolean, default=False)
    predicted_sales = Column(Float, nullable=False)
    prediction_interval_lower = Column(Float, nullable=True)
    prediction_interval_upper = Column(Float, nullable=True)
    model_version = Column(String(50), nullable=False)
    feature_values = Column(JSON, nullable=True)
    created_at = Column(DateTime, default=datetime.utcnow)
    
    # Relationships
    store = relationship("Store", back_populates="predictions")
    family = relationship("ProductFamily", back_populates="predictions")
    
    # Indexes
    __table_args__ = (
        Index("idx_predictions_store_family_date", "store_id", "family_id", "target_date"),
    )
    
    def __repr__(self):
        return f"<Prediction(store_id={self.store_id}, family_id={self.family_id}, target_date={self.target_date})>"


class HistoricalSales(Base):
    """Historical sales data."""
    
    __tablename__ = "historical_sales"
    
    id = Column(Integer, primary_key=True, autoincrement=True)
    store_id = Column(Integer, ForeignKey("stores.id"), nullable=False)
    family_id = Column(Integer, ForeignKey("product_families.id"), nullable=False)
    date = Column(DateTime, nullable=False)
    sales = Column(Float, nullable=False)
    onpromotion = Column(Boolean, default=False)
    created_at = Column(DateTime, default=datetime.utcnow)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)
    
    # Relationships
    store = relationship("Store", back_populates="historical_sales")
    family = relationship("ProductFamily", back_populates="historical_sales")
    
    # Indexes
    __table_args__ = (
        Index("idx_historical_sales_store_family_date", "store_id", "family_id", "date"),
    )
    
    def __repr__(self):
        return f"<HistoricalSales(store_id={self.store_id}, family_id={self.family_id}, date={self.date})>"


class ModelMetric(Base):
    """Model metrics for tracking model performance."""
    
    __tablename__ = "model_metrics"
    
    id = Column(Integer, primary_key=True, autoincrement=True)
    model_name = Column(String(100), nullable=False)
    model_version = Column(String(50), nullable=False)
    metric_name = Column(String(100), nullable=False)
    metric_value = Column(Float, nullable=False)
    timestamp = Column(DateTime, default=datetime.utcnow)
    
    # Indexes
    __table_args__ = (
        Index("idx_model_metrics_model_version", "model_name", "model_version"),
    )
    
    def __repr__(self):
        return f"<ModelMetric(model_name={self.model_name}, version={self.model_version}, metric={self.metric_name})>"


class FeatureImportance(Base):
    """Feature importance for model explainability."""
    
    __tablename__ = "feature_importance"
    
    id = Column(Integer, primary_key=True, autoincrement=True)
    model_name = Column(String(100), nullable=False)
    model_version = Column(String(50), nullable=False)
    feature_name = Column(String(100), nullable=False)
    importance_value = Column(Float, nullable=False)
    timestamp = Column(DateTime, default=datetime.utcnow)
    
    # Indexes
    __table_args__ = (
        Index("idx_feature_importance", "model_name", "model_version", "feature_name"),
    )
    
    def __repr__(self):
        return f"<FeatureImportance(model={self.model_name}, feature={self.feature_name})>"


class ModelDrift(Base):
    """Model drift metrics tracking."""
    
    __tablename__ = "model_drift"
    
    id = Column(Integer, primary_key=True, autoincrement=True)
    model_name = Column(String(100), nullable=False)
    model_version = Column(String(50), nullable=False)
    drift_metric = Column(String(100), nullable=False)
    drift_value = Column(Float, nullable=False)
    threshold = Column(Float, nullable=True)
    exceeded_threshold = Column(Boolean, default=False)
    timestamp = Column(DateTime, default=datetime.utcnow)
    
    def __repr__(self):
        return f"<ModelDrift(model={self.model_name}, metric={self.drift_metric})>" 