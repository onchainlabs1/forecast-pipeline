#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Repository layer for data access.
"""

import logging
from typing import List, Dict, Any, Optional, Tuple, Union
from datetime import datetime, timedelta
import uuid
from types import SimpleNamespace

import pandas as pd
from sqlalchemy import func, desc, and_, or_, text
from sqlalchemy.orm import Session

from src.database.models import (
    Store, ProductFamily, Prediction, HistoricalSales,
    ModelMetric, FeatureImportance, ModelDrift
)

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)


class StoreRepository:
    """Repository for Store operations."""
    
    @staticmethod
    def get_all(db: Session) -> List[Store]:
        """Get all stores."""
        return db.query(Store).all()
    
    @staticmethod
    def get_by_id(db: Session, store_id: int) -> Optional[Store]:
        """Get store by ID."""
        return db.query(Store).filter(Store.id == store_id).first()
    
    @staticmethod
    def get_by_store_nbr(db: Session, store_nbr: int) -> Optional[Store]:
        """Get store by store number."""
        return db.query(Store).filter(Store.store_nbr == store_nbr).first()
    
    @staticmethod
    def create(db: Session, store_data: Dict[str, Any]) -> Store:
        """Create a new store."""
        store = Store(**store_data)
        db.add(store)
        db.commit()
        db.refresh(store)
        return store
    
    @staticmethod
    def update(db: Session, store_id: int, store_data: Dict[str, Any]) -> Optional[Store]:
        """Update a store."""
        store = db.query(Store).filter(Store.id == store_id).first()
        if store:
            for key, value in store_data.items():
                setattr(store, key, value)
            db.commit()
            db.refresh(store)
        return store
    
    @staticmethod
    def delete(db: Session, store_id: int) -> bool:
        """Delete a store."""
        store = db.query(Store).filter(Store.id == store_id).first()
        if store:
            db.delete(store)
            db.commit()
            return True
        return False
    
    @staticmethod
    def get_or_create_by_store_nbr(db: Session, store_nbr: int, store_data: Optional[Dict[str, Any]] = None) -> Store:
        """Get a store by store number or create it if it doesn't exist."""
        store = db.query(Store).filter(Store.store_nbr == store_nbr).first()
        if not store:
            store_data = store_data or {}
            store_data["store_nbr"] = store_nbr
            store = Store(**store_data)
            db.add(store)
            db.commit()
            db.refresh(store)
        return store


class ProductFamilyRepository:
    """Repository for ProductFamily operations."""
    
    @staticmethod
    def get_all(db: Session) -> List[ProductFamily]:
        """Get all product families."""
        return db.query(ProductFamily).all()
    
    @staticmethod
    def get_by_id(db: Session, family_id: int) -> Optional[ProductFamily]:
        """Get product family by ID."""
        return db.query(ProductFamily).filter(ProductFamily.id == family_id).first()
    
    @staticmethod
    def get_by_name(db: Session, name: str) -> Optional[ProductFamily]:
        """Get product family by name."""
        return db.query(ProductFamily).filter(ProductFamily.name == name).first()
    
    @staticmethod
    def create(db: Session, family_data: Dict[str, Any]) -> ProductFamily:
        """Create a new product family."""
        family = ProductFamily(**family_data)
        db.add(family)
        db.commit()
        db.refresh(family)
        return family
    
    @staticmethod
    def update(db: Session, family_id: int, family_data: Dict[str, Any]) -> Optional[ProductFamily]:
        """Update a product family."""
        family = db.query(ProductFamily).filter(ProductFamily.id == family_id).first()
        if family:
            for key, value in family_data.items():
                setattr(family, key, value)
            db.commit()
            db.refresh(family)
        return family
    
    @staticmethod
    def delete(db: Session, family_id: int) -> bool:
        """Delete a product family."""
        family = db.query(ProductFamily).filter(ProductFamily.id == family_id).first()
        if family:
            db.delete(family)
            db.commit()
            return True
        return False
    
    @staticmethod
    def get_or_create_by_name(db: Session, name: str, family_data: Optional[Dict[str, Any]] = None) -> ProductFamily:
        """Get a product family by name or create it if it doesn't exist."""
        family = db.query(ProductFamily).filter(ProductFamily.name == name).first()
        if not family:
            family_data = family_data or {}
            family_data["name"] = name
            family = ProductFamily(**family_data)
            db.add(family)
            db.commit()
            db.refresh(family)
        return family


class PredictionRepository:
    """Repository for Prediction operations."""
    
    @staticmethod
    def create(db: Session, prediction_data: Dict[str, Any]) -> Prediction:
        """Create a new prediction."""
        # Generate a UUID if not provided
        if "id" not in prediction_data:
            prediction_data["id"] = str(uuid.uuid4())
        
        prediction = Prediction(**prediction_data)
        db.add(prediction)
        db.commit()
        db.refresh(prediction)
        return prediction
    
    @staticmethod
    def get_by_id(db: Session, prediction_id: str) -> Optional[Prediction]:
        """Get prediction by ID."""
        return db.query(Prediction).filter(Prediction.id == prediction_id).first()
    
    @staticmethod
    def get_for_store_family_date(
        db: Session, 
        store_id: int, 
        family_id: int, 
        target_date: datetime
    ) -> Optional[Prediction]:
        """Get prediction for a specific store, family, and date."""
        return db.query(Prediction).filter(
            Prediction.store_id == store_id,
            Prediction.family_id == family_id,
            Prediction.target_date == target_date
        ).order_by(desc(Prediction.prediction_date)).first()
    
    @staticmethod
    def get_recent_predictions(
        db: Session, 
        store_id: Optional[int] = None, 
        family_id: Optional[int] = None,
        days: int = 30,
        limit: int = 100
    ) -> List[Prediction]:
        """Get recent predictions with optional filtering."""
        query = db.query(Prediction)
        
        if store_id:
            query = query.filter(Prediction.store_id == store_id)
        
        if family_id:
            query = query.filter(Prediction.family_id == family_id)
        
        # Filter by target date (future predictions)
        since_date = datetime.utcnow() - timedelta(days=days)
        query = query.filter(Prediction.target_date >= since_date)
        
        return query.order_by(Prediction.target_date).limit(limit).all()
    
    @staticmethod
    def get_prediction_history(
        db: Session,
        store_id: int,
        family_id: int,
        days: int = 30
    ) -> List[Prediction]:
        """Get prediction history for a specific store and family."""
        since_date = datetime.utcnow() - timedelta(days=days)
        return db.query(Prediction).filter(
            Prediction.store_id == store_id,
            Prediction.family_id == family_id,
            Prediction.prediction_date >= since_date
        ).order_by(Prediction.prediction_date).all()
    
    @staticmethod
    def get_predictions_as_dataframe(
        db: Session,
        store_id: Optional[int] = None,
        family_id: Optional[int] = None,
        days: int = 30
    ) -> pd.DataFrame:
        """Get predictions as a pandas DataFrame."""
        query = db.query(
            Prediction.id,
            Store.store_nbr, 
            ProductFamily.name.label("family"),
            Prediction.target_date,
            Prediction.prediction_date,
            Prediction.onpromotion,
            Prediction.predicted_sales,
            Prediction.model_version
        ).join(Store).join(ProductFamily)
        
        if store_id:
            query = query.filter(Prediction.store_id == store_id)
        
        if family_id:
            query = query.filter(Prediction.family_id == family_id)
        
        # Filter by prediction date (recent predictions)
        since_date = datetime.utcnow() - timedelta(days=days)
        query = query.filter(Prediction.prediction_date >= since_date)
        
        # Execute query and convert to DataFrame
        result = query.all()
        if not result:
            return pd.DataFrame(columns=[
                "id", "store_nbr", "family", "target_date", "prediction_date",
                "onpromotion", "predicted_sales", "model_version"
            ])
        
        return pd.DataFrame(result)


class HistoricalSalesRepository:
    """Repository for HistoricalSales operations."""
    
    @staticmethod
    def create(db: Session, sales_data: Dict[str, Any]) -> HistoricalSales:
        """Create a new historical sales record."""
        sales = HistoricalSales(**sales_data)
        db.add(sales)
        db.commit()
        db.refresh(sales)
        return sales
    
    @staticmethod
    def create_many(db: Session, sales_data_list: List[Dict[str, Any]]) -> List[HistoricalSales]:
        """Create multiple historical sales records."""
        sales_objects = [HistoricalSales(**data) for data in sales_data_list]
        db.add_all(sales_objects)
        db.commit()
        return sales_objects
    
    @staticmethod
    def get_by_id(db: Session, sales_id: int) -> Optional[HistoricalSales]:
        """Get historical sales by ID."""
        return db.query(HistoricalSales).filter(HistoricalSales.id == sales_id).first()
    
    @staticmethod
    def get_for_store_family_date(
        db: Session, 
        store_id: int, 
        family_id: int, 
        date: datetime
    ) -> Optional[HistoricalSales]:
        """Get historical sales for a specific store, family, and date."""
        return db.query(HistoricalSales).filter(
            HistoricalSales.store_id == store_id,
            HistoricalSales.family_id == family_id,
            HistoricalSales.date == date
        ).first()
    
    @staticmethod
    def get_sales_history(
        db: Session,
        store_id: int,
        family_id: int,
        days: int = 90
    ) -> List[HistoricalSales]:
        """Get sales history for a specific store and family."""
        since_date = datetime.utcnow() - timedelta(days=days)
        return db.query(HistoricalSales).filter(
            HistoricalSales.store_id == store_id,
            HistoricalSales.family_id == family_id,
            HistoricalSales.date >= since_date
        ).order_by(HistoricalSales.date).all()
    
    @staticmethod
    def get_sales_history_as_dataframe(
        db: Session,
        store_id: Optional[int] = None,
        family_id: Optional[int] = None,
        days: int = 90
    ) -> pd.DataFrame:
        """Get sales history as a pandas DataFrame."""
        query = db.query(
            Store.store_nbr, 
            ProductFamily.name.label("family"),
            HistoricalSales.date,
            HistoricalSales.sales,
            HistoricalSales.onpromotion
        ).join(Store).join(ProductFamily)
        
        if store_id:
            query = query.filter(HistoricalSales.store_id == store_id)
        
        if family_id:
            query = query.filter(HistoricalSales.family_id == family_id)
        
        # Filter by date
        since_date = datetime.utcnow() - timedelta(days=days)
        query = query.filter(HistoricalSales.date >= since_date)
        
        # Execute query and convert to DataFrame
        result = query.all()
        if not result:
            return pd.DataFrame(columns=["store_nbr", "family", "date", "sales", "onpromotion"])
        
        return pd.DataFrame(result)
    
    @staticmethod
    def get_family_performance(db: Session, days: int = 30) -> pd.DataFrame:
        """Get performance metrics by product family."""
        since_date = datetime.utcnow() - timedelta(days=days)
        
        query = db.query(
            ProductFamily.name.label("family"),
            func.sum(HistoricalSales.sales).label("total_sales"),
            func.avg(HistoricalSales.sales).label("avg_sales"),
            func.count(HistoricalSales.id).label("count")
        ).join(ProductFamily)\
         .filter(HistoricalSales.date >= since_date)\
         .group_by(ProductFamily.name)\
         .order_by(desc("total_sales"))
        
        result = query.all()
        if not result:
            return pd.DataFrame(columns=["family", "total_sales", "avg_sales", "count"])
        
        return pd.DataFrame(result)
    
    @staticmethod
    def get_store_comparison(db: Session, days: int = 30) -> pd.DataFrame:
        """Get performance metrics by store."""
        since_date = datetime.utcnow() - timedelta(days=days)
        
        query = db.query(
            Store.store_nbr,
            func.sum(HistoricalSales.sales).label("total_sales"),
            func.avg(HistoricalSales.sales).label("avg_sales"),
            func.count(HistoricalSales.id).label("count")
        ).join(Store)\
         .filter(HistoricalSales.date >= since_date)\
         .group_by(Store.store_nbr)\
         .order_by(desc("total_sales"))
        
        result = query.all()
        if not result:
            return pd.DataFrame(columns=["store_nbr", "total_sales", "avg_sales", "count"])
        
        return pd.DataFrame(result)


class ModelMetricRepository:
    """Repository for ModelMetric operations."""
    
    @staticmethod
    def create(db: Session, metric_data: Dict[str, Any]) -> ModelMetric:
        """Create a new model metric."""
        metric = ModelMetric(**metric_data)
        db.add(metric)
        db.commit()
        db.refresh(metric)
        return metric
    
    @staticmethod
    def create_many(db: Session, metrics_data_list: List[Dict[str, Any]]) -> List[ModelMetric]:
        """Create multiple model metrics."""
        metric_objects = [ModelMetric(**data) for data in metrics_data_list]
        db.add_all(metric_objects)
        db.commit()
        return metric_objects
    
    @staticmethod
    def get_by_model_name(
        db: Session, 
        model_name: str,
        days: int = 30
    ) -> List[ModelMetric]:
        """Get metrics for a specific model."""
        since_date = datetime.utcnow() - timedelta(days=days)
        return db.query(ModelMetric).filter(
            ModelMetric.model_name == model_name,
            ModelMetric.timestamp >= since_date
        ).order_by(ModelMetric.timestamp).all()
    
    @staticmethod
    def get_latest_metrics(
        db: Session, 
        model_name: str, 
        model_version: Optional[str] = None
    ) -> Dict[str, float]:
        """Get the latest metrics for a model."""
        query = db.query(
            ModelMetric.metric_name,
            ModelMetric.metric_value
        ).filter(ModelMetric.model_name == model_name)
        
        if model_version:
            query = query.filter(ModelMetric.model_version == model_version)
        
        # Get the latest timestamp for each metric
        subquery = db.query(
            ModelMetric.metric_name,
            func.max(ModelMetric.timestamp).label("max_timestamp")
        ).filter(ModelMetric.model_name == model_name)
        
        if model_version:
            subquery = subquery.filter(ModelMetric.model_version == model_version)
        
        subquery = subquery.group_by(ModelMetric.metric_name).subquery()
        
        # Join with the original query
        query = query.join(
            subquery,
            and_(
                ModelMetric.metric_name == subquery.c.metric_name,
                ModelMetric.timestamp == subquery.c.max_timestamp
            )
        )
        
        # Convert to dict
        result = query.all()
        return {metric_name: metric_value for metric_name, metric_value in result}
    
    @staticmethod
    def get_metrics_by_version(
        db: Session, 
        model_name: str
    ) -> pd.DataFrame:
        """Get metrics for all versions of a model."""
        query = db.query(
            ModelMetric.model_version,
            ModelMetric.metric_name,
            ModelMetric.metric_value
        ).filter(ModelMetric.model_name == model_name)
        
        # Get the latest timestamp for each version and metric
        subquery = db.query(
            ModelMetric.model_version,
            ModelMetric.metric_name,
            func.max(ModelMetric.timestamp).label("max_timestamp")
        ).filter(ModelMetric.model_name == model_name)\
         .group_by(ModelMetric.model_version, ModelMetric.metric_name).subquery()
        
        # Join with the original query
        query = query.join(
            subquery,
            and_(
                ModelMetric.model_version == subquery.c.model_version,
                ModelMetric.metric_name == subquery.c.metric_name,
                ModelMetric.timestamp == subquery.c.max_timestamp
            )
        )
        
        # Execute query and convert to DataFrame
        result = query.all()
        if not result:
            return pd.DataFrame(columns=["model_version", "metric_name", "metric_value"])
        
        df = pd.DataFrame(result)
        
        # Pivot the DataFrame to have metrics as columns
        return df.pivot(index="model_version", columns="metric_name", values="metric_value").reset_index()


class FeatureImportanceRepository:
    """Repository for FeatureImportance operations."""
    
    @staticmethod
    def create(db: Session, importance_data: Dict[str, Any]) -> FeatureImportance:
        """Create a new feature importance record."""
        importance = FeatureImportance(**importance_data)
        db.add(importance)
        db.commit()
        db.refresh(importance)
        return importance
    
    @staticmethod
    def create_many(db: Session, importance_data_list: List[Dict[str, Any]]) -> List[FeatureImportance]:
        """Create multiple feature importance records."""
        importance_objects = [FeatureImportance(**data) for data in importance_data_list]
        db.add_all(importance_objects)
        db.commit()
        return importance_objects
    
    @staticmethod
    def get_by_model_name_version(
        db: Session, 
        model_name: str,
        model_version: str
    ) -> List[FeatureImportance]:
        """Get feature importance for a specific model and version."""
        return db.query(FeatureImportance).filter(
            FeatureImportance.model_name == model_name,
            FeatureImportance.model_version == model_version
        ).order_by(desc(FeatureImportance.importance_value)).all()
    
    @staticmethod
    def get_feature_importance_as_dataframe(
        db: Session,
        model_name: str,
        model_version: Optional[str] = None
    ) -> pd.DataFrame:
        """Get feature importance as a pandas DataFrame."""
        query = db.query(
            FeatureImportance.model_version,
            FeatureImportance.feature_name,
            FeatureImportance.importance_value
        ).filter(FeatureImportance.model_name == model_name)
        
        if model_version:
            query = query.filter(FeatureImportance.model_version == model_version)
        
        # Get the latest timestamp for each version and feature
        subquery = db.query(
            FeatureImportance.model_version,
            FeatureImportance.feature_name,
            func.max(FeatureImportance.timestamp).label("max_timestamp")
        ).filter(FeatureImportance.model_name == model_name)
        
        if model_version:
            subquery = subquery.filter(FeatureImportance.model_version == model_version)
        
        subquery = subquery.group_by(
            FeatureImportance.model_version, 
            FeatureImportance.feature_name
        ).subquery()
        
        # Join with the original query
        query = query.join(
            subquery,
            and_(
                FeatureImportance.model_version == subquery.c.model_version,
                FeatureImportance.feature_name == subquery.c.feature_name,
                FeatureImportance.timestamp == subquery.c.max_timestamp
            )
        )
        
        # Order by importance value
        query = query.order_by(desc(FeatureImportance.importance_value))
        
        # Execute query and convert to DataFrame
        result = query.all()
        if not result:
            return pd.DataFrame(columns=["model_version", "feature_name", "importance_value"])
        
        return pd.DataFrame(result)


class ModelDriftRepository:
    """Repository for ModelDrift operations."""
    
    @staticmethod
    def create(db: Session, drift_data: Dict[str, Any]) -> ModelDrift:
        """Create a new model drift record."""
        drift = ModelDrift(**drift_data)
        db.add(drift)
        db.commit()
        db.refresh(drift)
        return drift
    
    @staticmethod
    def create_many(db: Session, drift_data_list: List[Dict[str, Any]]) -> List[ModelDrift]:
        """Create multiple model drift records."""
        drift_objects = [ModelDrift(**data) for data in drift_data_list]
        db.add_all(drift_objects)
        db.commit()
        return drift_objects
    
    @staticmethod
    def get_recent_drift(
        db: Session,
        model_name: str,
        days: int = 7
    ) -> List[Any]:
        """
        Get recent model drift data grouped by date.
        
        Parameters
        ----------
        db : Session
            Database session
        model_name : str
            Name of the model
        days : int
            Number of days to look back
            
        Returns
        -------
        List[Any]
            List of drift data records with date, rmse, mae, and drift score
        """
        since_date = datetime.utcnow() - timedelta(days=days)
        
        # Using raw SQL for date grouping
        # This creates a daily summary with RMSE, MAE and drift metrics
        query = text("""
            WITH daily_metrics AS (
                SELECT 
                    DATE(timestamp) as date,
                    CASE WHEN drift_metric = 'rmse' THEN drift_value ELSE NULL END as rmse,
                    CASE WHEN drift_metric = 'mae' THEN drift_value ELSE NULL END as mae,
                    exceeded_threshold as drift_detected
                FROM model_drift
                WHERE model_name = :model_name AND timestamp >= :since_date
            )
            SELECT 
                date,
                MAX(rmse) as rmse, 
                MAX(mae) as mae,
                MAX(CASE WHEN drift_detected THEN 1 ELSE 0 END) as drift_detected
            FROM daily_metrics
            GROUP BY date
            ORDER BY date DESC
            LIMIT :days
        """)
        
        result = db.execute(
            query, 
            {"model_name": model_name, "since_date": since_date, "days": days}
        ).all()
        
        # Convert result to list of SimpleNamespace objects for easy attribute access
        drift_data = []
        for row in result:
            # Convert date string to datetime object if needed
            date_obj = pd.to_datetime(row.date) if isinstance(row.date, str) else row.date
            
            drift_data.append(SimpleNamespace(
                date=date_obj,
                rmse=row.rmse if row.rmse is not None else 0.0,
                mae=row.mae if row.mae is not None else 0.0,
                drift_detected=row.drift_detected
            ))
            
        return drift_data
    
    @staticmethod
    def get_by_model_name_version(
        db: Session, 
        model_name: str,
        model_version: str,
        days: int = 30
    ) -> List[ModelDrift]:
        """Get model drift for a specific model and version."""
        since_date = datetime.utcnow() - timedelta(days=days)
        return db.query(ModelDrift).filter(
            ModelDrift.model_name == model_name,
            ModelDrift.model_version == model_version,
            ModelDrift.timestamp >= since_date
        ).order_by(ModelDrift.timestamp).all()
    
    @staticmethod
    def get_model_drift_as_dataframe(
        db: Session,
        model_name: str,
        model_version: Optional[str] = None,
        days: int = 30
    ) -> pd.DataFrame:
        """Get model drift as a pandas DataFrame."""
        since_date = datetime.utcnow() - timedelta(days=days)
        
        query = db.query(
            ModelDrift.model_version,
            ModelDrift.drift_metric,
            ModelDrift.drift_value,
            ModelDrift.threshold,
            ModelDrift.exceeded_threshold,
            ModelDrift.timestamp
        ).filter(
            ModelDrift.model_name == model_name,
            ModelDrift.timestamp >= since_date
        )
        
        if model_version:
            query = query.filter(ModelDrift.model_version == model_version)
        
        # Order by timestamp
        query = query.order_by(ModelDrift.timestamp)
        
        # Execute query and convert to DataFrame
        result = query.all()
        if not result:
            return pd.DataFrame(columns=[
                "model_version", "drift_metric", "drift_value", 
                "threshold", "exceeded_threshold", "timestamp"
            ])
        
        return pd.DataFrame(result) 