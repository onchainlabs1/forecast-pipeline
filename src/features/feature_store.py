#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Feature Store for managing, storing, and retrieving features.
Provides centralized feature management for training and inference.
"""

import os
import logging
import json
import pickle
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Any, Union, Optional

import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, OneHotEncoder

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)

# Define paths
PROJECT_DIR = Path(__file__).resolve().parents[2]
FEATURE_STORE_DIR = PROJECT_DIR / "feature_store"


class FeatureStore:
    """
    Feature Store for managing and serving features.
    
    This class provides functionality to:
    - Register feature definitions
    - Generate and store feature values
    - Retrieve features for training and inference
    - Track feature metadata and lineage
    """
    
    def __init__(self):
        """Initialize the feature store."""
        self.feature_registry = {}
        self.transformers = {}
        self.metadata = {}
        
        # Create feature store directory if it doesn't exist
        os.makedirs(FEATURE_STORE_DIR, exist_ok=True)
        
        # Load feature registry if it exists
        self._load_registry()
    
    def _load_registry(self):
        """Load feature registry from disk if it exists."""
        registry_path = FEATURE_STORE_DIR / "feature_registry.json"
        if registry_path.exists():
            with open(registry_path, "r") as f:
                self.feature_registry = json.load(f)
            logger.info(f"Loaded feature registry with {len(self.feature_registry)} features")
    
    def _save_registry(self):
        """Save feature registry to disk."""
        registry_path = FEATURE_STORE_DIR / "feature_registry.json"
        with open(registry_path, "w") as f:
            json.dump(self.feature_registry, f, indent=2)
        logger.info(f"Saved feature registry with {len(self.feature_registry)} features")
    
    def register_feature(self, 
                        feature_name: str, 
                        description: str,
                        feature_type: str,
                        entity_type: str,
                        transformation: Optional[str] = None,
                        dependencies: Optional[List[str]] = None):
        """
        Register a new feature definition.
        
        Parameters
        ----------
        feature_name : str
            Unique name for the feature
        description : str
            Description of what the feature represents
        feature_type : str
            Data type of the feature (numeric, categorical, temporal, etc.)
        entity_type : str
            Type of entity this feature belongs to (e.g., store, product)
        transformation : str, optional
            Transformation to apply (e.g., standardization, one-hot encoding)
        dependencies : List[str], optional
            List of features this feature depends on
        """
        feature_def = {
            "name": feature_name,
            "description": description,
            "type": feature_type,
            "entity_type": entity_type,
            "transformation": transformation,
            "dependencies": dependencies or [],
            "created_at": datetime.now().isoformat(),
            "updated_at": datetime.now().isoformat()
        }
        
        self.feature_registry[feature_name] = feature_def
        logger.info(f"Registered feature: {feature_name}")
        
        # Save the updated registry
        self._save_registry()
        
        return feature_def
    
    def generate_features(self, 
                         data: pd.DataFrame, 
                         feature_list: List[str],
                         mode: str = "training") -> pd.DataFrame:
        """
        Generate features from input data.
        
        Parameters
        ----------
        data : pd.DataFrame
            Input data for feature generation
        feature_list : List[str]
            List of features to generate
        mode : str
            'training' or 'inference' mode
            
        Returns
        -------
        pd.DataFrame
            DataFrame with generated features
        """
        features_df = data.copy()
        
        # Track which features we've processed to respect dependencies
        processed_features = set()
        
        # Track transformers for future use
        if mode == "training":
            self.transformers = {}
        
        # Process features in order of dependencies
        while len(processed_features) < len(feature_list):
            for feature_name in feature_list:
                if feature_name in processed_features:
                    continue
                    
                feature_def = self.feature_registry.get(feature_name)
                if not feature_def:
                    logger.warning(f"Feature not found in registry: {feature_name}")
                    processed_features.add(feature_name)
                    continue
                
                # Check if all dependencies are processed
                dependencies = feature_def.get("dependencies", [])
                if not all(dep in processed_features for dep in dependencies):
                    continue
                
                # Apply transformation if needed
                transformation = feature_def.get("transformation")
                if transformation:
                    features_df = self._apply_transformation(
                        features_df, 
                        feature_name, 
                        transformation, 
                        mode
                    )
                
                processed_features.add(feature_name)
        
        # Save transformers if in training mode
        if mode == "training":
            self._save_transformers()
        
        return features_df
    
    def _apply_transformation(self, 
                             df: pd.DataFrame, 
                             feature_name: str, 
                             transformation: str, 
                             mode: str) -> pd.DataFrame:
        """
        Apply transformation to a feature.
        
        Parameters
        ----------
        df : pd.DataFrame
            Input DataFrame
        feature_name : str
            Name of the feature to transform
        transformation : str
            Type of transformation to apply
        mode : str
            'training' or 'inference' mode
            
        Returns
        -------
        pd.DataFrame
            DataFrame with transformed feature
        """
        if feature_name not in df.columns:
            logger.warning(f"Feature {feature_name} not found in dataframe")
            return df
        
        # Handle different transformation types
        if transformation == "standardize":
            if mode == "training":
                scaler = StandardScaler()
                df[feature_name] = scaler.fit_transform(df[[feature_name]])
                self.transformers[feature_name] = scaler
            else:
                scaler = self.transformers.get(feature_name)
                if scaler:
                    df[feature_name] = scaler.transform(df[[feature_name]])
        
        elif transformation == "one_hot":
            if mode == "training":
                encoder = OneHotEncoder(sparse=False, handle_unknown="ignore")
                encoded = encoder.fit_transform(df[[feature_name]])
                self.transformers[feature_name] = encoder
                
                # Create new columns for encoded values
                categories = encoder.categories_[0]
                for i, category in enumerate(categories):
                    df[f"{feature_name}_{category}"] = encoded[:, i]
                
                # Drop original column
                df = df.drop(feature_name, axis=1)
            else:
                encoder = self.transformers.get(feature_name)
                if encoder:
                    encoded = encoder.transform(df[[feature_name]])
                    
                    # Create new columns for encoded values
                    categories = encoder.categories_[0]
                    for i, category in enumerate(categories):
                        df[f"{feature_name}_{category}"] = encoded[:, i]
                    
                    # Drop original column
                    df = df.drop(feature_name, axis=1)
        
        return df
    
    def _save_transformers(self):
        """Save transformers to disk."""
        transformers_path = FEATURE_STORE_DIR / "transformers.pkl"
        with open(transformers_path, "wb") as f:
            pickle.dump(self.transformers, f)
        logger.info(f"Saved {len(self.transformers)} transformers")
    
    def load_transformers(self):
        """Load transformers from disk."""
        transformers_path = FEATURE_STORE_DIR / "transformers.pkl"
        if transformers_path.exists():
            with open(transformers_path, "rb") as f:
                self.transformers = pickle.load(f)
            logger.info(f"Loaded {len(self.transformers)} transformers")
    
    def get_feature_metadata(self, feature_name: str) -> Dict:
        """
        Get metadata for a specific feature.
        
        Parameters
        ----------
        feature_name : str
            Name of the feature
            
        Returns
        -------
        Dict
            Feature metadata
        """
        return self.feature_registry.get(feature_name, {})
    
    def list_features(self, entity_type: Optional[str] = None) -> List[str]:
        """
        List all registered features, optionally filtered by entity type.
        
        Parameters
        ----------
        entity_type : str, optional
            Entity type to filter by
            
        Returns
        -------
        List[str]
            List of feature names
        """
        if entity_type:
            return [
                name for name, def_dict in self.feature_registry.items()
                if def_dict.get("entity_type") == entity_type
            ]
        return list(self.feature_registry.keys())
    
    def create_feature_group(self, 
                           group_name: str, 
                           feature_list: List[str],
                           description: str) -> Dict:
        """
        Create a feature group for easy access to related features.
        
        Parameters
        ----------
        group_name : str
            Name of the feature group
        feature_list : List[str]
            List of features in the group
        description : str
            Description of the feature group
            
        Returns
        -------
        Dict
            Feature group definition
        """
        # Validate features exist
        for feature in feature_list:
            if feature not in self.feature_registry:
                logger.warning(f"Feature {feature} not in registry, skipping")
                feature_list.remove(feature)
        
        group_def = {
            "name": group_name,
            "features": feature_list,
            "description": description,
            "created_at": datetime.now().isoformat()
        }
        
        # Save the group definition
        groups_path = FEATURE_STORE_DIR / "feature_groups.json"
        
        # Load existing groups
        groups = {}
        if groups_path.exists():
            with open(groups_path, "r") as f:
                groups = json.load(f)
        
        groups[group_name] = group_def
        
        # Save updated groups
        with open(groups_path, "w") as f:
            json.dump(groups, f, indent=2)
        
        logger.info(f"Created feature group {group_name} with {len(feature_list)} features")
        return group_def
    
    def get_feature_group(self, group_name: str) -> Dict:
        """
        Get a feature group definition.
        
        Parameters
        ----------
        group_name : str
            Name of the feature group
            
        Returns
        -------
        Dict
            Feature group definition
        """
        groups_path = FEATURE_STORE_DIR / "feature_groups.json"
        if not groups_path.exists():
            return {}
        
        with open(groups_path, "r") as f:
            groups = json.load(f)
        
        return groups.get(group_name, {})
    
    def list_feature_groups(self) -> List[str]:
        """
        List all feature groups.
        
        Returns
        -------
        List[str]
            List of feature group names
        """
        groups_path = FEATURE_STORE_DIR / "feature_groups.json"
        if not groups_path.exists():
            return []
        
        with open(groups_path, "r") as f:
            groups = json.load(f)
        
        return list(groups.keys()) 