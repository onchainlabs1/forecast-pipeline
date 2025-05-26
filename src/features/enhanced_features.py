#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Enhanced feature generation module that adds advanced features
without interfering with the existing feature pipeline.
"""

import logging
from typing import Dict, List, Optional
import pandas as pd
import numpy as np
from statsmodels.tsa.seasonal import seasonal_decompose
from datetime import datetime, timedelta

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)


class EnhancedFeatureGenerator:
    """
    Generates advanced features for the sales forecasting model.
    Works alongside the existing FeatureStore without modification.
    """
    
    def __init__(self):
        self.seasonal_periods = {
            'weekly': 7,
            'monthly': 30,
            'yearly': 365
        }
    
    def add_temporal_decomposition(
        self,
        data: pd.DataFrame,
        date_col: str = 'date',
        target_col: str = 'sales',
        group_cols: List[str] = None
    ) -> pd.DataFrame:
        """
        Add temporal decomposition features (trend, seasonality) to the dataset.
        
        Parameters
        ----------
        data : pd.DataFrame
            Input DataFrame with date and target columns
        date_col : str
            Name of the date column
        target_col : str
            Name of the target column (e.g., 'sales')
        group_cols : List[str], optional
            Columns to group by before decomposition (e.g., ['store_nbr', 'family'])
            
        Returns
        -------
        pd.DataFrame
            DataFrame with added decomposition features
        """
        logger.info("Adding temporal decomposition features")
        
        # Create copy to avoid modifying original data
        result_df = data.copy()
        
        try:
            # Convert date column to datetime if needed
            if not pd.api.types.is_datetime64_any_dtype(result_df[date_col]):
                result_df[date_col] = pd.to_datetime(result_df[date_col])
            
            # Sort by date
            result_df = result_df.sort_values(date_col)
            
            # Function to decompose a single series
            def decompose_series(series):
                try:
                    # Decompose with different seasonal periods
                    decompositions = {}
                    for period_name, period in self.seasonal_periods.items():
                        # Check if we have enough data for this period
                        if len(series) >= period * 2:  # Need at least 2 full periods
                            try:
                                decomp = seasonal_decompose(
                                    series,
                                    period=period,
                                    extrapolate_trend=True
                                )
                                decompositions[period_name] = {
                                    'trend': decomp.trend,
                                    'seasonal': decomp.seasonal,
                                    'resid': decomp.resid
                                }
                                logger.info(f"Successfully decomposed {period_name} seasonality")
                            except Exception as e:
                                logger.warning(f"Failed to decompose {period_name} seasonality: {e}")
                        else:
                            logger.warning(
                                f"Not enough data for {period_name} decomposition. "
                                f"Need {period * 2} points, but only have {len(series)}"
                            )
                    return decompositions
                except Exception as e:
                    logger.warning(f"Decomposition failed: {e}")
                    return None
            
            # If group columns provided, decompose by group
            if group_cols:
                for name, group in result_df.groupby(group_cols):
                    decompositions = decompose_series(group[target_col])
                    if decompositions:
                        for period_name, components in decompositions.items():
                            for comp_name, values in components.items():
                                col_name = f"{target_col}_{period_name}_{comp_name}"
                                result_df.loc[group.index, col_name] = values
            else:
                # Decompose entire series
                decompositions = decompose_series(result_df[target_col])
                if decompositions:
                    for period_name, components in decompositions.items():
                        for comp_name, values in components.items():
                            col_name = f"{target_col}_{period_name}_{comp_name}"
                            result_df[col_name] = values
            
            # Fill any missing values from decomposition using forward fill then backward fill
            decomp_cols = [col for col in result_df.columns if '_trend' in col or '_seasonal' in col or '_resid' in col]
            if decomp_cols:
                result_df[decomp_cols] = result_df[decomp_cols].ffill().bfill()
            
            logger.info(f"Added {len(decomp_cols)} temporal decomposition features")
            return result_df
            
        except Exception as e:
            logger.error(f"Error in temporal decomposition: {e}")
            return data  # Return original data if decomposition fails
    
    def add_weather_features(
        self,
        data: pd.DataFrame,
        store_locations: Dict[int, Dict[str, float]],
        date_col: str = 'date'
    ) -> pd.DataFrame:
        """
        Add weather-related features using store locations.
        This is a placeholder that should be connected to a weather API.
        
        Parameters
        ----------
        data : pd.DataFrame
            Input DataFrame
        store_locations : Dict[int, Dict[str, float]]
            Dictionary mapping store_nbr to location (lat, lon)
        date_col : str
            Name of the date column
            
        Returns
        -------
        pd.DataFrame
            DataFrame with added weather features
        """
        logger.info("Adding weather features")
        
        # Create copy to avoid modifying original data
        result_df = data.copy()
        
        try:
            # Here you would typically:
            # 1. Call a weather API for historical data
            # 2. Match weather data to store locations
            # 3. Add features like temperature, precipitation, etc.
            
            # For now, we'll add placeholder features
            # In production, replace this with actual weather API calls
            result_df['temperature'] = 25.0  # placeholder
            result_df['precipitation'] = 0.0  # placeholder
            result_df['humidity'] = 70.0  # placeholder
            
            logger.info("Added placeholder weather features")
            return result_df
            
        except Exception as e:
            logger.error(f"Error adding weather features: {e}")
            return data  # Return original data if weather features fail
    
    def add_promotion_features(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        Add enhanced promotion-related features.
        
        Parameters
        ----------
        data : pd.DataFrame
            Input DataFrame
            
        Returns
        -------
        pd.DataFrame
            DataFrame with added promotion features
        """
        logger.info("Adding enhanced promotion features")
        
        result_df = data.copy()
        
        try:
            # Days since last promotion
            result_df['days_since_promo'] = result_df.groupby(['store_nbr', 'family'])\
                ['onpromotion'].transform(lambda x: (~x).cumsum())
            
            # Days until next promotion (forward fill)
            result_df['days_to_promo'] = result_df.groupby(['store_nbr', 'family'])\
                ['onpromotion'].transform(lambda x: x.iloc[::-1].cumsum().iloc[::-1])
            
            # Promotion density (% of items on promotion in store)
            result_df['store_promo_density'] = result_df.groupby(['store_nbr', 'date'])\
                ['onpromotion'].transform('mean')
            
            # Promotion density for product family
            result_df['family_promo_density'] = result_df.groupby(['family', 'date'])\
                ['onpromotion'].transform('mean')
            
            # Fill any missing values with 0
            promo_cols = [
                'days_since_promo',
                'days_to_promo',
                'store_promo_density',
                'family_promo_density'
            ]
            result_df[promo_cols] = result_df[promo_cols].fillna(0)
            
            logger.info("Added enhanced promotion features")
            return result_df
            
        except Exception as e:
            logger.error(f"Error adding promotion features: {e}")
            return data  # Return original data if promotion features fail 