"""
Data preprocessing and feature engineering pipeline.
Handles alignment, imputation, and feature creation for crop stress detection.
"""

import numpy as np
import pandas as pd
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.base import BaseEstimator, TransformerMixin
from datetime import datetime, timedelta


class NDVIPreprocessor(BaseEstimator, TransformerMixin):
    """Clean and align NDVI time series."""
    
    def __init__(self, quality_threshold='good'):
        self.quality_threshold = quality_threshold
    
    def fit(self, X, y=None):
        return self
    
    def transform(self, X):
        df = X.copy()
        df['date'] = pd.to_datetime(df['date'])
        
        # Filter by quality
        df = df[df['quality_flag'] == self.quality_threshold]
        
        # Sort and remove duplicates (keep first)
        df = df.sort_values('date').drop_duplicates(subset=['field_id', 'date'], keep='first')
        
        return df


class SensorPreprocessor(BaseEstimator, TransformerMixin):
    """Clean and aggregate soil sensor measurements."""
    
    def __init__(self, agg_freq='1D'):
        self.agg_freq = agg_freq  # pandas frequency string
    
    def fit(self, X, y=None):
        return self
    
    def transform(self, X):
        df = X.copy()
        df['timestamp'] = pd.to_datetime(df['timestamp'])
        
        # Aggregate multiple sensors per field per time period
        df_agg = df.groupby(
            ['field_id', pd.Grouper(key='timestamp', freq=self.agg_freq)]
        ).agg({
            'soil_moisture_vol': 'mean',
            'soil_temp_c': 'mean',
            'soil_ec_ds_m': 'mean'
        }).reset_index()
        
        # Rename timestamp to date for alignment
        df_agg['date'] = df_agg['timestamp'].dt.strftime('%Y-%m-%d')
        
        return df_agg


class FeatureEngineer(BaseEstimator, TransformerMixin):
    """Create lag, rolling, and interaction features."""
    
    def __init__(self, lags=[1, 3, 7], rolling_windows=[7, 14]):
        self.lags = lags
        self.rolling_windows = rolling_windows
    
    def fit(self, X, y=None):
        return self
    
    def transform(self, X):
        df = X.copy()
        
        # Ensure proper sorting
        df = df.sort_values(['field_id', 'date']).reset_index(drop=True)
        
        # Rolling statistics for NDVI
        for window in self.rolling_windows:
            df[f'ndvi_rolling_mean_{window}d'] = (
                df.groupby('field_id')['ndvi'].transform(
                    lambda x: x.rolling(window, min_periods=1).mean()
                )
            )
            df[f'ndvi_rolling_std_{window}d'] = (
                df.groupby('field_id')['ndvi'].transform(
                    lambda x: x.rolling(window, min_periods=1).std()
                )
            )
        
        # Lag features for NDVI
        for lag in self.lags:
            df[f'lag_{lag}_ndvi'] = (
                df.groupby('field_id')['ndvi'].shift(lag)
            )
        
        # NDVI change percentages
        df['ndvi_change_pct'] = (
            df.groupby('field_id')['ndvi'].pct_change() * 100
        )
        df['ndvi_change_from_7d_ago'] = (
            df['ndvi'] - df['lag_7_ndvi']
        )
        
        # Lag features for soil moisture
        for lag in self.lags:
            df[f'lag_{lag}_soil_moisture'] = (
                df.groupby('field_id')['soil_moisture_vol'].shift(lag)
            )
        
        # Soil moisture deficit (proxy for stress)
        df['soil_moisture_deficit'] = (
            0.4 - df['soil_moisture_vol']  # Assume 0.4 is target moisture
        ).clip(lower=0)
        
        # Rainfall proxy (inverse of soil moisture trend)
        df['rainfall_proxy'] = (
            df.groupby('field_id')['soil_moisture_vol'].transform(
                lambda x: x.rolling(7, min_periods=1).mean()
            )
        )
        
        # Temperature stress indicator
        df['extreme_heat_proxy'] = (df['soil_temp_c'] > 28).astype(int)
        
        # Interaction: moisture x temperature
        df['moisture_temp_interaction'] = (
            df['soil_moisture_vol'] * (1 / (1 + df['soil_temp_c']))
        )
        
        return df


def align_multimodal_data(ndvi_df, sensor_df, labels_df=None):
    """
    Align NDVI and sensor data by field and date.
    
    Returns merged DataFrame with forward-filled missing sensor data.
    """
    # Ensure dates are aligned
    ndvi_df['date'] = pd.to_datetime(ndvi_df['date']).dt.strftime('%Y-%m-%d')
    sensor_df['date'] = pd.to_datetime(sensor_df['timestamp']).dt.strftime('%Y-%m-%d')
    
    # Merge on field_id and date
    merged = ndvi_df.merge(
        sensor_df[['field_id', 'date', 'soil_moisture_vol', 'soil_temp_c', 'soil_ec_ds_m']],
        on=['field_id', 'date'],
        how='outer'
    ).sort_values(['field_id', 'date']).reset_index(drop=True)
    
    # Forward-fill missing sensor values (max 3 days)
    for col in ['soil_moisture_vol', 'soil_temp_c', 'soil_ec_ds_m']:
        merged[col] = (
            merged.groupby('field_id')[col]
            .transform(lambda x: x.fillna(method='ffill', limit=3))
        )
    
    # Backward-fill remaining NaNs
    for col in ['soil_moisture_vol', 'soil_temp_c', 'soil_ec_ds_m']:
        merged[col] = (
            merged.groupby('field_id')[col]
            .transform(lambda x: x.fillna(method='bfill', limit=3))
        )
    
    # Drop rows where both NDVI and sensor data are unavailable
    merged = merged.dropna(subset=['ndvi', 'soil_moisture_vol'])
    
    # Merge labels if provided
    if labels_df is not None:
        labels_df['date'] = pd.to_datetime(labels_df['date']).dt.strftime('%Y-%m-%d')
        merged = merged.merge(
            labels_df[['field_id', 'date', 'stress_label']],
            on=['field_id', 'date'],
            how='left'
        )
    
    return merged.reset_index(drop=True)


def build_preprocessing_pipeline():
    """Build sklearn-compatible preprocessing pipeline."""
    return Pipeline([
        ('imputer', SimpleImputer(strategy='mean')),
        ('scaler', StandardScaler())
    ])


def select_features(df, exclude_cols=None):
    """Select model features from engineered dataset."""
    if exclude_cols is None:
        exclude_cols = ['field_id', 'date', 'quality_flag', 'timestamp', 'sensor_id']
    
    feature_cols = [
        col for col in df.columns
        if col not in exclude_cols and col != 'stress_label'
    ]
    
    return feature_cols


def prepare_train_test_split(df, test_size=0.2, val_size=0.1, random_state=42):
    """
    Split data by field and time to avoid temporal/spatial leakage.
    
    Returns:
        train_df, val_df, test_df
    """
    fields = df['field_id'].unique()
    np.random.seed(random_state)
    
    # Select random field for test (time-based)
    test_field = np.random.choice(fields)
    remaining_fields = [f for f in fields if f != test_field]
    
    test_df = df[df['field_id'] == test_field].copy()
    
    # For remaining fields, use temporal split
    other_df = df[df['field_id'].isin(remaining_fields)].copy()
    
    # Sort by date and split
    other_df = other_df.sort_values('date').reset_index(drop=True)
    n = len(other_df)
    val_start = int(n * (1 - test_size - val_size))
    test_start = int(n * (1 - test_size))
    
    train_df = other_df.iloc[:val_start].copy()
    val_df = other_df.iloc[val_start:test_start].copy()
    test_df = pd.concat([test_df, other_df.iloc[test_start:]], ignore_index=True)
    
    print(f"Train: {len(train_df)} | Val: {len(val_df)} | Test: {len(test_df)}")
    print(f"Train stress: {train_df['stress_label'].sum()} | "
          f"Val stress: {val_df['stress_label'].sum()} | "
          f"Test stress: {test_df['stress_label'].sum()}")
    
    return train_df, val_df, test_df
