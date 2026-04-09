"""Test fixtures and utility functions."""

import pytest
import pandas as pd
import numpy as np
from pathlib import Path
import sys

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent))


@pytest.fixture
def sample_ndvi_data():
    """Generate sample NDVI time series."""
    dates = pd.date_range('2023-04-01', '2023-10-31', freq='8D')
    data = []
    for field_id in ['field_001', 'field_002', 'field_003']:
        for date in dates:
            data.append({
                'field_id': field_id,
                'date': date.strftime('%Y-%m-%d'),
                'ndvi': 0.3 + 0.4 * np.sin(np.pi * (date - dates[0]).days / 214) + 0.02 * np.random.randn(),
                'quality_flag': 'good'
            })
    return pd.DataFrame(data)


@pytest.fixture
def sample_sensor_data():
    """Generate sample soil sensor data."""
    dates = pd.date_range('2023-04-01', '2023-10-31', freq='6H')
    data = []
    for field_id in ['field_001', 'field_002', 'field_003']:
        for date in dates:
            data.append({
                'field_id': field_id,
                'timestamp': date.strftime('%Y-%m-%d %H:%M:%S'),
                'soil_moisture_vol': 0.35 + 0.02 * np.random.randn(),
                'soil_temp_c': 15 + 12 * np.sin(np.pi * (date - dates[0]).days / 214),
                'soil_ec_ds_m': 0.3 + 0.05 * np.random.randn(),
                'sensor_id': f'{field_id}_1'
            })
    return pd.DataFrame(data)


@pytest.fixture
def sample_labels():
    """Generate sample stress labels."""
    dates = pd.date_range('2023-04-01', '2023-10-31', freq='8D')
    data = []
    for field_id in ['field_001', 'field_002', 'field_003']:
        for date in dates:
            # Random labels for testing
            label = 1 if np.random.rand() > 0.85 else 0
            data.append({
                'field_id': field_id,
                'date': date.strftime('%Y-%m-%d'),
                'stress_label': label
            })
    return pd.DataFrame(data)


@pytest.fixture
def feature_data(sample_ndvi_data, sample_sensor_data, sample_labels):
    """Generate feature-engineered data."""
    from src.preprocessing import align_multimodal_data
    
    df = align_multimodal_data(
        sample_ndvi_data,
        sample_sensor_data,
        sample_labels
    )
    
    # Add basic features
    df['ndvi_rolling_mean_7d'] = df.groupby('field_id')['ndvi'].rolling(7, min_periods=1).mean().reset_index(drop=True)
    df['ndvi_rolling_std_7d'] = df.groupby('field_id')['ndvi'].rolling(7, min_periods=1).std().reset_index(drop=True)
    df['lag_1_ndvi'] = df.groupby('field_id')['ndvi'].shift(1).reset_index(drop=True)
    df['lag_7_ndvi'] = df.groupby('field_id')['ndvi'].shift(7).reset_index(drop=True)
    df['ndvi_change_pct'] = df.groupby('field_id')['ndvi'].pct_change().reset_index(drop=True)
    df['ndvi_change_from_7d_ago'] = df['ndvi'] - df['lag_7_ndvi']
    df['lag_1_soil_moisture'] = df.groupby('field_id')['soil_moisture_vol'].shift(1).reset_index(drop=True)
    df['lag_7_soil_moisture'] = df.groupby('field_id')['soil_moisture_vol'].shift(7).reset_index(drop=True)
    df['soil_moisture_deficit'] = (0.4 - df['soil_moisture_vol']).clip(lower=0)
    df['rainfall_proxy'] = df.groupby('field_id')['soil_moisture_vol'].rolling(7, min_periods=1).mean().reset_index(drop=True)
    df['extreme_heat_proxy'] = (df['soil_temp_c'] > 28).astype(int)
    df['moisture_temp_interaction'] = df['soil_moisture_vol'] * (1 / (1 + df['soil_temp_c']))
    df['soil_ec_ds_m'] = df['soil_ec_ds_m'].fillna(0.3)
    df['lag_3_ndvi'] = df.groupby('field_id')['ndvi'].shift(3).reset_index(drop=True)
    
    return df.dropna()
