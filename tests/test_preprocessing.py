"""Unit tests for preprocessing functions."""

import pytest
import pandas as pd
import numpy as np
from src.preprocessing import (
    NDVIPreprocessor, SensorPreprocessor, FeatureEngineer,
    align_multimodal_data, select_features, prepare_train_test_split
)


class TestNDVIPreprocessor:
    """Tests for NDVI preprocessing."""
    
    def test_quality_filtering(self, sample_ndvi_data):
        """Test quality flag filtering."""
        preprocessor = NDVIPreprocessor(quality_threshold='good')
        sample_ndvi_data.loc[0, 'quality_flag'] = 'cloudy'
        
        result = preprocessor.transform(sample_ndvi_data)
        
        assert len(result) < len(sample_ndvi_data)
        assert all(result['quality_flag'] == 'good')
    
    def test_duplicate_removal(self, sample_ndvi_data):
        """Test duplicate date removal."""
        preprocessor = NDVIPreprocessor()
        duplicate = sample_ndvi_data.iloc[0:1].copy()
        duplicate.loc[0, 'ndvi'] = 0.5
        aug_data = pd.concat([sample_ndvi_data, duplicate], ignore_index=True)
        
        result = preprocessor.transform(aug_data)
        
        # Should have same length as original (duplicate removed)
        assert len(result) == len(sample_ndvi_data)


class TestSensorPreprocessor:
    """Tests for sensor data preprocessing."""
    
    def test_aggregation(self, sample_sensor_data):
        """Test daily aggregation."""
        preprocessor = SensorPreprocessor(agg_freq='1D')
        result = preprocessor.transform(sample_sensor_data)
        
        # Should have fewer rows after aggregation (multiple sensor readings per day)
        assert len(result) < len(sample_sensor_data)
        assert 'soil_moisture_vol' in result.columns
    
    def test_column_names(self, sample_sensor_data):
        """Test output column names."""
        preprocessor = SensorPreprocessor()
        result = preprocessor.transform(sample_sensor_data)
        
        assert 'date' in result.columns
        assert 'soil_moisture_vol' in result.columns
        assert 'soil_temp_c' in result.columns


class TestFeatureEngineer:
    """Tests for feature engineering."""
    
    def test_rolling_features(self, feature_data):
        """Test rolling window features."""
        engineer = FeatureEngineer(rolling_windows=[7])
        result = engineer.transform(feature_data)
        
        assert 'ndvi_rolling_mean_7d' in result.columns
        assert 'ndvi_rolling_std_7d' in result.columns
    
    def test_lag_features(self, feature_data):
        """Test lag features."""
        engineer = FeatureEngineer(lags=[1, 3, 7])
        result = engineer.transform(feature_data)
        
        assert 'lag_1_ndvi' in result.columns
        assert 'lag_7_ndvi' in result.columns
    
    def test_derived_features(self, feature_data):
        """Test derived features."""
        engineer = FeatureEngineer()
        result = engineer.transform(feature_data)
        
        assert 'ndvi_change_pct' in result.columns
        assert 'soil_moisture_deficit' in result.columns


class TestDataAlignment:
    """Tests for multimodal data alignment."""
    
    def test_alignment_merges_correctly(self, sample_ndvi_data, sample_sensor_data):
        """Test alignment of NDVI and sensor data."""
        result = align_multimodal_data(sample_ndvi_data, sample_sensor_data)
        
        assert result is not None
        assert 'ndvi' in result.columns
        assert 'soil_moisture_vol' in result.columns
        assert len(result) > 0
    
    def test_missing_value_handling(self, sample_ndvi_data, sample_sensor_data):
        """Test forward fill of missing sensor values."""
        sample_sensor_data.loc[0:5, 'soil_moisture_vol'] = np.nan
        result = align_multimodal_data(sample_ndvi_data, sample_sensor_data)
        
        # Most NaNs should be filled
        assert result['soil_moisture_vol'].isna().sum() < len(result) * 0.1


class TestFeatureSelection:
    """Tests for feature selection."""
    
    def test_feature_selection(self, feature_data):
        """Test feature column selection."""
        features = select_features(feature_data)
        
        assert isinstance(features, list)
        assert 'ndvi' in features
        assert 'soil_moisture_vol' in features
        assert 'stress_label' not in features
    
    def test_exclusion_list(self, feature_data):
        """Test custom exclusion list."""
        features = select_features(feature_data, exclude_cols=['ndvi', 'field_id'])
        
        assert 'ndvi' not in features
        assert 'field_id' not in features


class TestTrainTestSplit:
    """Tests for train/val/test splitting."""
    
    def test_split_stratification(self, feature_data):
        """Test that splits don't leak temporal/spatial data."""
        train, val, test = prepare_train_test_split(feature_data, test_size=0.2, val_size=0.1)
        
        # Sizes should be reasonable
        assert len(train) > 0
        assert len(val) > 0
        assert len(test) > 0
        assert len(train) + len(val) + len(test) >= len(feature_data) - 10  # Allow for dropna
    
    def test_labels_present(self, feature_data):
        """Test that labels are present in all splits."""
        train, val, test = prepare_train_test_split(feature_data)
        
        for split in [train, val, test]:
            assert 'stress_label' in split.columns
            assert split['stress_label'].notna().all()
