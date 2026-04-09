"""Unit tests for model inference."""

import pytest
import pandas as pd
import numpy as np
from src.model import StressDetectionModel


class TestStressModel:
    """Tests for StressDetectionModel."""
    
    @pytest.fixture
    def trained_model(self, feature_data):
        """Create and train a simple model for testing."""
        from src.preprocessing import select_features, prepare_train_test_split
        
        train, val, test = prepare_train_test_split(feature_data, test_size=0.2, val_size=0.1)
        
        feature_cols = select_features(train)
        X_train = train[feature_cols].fillna(0)
        y_train = train['stress_label']
        X_val = val[feature_cols].fillna(0)
        y_val = val['stress_label']
        
        model = StressDetectionModel()
        model.train(X_train, y_train, X_val, y_val, verbose=False)
        
        return model, X_train, y_train, X_val, y_val, feature_cols
    
    def test_model_training(self, trained_model):
        """Test basic model training."""
        model, X_train, y_train, _, _, _ = trained_model
        
        assert model.model is not None
        assert len(model.feature_names) > 0
    
    def test_prediction_shape(self, trained_model):
        """Test prediction output shape."""
        model, X_train, _, _, _, _ = trained_model
        
        y_proba, y_pred = model.predict(X_train.iloc[:5], return_proba=True)
        
        assert len(y_proba) == 5
        assert len(y_pred) == 5
        assert all((y_proba >= 0) & (y_proba <= 1))
        assert all((y_pred == 0) | (y_pred == 1))
    
    def test_probability_range(self, trained_model):
        """Test that predictions are valid probabilities."""
        model, X_train, _, _, _, _ = trained_model
        
        y_proba, _ = model.predict(X_train, return_proba=True)
        
        assert all(y_proba >= 0)
        assert all(y_proba <= 1)
    
    def test_feature_importance(self, trained_model):
        """Test feature importance extraction."""
        model, _, _, _, _, _ = trained_model
        
        importance_df = model.get_feature_importance(top_n=5)
        
        assert len(importance_df) <= 5
        assert 'feature' in importance_df.columns
        assert 'importance' in importance_df.columns
        assert all(importance_df['importance'] >= 0)
    
    def test_evaluation_metrics(self, trained_model):
        """Test evaluation metric computation."""
        model, _, _, _, X_val, y_val = trained_model
        X_val = X_val.fillna(0)
        
        metrics = model.evaluate(X_val, y_val)
        
        assert 'precision' in metrics
        assert 'recall' in metrics
        assert 'f1' in metrics
        assert 'auc_roc' in metrics
        assert 0 <= metrics['precision'] <= 1
        assert 0 <= metrics['recall'] <= 1
        assert 0 <= metrics['auc_roc'] <= 1
    
    def test_serialization(self, trained_model, tmp_path):
        """Test model save/load."""
        model, _, _, _, _, _ = trained_model
        
        model_path = tmp_path / "test_model.pkl"
        model.save(str(model_path))
        
        assert model_path.exists()
        
        # Load model
        new_model = StressDetectionModel()
        new_model.load(str(model_path))
        
        assert new_model.model is not None
