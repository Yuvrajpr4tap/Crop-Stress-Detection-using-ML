"""
SHAP explainability integration for crop stress predictions.
Generates local and global explanations.
"""

import numpy as np
import pandas as pd
import shap
import json
from typing import List, Dict


class StressExplainer:
    """SHAP-based explainability for stress predictions."""
    
    def __init__(self, model, X_sample, feature_names=None):
        """
        Initialize explainer.
        
        Args:
            model: Trained model (StressDetectionModel or LightGBM model)
            X_sample: Background/reference dataset for SHAP (typically validation set)
            feature_names: List of feature names
        """
        # Extract the underlying LightGBM model if wrapper is used
        if hasattr(model, 'model'):
            lgb_model = model.model
        else:
            lgb_model = model
            
        self.model = lgb_model
        self.feature_names = feature_names or list(range(X_sample.shape[1]))
        
        # Use a sample of background data for efficiency
        sample_size = min(100, len(X_sample))
        background_indices = np.random.choice(len(X_sample), sample_size, replace=False)
        X_background = X_sample.iloc[background_indices] if isinstance(X_sample, pd.DataFrame) else X_sample[background_indices]
        
        # Initialize TreeExplainer
        self.explainer = shap.TreeExplainer(lgb_model, X_background)
        self.background_data = X_background
    
    def explain_prediction(self, X_instance, top_k=3):
        """
        Generate SHAP explanation for a single prediction.
        
        Returns:
            Dict with SHAP values, base value, and top-k features
        """
        if isinstance(X_instance, pd.DataFrame):
            X_instance = X_instance.values
        
        if len(X_instance.shape) == 1:
            X_instance = X_instance.reshape(1, -1)
        
        # Compute SHAP values
        shap_values = self.explainer.shap_values(X_instance)[0]  # binary class -> one set of values
        
        # Get base value (expected model output)
        base_value = float(self.explainer.expected_value)
        
        # Find top-k contributions
        abs_shap = np.abs(shap_values)
        top_indices = np.argsort(abs_shap)[-top_k:][::-1]
        
        top_features = [
            {
                'name': str(self.feature_names[i]),
                'shap_value': float(shap_values[i]),
                'instance_value': float(X_instance[0, i])
            }
            for i in top_indices
        ]
        
        return {
            'shap_values': shap_values.tolist(),
            'base_value': base_value,
            'top_features': top_features,
            'feature_names': self.feature_names
        }
    
    def global_feature_importance(self, X_data, top_k=10):
        """
        Compute global SHAP-based feature importance.
        
        Returns:
            DataFrame with mean absolute SHAP values
        """
        shap_values = self.explainer.shap_values(X_data)
        
        # Mean absolute SHAP values
        mean_abs_shap = np.abs(shap_values).mean(axis=0)
        
        importance_df = pd.DataFrame({
            'feature': self.feature_names,
            'shap_importance': mean_abs_shap
        }).sort_values('shap_importance', ascending=False)
        
        return importance_df.head(top_k)
    
    def generate_explanation_text(self, explanation_dict):
        """
        Generate human-readable explanation from SHAP values.
        
        Returns:
            str: One-sentence explanation
        """
        top_features = explanation_dict.get('top_features', [])
        
        if not top_features:
            return "Unable to generate explanation."
        
        # Map feature names to readable descriptions
        feature_descriptions = {
            'ndvi': 'vegetation index',
            'soil_moisture': 'soil moisture',
            'soil_temp': 'soil temperature',
            'rainfall': 'rainfall',
            'ndvi_change': 'NDVI decline',
            'lag': 'recent stress signal',
            'rolling_mean': 'persistent low vegetation',
            'deficit': 'moisture deficit',
            'ec': 'soil salinity'
        }
        
        # Extract top 2-3 feature descriptions
        descriptions = []
        for feature in top_features[:3]:
            name = feature['name'].lower()
            desc = next(
                (feature_descriptions[k] for k in feature_descriptions if k in name),
                name
            )
            descriptions.append(desc)
        
        # Generate sentence
        if len(descriptions) >= 2:
            return f"Stress driven by {descriptions[0]} and {descriptions[1]}. " \
                   f"Consider irrigation or pest management actions."
        else:
            return f"Primary stress indicator: {descriptions[0]}. " \
                   f"Monitor field closely for worsening conditions."
    
    def save(self, path):
        """Save explainer to pickle."""
        import pickle
        with open(path, 'wb') as f:
            pickle.dump(self, f)
        print(f"✓ Explainer saved to {path}")
    
    @staticmethod
    def load(path):
        """Load explainer from pickle."""
        import pickle
        with open(path, 'rb') as f:
            explainer = pickle.load(f)
        print(f"✓ Explainer loaded from {path}")
        return explainer
