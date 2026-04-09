"""
LightGBM model training and inference wrapper.
Includes class weighting and hyperparameter tuning.
"""

import numpy as np
import lightgbm as lgb
from sklearn.metrics import roc_auc_score, classification_report, confusion_matrix
import json


class StressDetectionModel:
    """Wrapper for LightGBM crop stress classifier."""
    
    def __init__(self, random_state=42):
        self.model = None
        self.feature_names = None
        self.random_state = random_state
        self.metadata = {}
    
    def train(self, X_train, y_train, X_val=None, y_val=None, verbose=False):
        """
        Train LightGBM with class weights and early stopping.
        
        Args:
            X_train: pd.DataFrame training features
            y_train: pd.Series training labels
            X_val: pd.DataFrame validation features (optional)
            y_val: pd.Series validation labels (optional)
            verbose: bool, print training progress
        """
        self.feature_names = list(X_train.columns)
        
        # Compute class weights (stress events are rarer)
        class_weight = len(y_train) / (2 * np.bincount(y_train))
        
        # LightGBM parameters
        params = {
            'objective': 'binary',
            'metric': ['auc', 'binary_logloss'],
            'learning_rate': 0.05,
            'num_leaves': 31,
            'max_depth': -1,
            'min_data_in_leaf': 20,
            'feature_fraction': 0.8,
            'bagging_fraction': 0.8,
            'bagging_freq': 1,
            'lambda_l1': 0.5,
            'lambda_l2': 0.5,
            'random_state': self.random_state,
            'verbose': -1
        }
        
        train_data = lgb.Dataset(
            X_train, label=y_train,
            feature_name=self.feature_names,
            weight=pd.Series(y_train.map(lambda x: class_weight[int(x)])),
            free_raw_data=False
        )
        
        # Validation dataset if provided
        valid_data = None
        if X_val is not None and y_val is not None:
            valid_data = lgb.Dataset(
                X_val, label=y_val,
                feature_name=self.feature_names,
                reference=train_data,
                free_raw_data=False
            )
        
        # Train
        self.model = lgb.train(
            params=params,
            train_set=train_data,
            num_boost_round=500,
            valid_sets=[valid_data] if valid_data else None,
            valid_names=['validation'],
            callbacks=[
                lgb.early_stopping(20),
                lgb.log_evaluation(period=50 if verbose else -1)
            ] if valid_data else [
                lgb.log_evaluation(period=50 if verbose else -1)
            ]
        )
        
        self.metadata['num_features'] = len(self.feature_names)
        self.metadata['num_rounds'] = self.model.num_trees()
        
        return self
    
    def predict(self, X, return_proba=True):
        """
        Predict stress probability and classification.
        
        Returns:
            If return_proba=True: tuple (y_pred_proba, y_pred_binary)
            If return_proba=False: y_pred_binary
        """
        if self.model is None:
            raise ValueError("Model not trained. Call train() first.")
        
        y_proba = self.model.predict(X, num_iteration=self.model.best_iteration)
        y_pred = (y_proba >= 0.5).astype(int)
        
        return (y_proba, y_pred) if return_proba else y_pred
    
    def get_feature_importance(self, importance_type='gain', top_n=10):
        """Get feature importances from the model."""
        if self.model is None:
            raise ValueError("Model not trained.")
        
        importance = self.model.feature_importance(importance_type=importance_type)
        feature_importance_df = pd.DataFrame({
            'feature': self.feature_names,
            'importance': importance
        }).sort_values('importance', ascending=False)
        
        return feature_importance_df.head(top_n)
    
    def evaluate(self, X_test, y_test, threshold=0.5):
        """
        Evaluate model on test set.
        
        Returns dict with metrics.
        """
        y_proba, y_pred = self.predict(X_test, return_proba=True)
        
        from sklearn.metrics import (
            precision_score, recall_score, f1_score,
            confusion_matrix, roc_auc_score
        )
        
        cm = confusion_matrix(y_test, y_pred)
        # Handle edge case where only one class is present
        if cm.size == 1:
            if y_test[0] == 0:
                tn, fp, fn, tp = cm[0,0], 0, 0, 0
            else:
                tn, fp, fn, tp = 0, 0, 0, cm[0,0]
        else:
            tn, fp, fn, tp = cm.ravel()
        
        metrics = {
            'auc_roc': float(roc_auc_score(y_test, y_proba)) if len(set(y_test)) > 1 else 0.0,
            'precision': float(precision_score(y_test, y_pred, zero_division=0)),
            'recall': float(recall_score(y_test, y_pred, zero_division=0)),
            'f1': float(f1_score(y_test, y_pred, zero_division=0)),
            'specificity': float(tn / (tn + fp) if (tn + fp) > 0 else 0),
            'false_alert_rate': float(fp / (fp + tn) if (fp + tn) > 0 else 0),
            'confusion_matrix': {'tn': int(tn), 'fp': int(fp), 'fn': int(fn), 'tp': int(tp)},
            'num_positives': int(y_test.sum()),
            'num_negatives': int((1 - y_test).sum())
        }
        
        return metrics
    
    def save(self, path):
        """Save model to file using joblib."""
        if self.model is None:
            raise ValueError("No model to save.")
        import joblib
        # Save the model dictionary with all metadata
        model_data = {
            'model': self.model,
            'feature_names': self.feature_names,
            'metadata': self.metadata
        }
        joblib.dump(model_data, path)
        print(f"✓ Model saved to {path}")
    
    def load(self, path):
        """Load model from joblib file."""
        import joblib
        model_data = joblib.load(path)
        self.model = model_data['model']
        self.feature_names = model_data['feature_names']
        self.metadata = model_data['metadata']
        print(f"✓ Model loaded from {path}")
        return self


import pandas as pd
