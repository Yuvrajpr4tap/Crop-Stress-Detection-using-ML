"""Utility functions for crop stress detection."""

import json
import numpy as np
import pandas as pd
from datetime import datetime
from pathlib import Path


def load_field_geometries(geojson_path):
    """Load field polygons from GeoJSON."""
    with open(geojson_path, "r") as f:
        return json.load(f)


def get_field_info(geojson_path, field_id):
    """Get metadata for a specific field."""
    geom = load_field_geometries(geojson_path)
    for feature in geom["features"]:
        if feature["id"] == field_id:
            return feature["properties"]
    return None


def serialize_numpy(obj):
    """JSON serializer for numpy types."""
    if isinstance(obj, (np.integer, np.floating)):
        return float(obj)
    elif isinstance(obj, np.ndarray):
        return obj.tolist()
    raise TypeError(f"Type {type(obj)} not serializable")


def format_shap_explanation(top_features):
    """
    Create a natural language explanation from SHAP values.
    
    Args:
        top_features: list of dicts with 'name' and 'shap_value' keys
    
    Returns:
        str: one-sentence explanation
    """
    if not top_features:
        return "Unable to generate explanation."
    
    feature_names = [f["name"] for f in top_features[:3]]
    
    # Template-based explanations
    templates = {
        "ndvi_change_pct": "rapid NDVI decline",
        "ndvi_rolling_mean": "persistently low vegetation index",
        "soil_moisture_vol": "low soil moisture",
        "soil_temp_c": "temperature stress",
        "soil_ec_ds_m": "soil salinity",
        "rainfall_proxy": "insufficient rainfall",
        "lag_1_ndvi": "recent vegetation stress",
        "lag_7_ndvi": "week-long degradation"
    }
    
    explanations = [templates.get(f, f) for f in feature_names]
    
    if len(explanations) >= 2:
        return f"Stress driven by {explanations[0]} and {explanations[1]}."
    else:
        return f"Primary stress indicator: {explanations[0]}."


def compute_alert_metrics(y_true, y_pred_proba, y_pred, threshold=0.5):
    """
    Compute classification metrics for stress detection.
    
    Returns dict with precision, recall, f1, specificity, false_alert_rate.
    """
    from sklearn.metrics import precision_score, recall_score, f1_score, confusion_matrix
    
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
    
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0
    specificity = tn / (tn + fp) if (tn + fp) > 0 else 0
    f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
    false_alert_rate = fp / (fp + tn) if (fp + tn) > 0 else 0
    
    return {
        "precision": float(precision),
        "recall": float(recall),
        "f1": float(f1),
        "specificity": float(specificity),
        "false_alert_rate": float(false_alert_rate),
        "confusion_matrix": {
            "tn": int(tn), "fp": int(fp),
            "fn": int(fn), "tp": int(tp)
        }
    }
