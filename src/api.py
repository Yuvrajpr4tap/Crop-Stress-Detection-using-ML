"""
FastAPI service for crop stress detection.
Exposes REST endpoints for prediction, alerts, and health checks.
"""

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
from typing import List, Dict, Optional
import numpy as np
import pandas as pd
import pickle
import json
from pathlib import Path
from datetime import datetime
import logging
import sys

# Ensure modules can be imported
sys.path.insert(0, str(Path(__file__).parent))

from model import StressDetectionModel
from explainability import StressExplainer

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Initialize FastAPI app
app = FastAPI(
    title="Crop Stress Detection API",
    description="Real-time crop stress prediction with explainability",
    version="0.1.0"
)

# Global state
MODEL = None
EXPLAINER = None
PREPROCESSOR = None
FEATURE_NAMES = None
ALERTS = []  # In-memory alert store


# ============== Request/Response Models ==============

class SensorInput(BaseModel):
    """Soil sensor measurement."""
    soil_moisture_vol: float = Field(..., ge=0, le=1)
    soil_temp_c: float = Field(..., ge=-40, le=60)
    soil_ec_ds_m: float = Field(..., ge=0, le=5)


class NDVIInput(BaseModel):
    """NDVI time series point."""
    date: str
    ndvi: float = Field(..., ge=0, le=1)
    quality_flag: str = "good"


class PredictionRequest(BaseModel):
    """Request for stress prediction."""
    field_id: str
    timestamp: str
    ndvi_series: List[NDVIInput]
    sensor_series: List[Dict]  # List of sensor readings


class FeatureExplanation(BaseModel):
    """SHAP feature explanation."""
    name: str
    shap_value: float
    instance_value: float


class PredictionResponse(BaseModel):
    """Stress prediction result."""
    field_id: str
    timestamp: str
    stress_probability: float
    alert: bool
    alert_threshold: float = 0.5
    top_features: List[FeatureExplanation]
    explanation: str
    model_version: str
    latency_ms: float


class AlertRecord(BaseModel):
    """Stored alert record."""
    field_id: str
    timestamp: str
    stress_probability: float
    top_features: List[FeatureExplanation]
    explanation: str
    created_at: str


class HealthResponse(BaseModel):
    """Service health status."""
    status: str
    model_loaded: bool
    model_version: str
    num_features: int
    timestamp: str


# ============== Initialization ==============

def load_artifacts(model_dir="models"):
    """Load trained model, explainer, and preprocessor."""
    global MODEL, EXPLAINER, PREPROCESSOR, FEATURE_NAMES
    
    try:
        import joblib
        model_path = Path(model_dir) / "model.pkl"
        explainer_path = Path(model_dir) / "shap_explainer.pkl"
        features_path = Path(model_dir) / "feature_names.json"
        
        if model_path.exists():
            model_data = joblib.load(model_path)
            MODEL = StressDetectionModel()
            MODEL.model = model_data['model']
            MODEL.feature_names = model_data['feature_names']
            MODEL.metadata = model_data['metadata']
            logger.info("✓ Model loaded")
        
        if explainer_path.exists():
            with open(explainer_path, 'rb') as f:
                EXPLAINER = pickle.load(f)
            logger.info("✓ Explainer loaded")
        
        if features_path.exists():
            with open(features_path, 'r') as f:
                FEATURE_NAMES = json.load(f)
            logger.info(f"✓ Feature names loaded ({len(FEATURE_NAMES)} features)")
    
    except Exception as e:
        logger.error(f"Error loading artifacts: {e}")


@app.on_event("startup")
async def startup():
    """Load models on startup."""
    load_artifacts()


# ============== Endpoints ==============

@app.get("/health", response_model=HealthResponse)
async def health_check():
    """Check service health and model status."""
    return HealthResponse(
        status="healthy" if MODEL else "degraded",
        model_loaded=MODEL is not None,
        model_version="0.1.0",
        num_features=len(FEATURE_NAMES) if FEATURE_NAMES else 0,
        timestamp=datetime.utcnow().isoformat()
    )


@app.post("/predict", response_model=PredictionResponse)
async def predict(request: PredictionRequest):
    """
    Predict crop stress for a field.
    
    Returns stress probability, alert flag, and SHAP explanations.
    """
    if MODEL is None or EXPLAINER is None or FEATURE_NAMES is None:
        raise HTTPException(
            status_code=503,
            detail="Model artifacts not loaded"
        )
    
    start_time = datetime.utcnow()
    
    try:
        # Prepare data: align NDVI and sensor series
        ndvi_data = pd.DataFrame([
            {'date': n.date, 'ndvi': n.ndvi, 'quality_flag': n.quality_flag}
            for n in request.ndvi_series
        ])
        
        sensor_data = pd.DataFrame(request.sensor_series)
        
        # Simple alignment: join on date closest match
        ndvi_data['date'] = pd.to_datetime(ndvi_data['date'])
        sensor_data['date'] = pd.to_datetime(
            sensor_data.get('timestamp', sensor_data.get('date', ''))
        ).dt.date
        ndvi_data['date'] = ndvi_data['date'].dt.date
        
        merged = ndvi_data.merge(sensor_data, on='date', how='inner')
        
        if len(merged) == 0:
            raise ValueError("No aligned NDVI-sensor data found")
        
        # Use most recent measurement
        latest = merged.iloc[-1]
        
        # Create feature vector (simplified - in production would compute all features)
        X_pred = pd.DataFrame([{
            'ndvi': latest['ndvi'],
            'soil_moisture_vol': latest.get('soil_moisture_vol', 0.3),
            'soil_temp_c': latest.get('soil_temp_c', 20),
            'soil_ec_ds_m': latest.get('soil_ec_ds_m', 0.3),
            'ndvi_rolling_mean_7d': merged['ndvi'].tail(7).mean(),
            'ndvi_rolling_std_7d': merged['ndvi'].tail(7).std(),
            'lag_1_ndvi': merged['ndvi'].iloc[-2] if len(merged) > 1 else latest['ndvi'],
            'lag_7_ndvi': merged['ndvi'].iloc[-7] if len(merged) > 7 else latest['ndvi'],
            'ndvi_change_pct': (merged['ndvi'].iloc[-1] - merged['ndvi'].iloc[-2]) / (merged['ndvi'].iloc[-2] + 1e-6) * 100 if len(merged) > 1 else 0,
            'ndvi_change_from_7d_ago': merged['ndvi'].iloc[-1] - (merged['ndvi'].iloc[-7] if len(merged) > 7 else merged['ndvi'].iloc[0]),
            'soil_moisture_deficit': max(0, 0.4 - latest.get('soil_moisture_vol', 0.3)),
            'rainfall_proxy': merged['soil_moisture_vol'].tail(7).mean() if 'soil_moisture_vol' in merged.columns else 0.3,
            'extreme_heat_proxy': 1 if latest.get('soil_temp_c', 0) > 28 else 0,
            'moisture_temp_interaction': latest.get('soil_moisture_vol', 0.3) * (1 / (1 + latest.get('soil_temp_c', 20)))
        }])
        
        # Ensure feature order matches training
        X_pred = X_pred[FEATURE_NAMES]
        
        # Predict
        stress_prob = float(MODEL.predict(X_pred, num_iteration=MODEL.best_iteration)[0])
        alert = stress_prob > 0.5
        
        # Explain
        explanation = EXPLAINER.explain_prediction(X_pred, top_k=3)
        top_features_list = [
            FeatureExplanation(**feat)
            for feat in explanation['top_features']
        ]
        
        # Generate text explanation
        from src.utils import format_shap_explanation
        explanation_text = format_shap_explanation(explanation['top_features'])
        
        # Compute latency
        latency_ms = (datetime.utcnow() - start_time).total_seconds() * 1000
        
        response = PredictionResponse(
            field_id=request.field_id,
            timestamp=request.timestamp,
            stress_probability=stress_prob,
            alert=alert,
            top_features=top_features_list,
            explanation=explanation_text,
            model_version="0.1.0",
            latency_ms=latency_ms
        )
        
        # Store alert if triggered
        if alert:
            ALERTS.append(AlertRecord(
                field_id=request.field_id,
                timestamp=request.timestamp,
                stress_probability=stress_prob,
                top_features=top_features_list,
                explanation=explanation_text,
                created_at=datetime.utcnow().isoformat()
            ))
        
        return response
    
    except Exception as e:
        logger.error(f"Prediction error: {e}")
        raise HTTPException(status_code=400, detail=str(e))


@app.get("/fields/{field_id}/alerts", response_model=Dict)
async def get_alerts(field_id: str, limit: int = 10):
    """Get recent alerts for a field."""
    field_alerts = [
        alert for alert in ALERTS
        if alert.field_id == field_id
    ][-limit:]
    
    return {
        "field_id": field_id,
        "num_alerts": len(field_alerts),
        "alerts": [alert.dict() for alert in field_alerts]
    }


@app.get("/")
async def root():
    """API welcome message."""
    return {
        "name": "Crop Stress Detection API",
        "version": "0.1.0",
        "docs": "/docs",
        "health": "/health",
        "endpoints": [
            "POST /predict",
            "GET /fields/{field_id}/alerts",
            "GET /health"
        ]
    }


# ============== Canned Example for Testing ==============

EXAMPLE_REQUEST = PredictionRequest(
    field_id="field_001",
    timestamp="2023-07-15T12:00:00Z",
    ndvi_series=[
        NDVIInput(date="2023-07-08", ndvi=0.72),
        NDVIInput(date="2023-07-12", ndvi=0.68),
        NDVIInput(date="2023-07-15", ndvi=0.62),
    ],
    sensor_series=[
        {
            "timestamp": "2023-07-15T12:00:00Z",
            "soil_moisture_vol": 0.22,
            "soil_temp_c": 32,
            "soil_ec_ds_m": 0.35
        }
    ]
)


@app.get("/example")
async def example():
    """Return example request for testing."""
    return {
        "description": "Example prediction request for testing",
        "example_request": EXAMPLE_REQUEST.dict()
    }


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
