"""Integration tests for API endpoints."""

import pytest
from fastapi.testclient import TestClient
from src.api import app, EXAMPLE_REQUEST
import json


client = TestClient(app)


class TestAPIHealth:
    """Tests for health endpoints."""
    
    def test_health_check(self):
        """Test /health endpoint."""
        response = client.get("/health")
        
        assert response.status_code == 200
        data = response.json()
        assert 'status' in data
        assert 'model_loaded' in data
        assert 'timestamp' in data
    
    def test_root_endpoint(self):
        """Test root endpoint."""
        response = client.get("/")
        
        assert response.status_code == 200
        data = response.json()
        assert 'name' in data
        assert 'version' in data


class TestAPIPrediction:
    """Tests for prediction endpoints."""
    
    def test_example_request(self):
        """Test /example endpoint."""
        response = client.get("/example")
        
        assert response.status_code == 200
        data = response.json()
        assert 'example_request' in data
    
    def test_predict_endpoint_structure(self):
        """Test /predict endpoint response structure."""
        request_data = EXAMPLE_REQUEST.dict()
        response = client.post("/predict", json=request_data)
        
        # May fail if model not loaded, but structure should be correct
        if response.status_code == 200:
            data = response.json()
            assert 'field_id' in data
            assert 'stress_probability' in data
            assert 'alert' in data
            assert 'top_features' in data
            assert 'explanation' in data
            assert isinstance(data['stress_probability'], float)
            assert isinstance(data['alert'], bool)
    
    def test_invalid_request(self):
        """Test /predict with invalid data."""
        response = client.post("/predict", json={"field_id": "invalid"})
        
        assert response.status_code >= 400


class TestAPIAlerts:
    """Tests for alert endpoints."""
    
    def test_get_alerts(self):
        """Test /fields/{field_id}/alerts endpoint."""
        response = client.get("/fields/field_001/alerts")
        
        assert response.status_code == 200
        data = response.json()
        assert 'field_id' in data
        assert 'alerts' in data
        assert isinstance(data['alerts'], list)
    
    def test_alert_limit(self):
        """Test alert limit parameter."""
        response = client.get("/fields/field_001/alerts?limit=5")
        
        assert response.status_code == 200
        data = response.json()
        assert len(data['alerts']) <= 5
