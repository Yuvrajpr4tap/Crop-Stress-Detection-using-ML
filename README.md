# 🌾 Crop Stress Detection – End-to-End AI Prototype

A production-ready ML system that combines satellite NDVI time series and in-field soil sensor data to detect crop stress events in real-time. Includes REST API, interactive dashboard, SHAP explainability, and comprehensive documentation.

## Quick Start

### Prerequisites
- Python 3.10+
- pip or conda

### 1. Clone & Setup

```bash
cd crop-stress-detection
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
pip install -r requirements.txt
```

### 2. Generate Synthetic Data

```bash
python data/synthetic_data_generator.py
```

This creates:
- `data/sample_fields.geojson` – 3 field polygons
- `data/sample_ndvi_timeseries.csv` – NDVI time series
- `data/sample_soil_sensors.csv` – Soil sensor measurements
- `data/sample_labels.csv` – Stress labels

### 3. Train the Model

```bash
bash scripts/train.sh
```

This:
- Loads and preprocesses data
- Engineers features (rolling means, lags, interactions)
- Trains LightGBM with class weighting
- Builds SHAP explainer
- Saves artifacts to `models/`

**Expected Training Time:** ~2 minutes on a typical laptop

### 4. Start the API

```bash
bash scripts/run_api.sh
```

API will be available at `http://localhost:8000`
- Swagger docs: `http://localhost:8000/docs`
- Example: `http://localhost:8000/example`

### 5. Open the Dashboard

In a new terminal:

```bash
bash scripts/run_dashboard.sh
```

Dashboard available at `http://localhost:8501`

## Project Structure

```
crop-stress-detection/
├── data/                          # Data layer
│   ├── synthetic_data_generator.py
│   ├── sample_fields.geojson
│   ├── sample_ndvi_timeseries.csv
│   └── sample_soil_sensors.csv
│
├── notebooks/                     # EDA & modeling
│   ├── 01_eda_and_feature_engineering.ipynb
│   └── 02_model_training_and_evaluation.ipynb
│
├── src/                           # Core ML pipeline
│   ├── preprocessing.py           # Feature engineering
│   ├── model.py                   # LightGBM wrapper
│   ├── explainability.py          # SHAP integration
│   ├── api.py                     # FastAPI service
│   └── utils.py                   # Helpers
│
├── dashboard/                     # Streamlit visualization
│   └── app.py
│
├── tests/                         # Unit & integration tests
│   ├── test_preprocessing.py
│   ├── test_model.py
│   └── test_api.py
│
├── models/                        # Trained artifacts
│   ├── model.pkl                  # LightGBM
│   ├── shap_explainer.pkl         # SHAP explainer
│   ├── preprocessor.pkl           # Feature scaler
│   └── feature_names.json         # Feature list
│
├── Dockerfile                     # Container setup
├── requirements.txt
├── README.md                      # This file
├── design_doc.md                  # Architecture & design
├── demo_script.md                 # 2-minute demo guide
├── interview_talking_points.md    # Interview Q&A
└── evaluation_report.md           # Performance metrics
```

## API Reference

### POST `/predict`

**Request:**
```json
{
  "field_id": "field_001",
  "timestamp": "2023-07-15T12:00:00Z",
  "ndvi_series": [
    {"date": "2023-07-08", "ndvi": 0.72, "quality_flag": "good"},
    {"date": "2023-07-12", "ndvi": 0.68, "quality_flag": "good"},
    {"date": "2023-07-15", "ndvi": 0.62, "quality_flag": "good"}
  ],
  "sensor_series": [
    {
      "timestamp": "2023-07-15T12:00:00Z",
      "soil_moisture_vol": 0.22,
      "soil_temp_c": 32,
      "soil_ec_ds_m": 0.35
    }
  ]
}
```

**Response:**
```json
{
  "field_id": "field_001",
  "timestamp": "2023-07-15T12:00:00Z",
  "stress_probability": 0.72,
  "alert": true,
  "alert_threshold": 0.5,
  "top_features": [
    {
      "name": "ndvi_change_pct",
      "shap_value": -0.18,
      "instance_value": -8.8
    },
    {
      "name": "soil_moisture_vol",
      "shap_value": -0.12,
      "instance_value": 0.22
    },
    {
      "name": "lag_7_ndvi",
      "shap_value": -0.08,
      "instance_value": 0.71
    }
  ],
  "explanation": "Stress driven by rapid NDVI decline and low soil moisture. Consider irrigation or pest management actions.",
  "model_version": "0.1.0",
  "latency_ms": 45.2
}
```

### GET `/health`

Returns service and model status:
```json
{
  "status": "healthy",
  "model_loaded": true,
  "model_version": "0.1.0",
  "num_features": 14,
  "timestamp": "2024-04-07T12:00:00Z"
}
```

### GET `/fields/{field_id}/alerts`

Returns recent alerts for a field:
```json
{
  "field_id": "field_001",
  "num_alerts": 3,
  "alerts": [...]
}
```

## Dashboard Features

1. **Field Map** – Visual overview of all fields colored by stress probability
2. **Field Details** – NDVI & sensor time series per field with stress timeline
3. **Alerts Panel** – Active stress alerts with one-sentence explanations
4. **Model Info** – Feature list, performance metrics, and acceptance criteria

## Running Tests

```bash
# Run all tests
pytest

# Run specific test file
pytest tests/test_preprocessing.py

# With coverage report
pytest --cov=src --cov-report=html
```

**Expected Coverage:** >85% on core modules

## Running Docker

```bash
# Build images
bash scripts/build_docker.sh

# Run API
docker run -p 8000:8000 crop-stress-detection:api-0.1.0

# Run Dashboard
docker run -p 8501:8501 crop-stress-detection:dashboard-0.1.0
```

## Key Features

### Data Pipeline
- Synthetic dataset generator (3 fields, 7 months of data)
- NDVI preprocessing (quality filtering, temporal alignment)
- Soil sensor aggregation & imputation
- Multimodal data alignment with forward-fill

### Feature Engineering
- Rolling statistics (7d, 14d windows)
- Lag features (t-1, t-3, t-7 days)
- Derived features (NDVI change %, moisture deficit)
- Interaction terms (moisture × temperature)

### Modeling
- **Algorithm:** LightGBM with class weighting
- **Validation:** Field-stratified + temporal cross-validation
- **Training Time:** <2 min on CPU
- **Inference:** <50ms per prediction

### Explainability
- SHAP TreeExplainer for local explanations
- Top-3 feature contributions per prediction
- Natural language explanation templates
- Global feature importance plots

### API & Deployment
- FastAPI with input validation
- Error handling & health checks
- Canned example for CI testing
- Docker multi-stage build

### Visualization
- Real-time dashboard (Streamlit)
- Interactive NDVI/sensor plots (Plotly)
- Alert timeline with recommendations
- SHAP value explanations

## Performance

| Metric | Value |
|--------|-------|
| Precision | 0.62–0.68 |
| Recall | 0.52–0.58 |
| F1 Score | 0.57–0.63 |
| AUC-ROC | 0.72–0.78 |
| Inference Latency | 35–50ms |
| Memory (Model) | ~15MB |

> See `evaluation_report.md` for detailed results

## Data Sources & Limitations

### Data
- **NDVI:** Synthetic time series (Sentinel-2 style)
- **Sensors:** Synthetic soil measurements (moisture, temp, EC)
- **Labels:** Synthetic stress events (NDVI drop > 0.10 over 2 weeks)

