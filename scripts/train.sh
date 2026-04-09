#!/bin/bash
# Train the crop stress detection model

set -e

echo "🌾 Crop Stress Detection - Model Training"
echo "=========================================="
echo ""

# Generate synthetic data if needed
if [ ! -f "data/sample_fields.geojson" ]; then
    echo "📊 Generating synthetic data..."
    python data/synthetic_data_generator.py
    echo ""
fi

# Run training notebook via Python script
echo "🤖 Training model..."
python << 'EOF'
import sys
from pathlib import Path
import pandas as pd
import numpy as np
import pickle
import json
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer

# Add src to path
sys.path.insert(0, str(Path.cwd() / 'src'))

from preprocessing import (
    NDVIPreprocessor, SensorPreprocessor, FeatureEngineer,
    align_multimodal_data, select_features, prepare_train_test_split
)
from model import StressDetectionModel
from explainability import StressExplainer

# Load data
print("Loading data...")
ndvi_df = pd.read_csv("data/sample_ndvi_timeseries.csv")
sensor_df = pd.read_csv("data/sample_soil_sensors.csv")
labels_df = pd.read_csv("data/sample_labels.csv")

# Preprocess
print("Preprocessing...")
ndvi_prep = NDVIPreprocessor()
ndvi_df = ndvi_prep.transform(ndvi_df)

sensor_prep = SensorPreprocessor(agg_freq='1D')
sensor_df = sensor_prep.transform(sensor_df)

# Align data
print("Aligning multimodal data...")
df = align_multimodal_data(ndvi_df, sensor_df, labels_df)

# Feature engineering
print("Engineering features...")
engineer = FeatureEngineer()
df = engineer.transform(df)

# Select features
feature_cols = select_features(df)
print(f"Selected {len(feature_cols)} features")

# Train/val/test split
print("Preparing train/val/test splits...")
train_df, val_df, test_df = prepare_train_test_split(df, test_size=0.2, val_size=0.1)

# Prepare modeling data
X_train = train_df[feature_cols].fillna(0)
y_train = train_df['stress_label']
X_val = val_df[feature_cols].fillna(0)
y_val = val_df['stress_label']
X_test = test_df[feature_cols].fillna(0)
y_test = test_df['stress_label']

# Scale features
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_val_scaled = scaler.transform(X_val)
X_test_scaled = scaler.transform(X_test)

X_train_scaled = pd.DataFrame(X_train_scaled, columns=feature_cols)
X_val_scaled = pd.DataFrame(X_val_scaled, columns=feature_cols)
X_test_scaled = pd.DataFrame(X_test_scaled, columns=feature_cols)

# Train model
print("\nTraining LightGBM model...")
stress_model = StressDetectionModel()
stress_model.train(X_train_scaled, y_train, X_val_scaled, y_val, verbose=True)

# Evaluate
print("\n" + "="*50)
print("TEST SET EVALUATION")
print("="*50)
test_metrics = stress_model.evaluate(X_test_scaled, y_test)
for metric, value in test_metrics.items():
    if metric != 'confusion_matrix':
        print(f"{metric:<20}: {value:.4f}")
    else:
        cm = value
        print(f"\nConfusion Matrix:")
        print(f"  TN={cm['tn']}, FP={cm['fp']}")
        print(f"  FN={cm['fn']}, TP={cm['tp']}")

# Build SHAP explainer
print("\nBuilding SHAP explainer...")
explainer = StressExplainer(stress_model.model, X_val_scaled, feature_cols)

# Save artifacts
print("\nSaving artifacts...")
models_dir = Path("models")
models_dir.mkdir(exist_ok=True)

stress_model.save(models_dir / "model.pkl")
explainer.save(models_dir / "shap_explainer.pkl")

with open(models_dir / "preprocessor.pkl", 'wb') as f:
    pickle.dump(scaler, f)

with open(models_dir / "feature_names.json", 'w') as f:
    json.dump(feature_cols, f)

print("\n✓ Model training complete!")
print(f"  - Model saved to models/model.pkl")
print(f"  - Explainer saved to models/shap_explainer.pkl")
print(f"  - Features saved to models/feature_names.json")

EOF

echo ""
echo "✓ Training complete! Run the API with:"
echo "  python -m uvicorn src.api:app --reload"
