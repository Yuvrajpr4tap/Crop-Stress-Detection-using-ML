"""
Synthetic data generator for crop stress detection.
Creates 3 sample fields with NDVI time series and soil sensor data.
Includes synthetic stress labels for training.
"""

import json
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
from pathlib import Path


def generate_field_geometries(output_path="sample_fields.geojson"):
    """Generate 3 sample field polygons as GeoJSON."""
    fields = {
        "type": "FeatureCollection",
        "features": [
            {
                "type": "Feature",
                "id": "field_001",
                "properties": {
                    "name": "North Field",
                    "crop": "corn",
                    "area_ha": 42.5,
                    "elevation_m": 245
                },
                "geometry": {
                    "type": "Polygon",
                    "coordinates": [[
                        [-93.5, 40.5],
                        [-93.48, 40.5],
                        [-93.48, 40.52],
                        [-93.5, 40.52],
                        [-93.5, 40.5]
                    ]]
                }
            },
            {
                "type": "Feature",
                "id": "field_002",
                "properties": {
                    "name": "South Field",
                    "crop": "soybeans",
                    "area_ha": 38.2,
                    "elevation_m": 230
                },
                "geometry": {
                    "type": "Polygon",
                    "coordinates": [[
                        [-93.52, 40.45],
                        [-93.50, 40.45],
                        [-93.50, 40.47],
                        [-93.52, 40.47],
                        [-93.52, 40.45]
                    ]]
                }
            },
            {
                "type": "Feature",
                "id": "field_003",
                "properties": {
                    "name": "East Field",
                    "crop": "corn",
                    "area_ha": 35.8,
                    "elevation_m": 250
                },
                "geometry": {
                    "type": "Polygon",
                    "coordinates": [[
                        [-93.47, 40.48],
                        [-93.45, 40.48],
                        [-93.45, 40.50],
                        [-93.47, 40.50],
                        [-93.47, 40.48]
                    ]]
                }
            }
        ]
    }
    
    with open(output_path, "w") as f:
        json.dump(fields, f, indent=2)
    print(f"✓ Field geometries saved to {output_path}")


def generate_ndvi_timeseries(
    output_path="sample_ndvi_timeseries.csv",
    start_date="2023-04-01",
    end_date="2023-10-31",
    freq_days=8
):
    """Generate synthetic NDVI time series for 3 fields."""
    
    field_ids = ["field_001", "field_002", "field_003"]
    dates = pd.date_range(start=start_date, end=end_date, freq=f"{freq_days}D")
    
    records = []
    np.random.seed(42)
    
    for field_id in field_ids:
        # Base NDVI trajectory (growing season ramp-up then decline)
        doy = np.array([(d - datetime.strptime(start_date, "%Y-%m-%d")).days for d in dates])
        base_ndvi = 0.3 + 0.4 * np.sin(np.pi * doy / 214)  # Growing season
        
        # Field-specific patterns
        if field_id == "field_001":
            # Stress event mid-season (drought)
            stress_mask = (doy > 80) & (doy < 130)
            ndvi = base_ndvi.copy()
            ndvi[stress_mask] -= 0.15 + 0.05 * np.random.randn(stress_mask.sum())
        elif field_id == "field_002":
            # Mild stress 
            stress_mask = (doy > 100) & (doy < 110)
            ndvi = base_ndvi.copy()
            ndvi[stress_mask] -= 0.08 + 0.03 * np.random.randn(stress_mask.sum())
        else:
            # Healthy field, minimal stress
            ndvi = base_ndvi + 0.02 * np.random.randn(len(base_ndvi))
        
        # Clip to valid NDVI range
        ndvi = np.clip(ndvi, 0.0, 1.0)
        
        for date, val in zip(dates, ndvi):
            records.append({
                "field_id": field_id,
                "date": date.strftime("%Y-%m-%d"),
                "ndvi": float(val),
                "quality_flag": "good" if np.random.rand() > 0.05 else "cloudy"
            })
    
    df = pd.DataFrame(records)
    df.to_csv(output_path, index=False)
    print(f"✓ NDVI time series saved to {output_path} ({len(df)} records)")


def generate_soil_sensor_data(
    output_path="sample_soil_sensors.csv",
    start_date="2023-04-01",
    end_date="2023-10-31",
    freq_hours=6
):
    """Generate synthetic soil sensor measurements (moisture, temp, EC)."""
    
    field_ids = ["field_001", "field_002", "field_003"]
    dates = pd.date_range(start=start_date, end=end_date, freq=f"{freq_hours}H")
    
    records = []
    np.random.seed(42)
    
    for field_id in field_ids:
        doy = np.array([(d - datetime.strptime(start_date, "%Y-%m-%d")).days for d in dates])
        
        # Seasonal moisture pattern
        base_moisture = 0.35 + 0.15 * np.sin(np.pi * doy / 214)
        
        # Stress-linked moisture deficits
        if field_id == "field_001":
            stress_mask = (doy > 80) & (doy < 130)
            moisture = base_moisture - 0.12 * stress_mask
        elif field_id == "field_002":
            stress_mask = (doy > 100) & (doy < 110)
            moisture = base_moisture - 0.08 * stress_mask
        else:
            moisture = base_moisture
        
        # Add noise
        moisture += 0.02 * np.random.randn(len(moisture))
        moisture = np.clip(moisture, 0.05, 0.5)
        
        # Temperature (seasonal cycle)
        temperature = 15 + 12 * np.sin(np.pi * doy / 214) + 3 * np.random.randn(len(moisture))
        
        # Electrical conductivity (soil salinity proxy)
        ec = 0.3 + 0.05 * np.random.randn(len(moisture))
        ec = np.clip(ec, 0.1, 0.8)
        
        for date, moist, temp, ec_val in zip(dates, moisture, temperature, ec):
            records.append({
                "field_id": field_id,
                "timestamp": date.strftime("%Y-%m-%d %H:%M:%S"),
                "soil_moisture_vol": float(moist),
                "soil_temp_c": float(temp),
                "soil_ec_ds_m": float(ec_val),
                "sensor_id": f"{field_id}_{np.random.choice([1, 2, 3])}"
            })
    
    df = pd.DataFrame(records)
    df.to_csv(output_path, index=False)
    print(f"✓ Soil sensor data saved to {output_path} ({len(df)} records)")


def generate_labels(
    ndvi_path="sample_ndvi_timeseries.csv",
    output_path="sample_labels.csv"
):
    """
    Generate synthetic stress labels based on NDVI thresholds.
    Label = 1 if NDVI drops > 0.10 over a 2-week window, else 0.
    """
    df = pd.read_csv(ndvi_path)
    df["date"] = pd.to_datetime(df["date"])
    df = df.sort_values(["field_id", "date"]).reset_index(drop=True)
    
    labels = []
    for field_id in df["field_id"].unique():
        field_data = df[df["field_id"] == field_id].reset_index(drop=True)
        
        for i in range(len(field_data)):
            current_date = field_data.loc[i, "date"]
            current_ndvi = field_data.loc[i, "ndvi"]
            
            # Look back 2 weeks
            past_window = field_data[
                (field_data["date"] >= current_date - timedelta(days=14)) &
                (field_data["date"] < current_date)
            ]
            
            # Stress = significant NDVI drop + quality good
            stress_label = 0
            if field_data.loc[i, "quality_flag"] == "good" and len(past_window) > 0:
                past_ndvi = past_window[past_window["quality_flag"] == "good"]["ndvi"]
                if len(past_ndvi) > 0:
                    ndvi_drop = past_ndvi.max() - current_ndvi
                    stress_label = 1 if ndvi_drop > 0.10 else 0
            
            labels.append({
                "field_id": field_id,
                "date": current_date.strftime("%Y-%m-%d"),
                "stress_label": stress_label
            })
    
    label_df = pd.DataFrame(labels)
    label_df.to_csv(output_path, index=False)
    print(f"✓ Stress labels saved to {output_path}")
    print(f"  Label distribution: {label_df['stress_label'].value_counts().to_dict()}")


if __name__ == "__main__":
    print("Generating synthetic crop stress detection dataset...\n")
    
    generate_field_geometries()
    generate_ndvi_timeseries()
    generate_soil_sensor_data()
    generate_labels()
    
    print("\n✓ All synthetic data generated successfully!")
    print("  - sample_fields.geojson (3 field polygons)")
    print("  - sample_ndvi_timeseries.csv (NDVI time series)")
    print("  - sample_soil_sensors.csv (soil measurements)")
    print("  - sample_labels.csv (stress labels)")
