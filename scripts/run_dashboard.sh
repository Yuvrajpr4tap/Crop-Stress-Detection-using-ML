#!/bin/bash
# Run the Streamlit dashboard

set -e

echo "📊 Starting Crop Stress Detection Dashboard..."
echo ""
echo "📍 Dashboard will be available at: http://localhost:8501"
echo ""

streamlit run dashboard/app.py
