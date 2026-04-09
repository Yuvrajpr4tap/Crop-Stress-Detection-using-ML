#!/bin/bash
# Run the FastAPI service

set -e

echo "🚀 Starting Crop Stress Detection API..."
echo ""
echo "📍 API will be available at: http://localhost:8000"
echo "📖 Swagger docs at: http://localhost:8000/docs"
echo ""

python -m uvicorn src.api:app --host 0.0.0.0 --port 8000 --reload
