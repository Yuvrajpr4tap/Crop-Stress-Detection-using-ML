#!/bin/bash
# Build and run Docker containers for crop stress detection

set -e

PROJECT_NAME="crop-stress-detection"
VERSION="0.1.0"

echo "🐳 Building crop stress detection Docker image..."

# Build API image
docker build -t ${PROJECT_NAME}:api-${VERSION} --target api .
echo "✓ API image built: ${PROJECT_NAME}:api-${VERSION}"

# Build Dashboard image
docker build -t ${PROJECT_NAME}:dashboard-${VERSION} --target dashboard .
echo "✓ Dashboard image built: ${PROJECT_NAME}:dashboard-${VERSION}"

# Build Training image
docker build -t ${PROJECT_NAME}:training-${VERSION} --target training .
echo "✓ Training image built: ${PROJECT_NAME}:training-${VERSION}"

echo ""
echo "🚀 To run:"
echo "  API:       docker run -p 8000:8000 ${PROJECT_NAME}:api-${VERSION}"
echo "  Dashboard: docker run -p 8501:8501 ${PROJECT_NAME}:dashboard-${VERSION}"
echo "  Training:  docker run -p 8888:8888 ${PROJECT_NAME}:training-${VERSION}"
echo ""
echo "📦 Or use docker-compose:"
echo "  docker-compose up"
