# Multi-stage build for crop stress detection

FROM python:3.10-slim as base

WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements
COPY requirements.txt .

# Install Python dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Copy application code
COPY . .

# Stage 1: API Service
FROM base as api

EXPOSE 8000

CMD ["python", "-m", "uvicorn", "src.api:app", "--host", "0.0.0.0", "--port", "8000"]

# Stage 2: Dashboard Service
FROM base as dashboard

EXPOSE 8501

CMD ["streamlit", "run", "dashboard/app.py", "--server.port=8501", "--server.address=0.0.0.0"]

# Stage 3: Training/Notebook Service
FROM base as training

EXPOSE 8888

RUN pip install --no-cache-dir jupyter

CMD ["jupyter", "notebook", "--ip=0.0.0.0", "--allow-root", "--no-browser"]

# Default: API
FROM api
