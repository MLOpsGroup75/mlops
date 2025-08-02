# Multi-stage Dockerfile for MLOps services
FROM python:3.11-slim AS base

# Set environment variables
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PYTHONPATH=/app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    build-essential \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Create non-root user
RUN groupadd -r mlops && useradd -r -g mlops mlops

# Set work directory
WORKDIR /app

# Copy requirements and install dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt

# Copy application code
COPY . .

# Create necessary directories
RUN mkdir -p logs data/raw data/processed data/features model/artifacts model/checkpoints model/configs pipeline/training pipeline/inference pipeline/monitoring

# Set ownership
RUN chown -R mlops:mlops /app

# Switch to non-root user
USER mlops

# Expose ports (will be overridden in docker-compose)
EXPOSE 8000 8001

# Default command (will be overridden in docker-compose)
CMD ["python", "-m", "services.api.main"]