#!/bin/bash
# Script to run MLOps services locally

echo "Starting MLOps Housing Price Prediction Services..."

# Create necessary directories
mkdir -p logs data/{raw,processed,features} model/{artifacts,checkpoints,configs}

# Start predict service in background
echo "Starting predict service..."
python -m uvicorn services.predict.main:app --host 0.0.0.0 --port 8001 &
PREDICT_PID=$!

# Wait a moment for predict service to start
sleep 5

# Start API service
echo "Starting API service..."
python -m uvicorn services.api.main:app --host 0.0.0.0 --port 8000 &
API_PID=$!

echo "Services started!"
echo "API Service: http://localhost:8000"
echo "Predict Service: http://localhost:8001"
echo "API Documentation: http://localhost:8000/docs"
echo "Predict Documentation: http://localhost:8001/docs"
echo ""
echo "To stop services, run: kill $API_PID $PREDICT_PID"

# Wait for services
wait