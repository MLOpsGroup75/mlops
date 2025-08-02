import time
import uuid
import numpy as np
import pandas as pd
from datetime import datetime
from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import PlainTextResponse
import structlog

from config.settings import settings
from services.common.models import (
    InternalPredictionRequest, InternalPredictionResponse,
    HealthResponse
)
from services.common.logging_config import setup_logging, get_logger
from services.common.middleware import LoggingMiddleware, MetricsMiddleware
from services.common.metrics import get_predict_metrics


# Initialize logging
setup_logging("predict-service")
logger = get_logger("predict-service")

# Create FastAPI app
app = FastAPI(
    title="MLOps Housing Price Prediction Service",
    description="Internal service for housing price prediction using ML models",
    version="1.0.0",
    docs_url="/docs",
    redoc_url="/redoc"
)

# Add middleware
app.add_middleware(LoggingMiddleware, service_name="predict")
app.add_middleware(MetricsMiddleware, metrics_collector=get_predict_metrics())

# Global variables for model
model = None
feature_columns = [
    'longitude', 'latitude', 'housing_median_age', 'total_rooms',
    'total_bedrooms', 'population', 'households', 'median_income',
    'median_house_value', 'ocean_proximity'
]


class DummyHousingModel:
    """Dummy housing price prediction model for demonstration"""
    
    def __init__(self, accuracy: float = 0.85):
        self.accuracy = accuracy
        self.feature_weights = {
            'longitude': -2000,
            'latitude': 1500,
            'housing_median_age': -300,
            'total_rooms': 50,
            'total_bedrooms': -100,
            'population': 5,
            'households': 100,
            'median_income': 40000,
            'median_house_value': 0.8,
            'ocean_proximity': {
                '<1H OCEAN': 50000,
                'INLAND': -20000,
                'ISLAND': 100000,
                'NEAR BAY': 30000,
                'NEAR OCEAN': 40000
            }
        }
        self.base_price = 200000
    
    def predict(self, features: dict) -> float:
        """Predict housing price using simple weighted sum"""
        price = self.base_price
        
        for feature, value in features.items():
            if feature == 'ocean_proximity':
                price += self.feature_weights[feature].get(value, 0)
            elif feature in self.feature_weights:
                price += self.feature_weights[feature] * value
        
        # Add some randomness to simulate model uncertainty
        noise = np.random.normal(0, price * 0.1)
        price += noise
        
        # Ensure positive price
        return max(price, 50000)
    
    def get_accuracy(self) -> float:
        """Get model accuracy"""
        return self.accuracy


def load_model():
    """Load the ML model (dummy implementation)"""
    global model
    
    try:
        # In a real implementation, you would load from settings.model_path
        # For demo purposes, we'll use a dummy model
        logger.info("Loading housing price prediction model", service="predict")
        
        model = DummyHousingModel(accuracy=settings.model_accuracy or 0.85)
        
        logger.info(
            "Model loaded successfully",
            model_type="DummyHousingModel",
            accuracy=model.get_accuracy(),
            service="predict"
        )
        
        return True
        
    except Exception as e:
        logger.error("Failed to load model", error=str(e), service="predict")
        return False


@app.on_event("startup")
async def startup_event():
    """Initialize services on startup"""
    success = load_model()
    if success:
        get_predict_metrics().set_service_status(True)
        logger.info("Predict service started", service="predict", version=settings.service_version)
    else:
        get_predict_metrics().set_service_status(False)
        logger.error("Failed to start predict service - model loading failed", service="predict")


@app.on_event("shutdown")
async def shutdown_event():
    """Cleanup on shutdown"""
    predict_metrics.set_service_status(False)
    logger.info("Predict service stopped", service="predict")


def prepare_features(request: InternalPredictionRequest) -> dict:
    """Prepare features from request for model prediction"""
    features = {
        'longitude': request.longitude,
        'latitude': request.latitude,
        'housing_median_age': request.housing_median_age,
        'total_rooms': request.total_rooms,
        'total_bedrooms': request.total_bedrooms,
        'population': request.population,
        'households': request.households,
        'median_income': request.median_income,
        'median_house_value': request.median_house_value,
        'ocean_proximity': request.ocean_proximity
    }
    return features


@app.post("/predict", response_model=InternalPredictionResponse)
async def predict_housing_price(
    request: Request,
    prediction_request: InternalPredictionRequest
) -> InternalPredictionResponse:
    """Internal predict endpoint for housing price prediction"""
    
    request_id = prediction_request.request_id or str(uuid.uuid4())
    start_time = time.time()
    
    logger.info(
        "Processing prediction request",
        request_id=request_id,
        service="predict"
    )
    
    if model is None:
        logger.error("Model not loaded", request_id=request_id, service="predict")
        raise HTTPException(
            status_code=503,
            detail="Model not available"
        )
    
    try:
        # Prepare features
        features = prepare_features(prediction_request)
        
        # Make prediction
        prediction_start = time.time()
        housing_price = model.predict(features)
        prediction_time = time.time() - prediction_start
        
        # Calculate total processing time
        total_time = time.time() - start_time
        
        # Record metrics
        get_predict_metrics().record_prediction(total_time, "success")
        
        # Create response
        response = InternalPredictionResponse(
            housing_price=float(housing_price),
            accuracy=model.get_accuracy(),
            request_id=request_id,
            processing_time=total_time
        )
        
        logger.info(
            "Prediction completed successfully",
            request_id=request_id,
            housing_price=housing_price,
            processing_time=total_time,
            prediction_time=prediction_time,
            service="predict"
        )
        
        return response
        
    except Exception as e:
        total_time = time.time() - start_time
        get_predict_metrics().record_prediction(total_time, "error")
        
        logger.error(
            "Error during prediction",
            request_id=request_id,
            error=str(e),
            processing_time=total_time,
            service="predict"
        )
        
        raise HTTPException(
            status_code=500,
            detail=f"Prediction failed: {str(e)}"
        )


@app.get("/metrics", response_class=PlainTextResponse)
async def get_metrics():
    """Export Prometheus metrics"""
    return get_predict_metrics().export_metrics()


@app.get("/health", response_model=HealthResponse)
async def health_check():
    """Health check endpoint"""
    model_status = "OK" if model is not None else "Model not loaded"
    
    return HealthResponse(
        status=model_status,
        timestamp=datetime.utcnow().isoformat(),
        service="predict"
    )


@app.get("/model/info")
async def get_model_info():
    """Get information about the loaded model"""
    if model is None:
        raise HTTPException(status_code=503, detail="Model not loaded")
    
    return {
        "model_type": type(model).__name__,
        "accuracy": model.get_accuracy(),
        "features": feature_columns,
        "loaded_at": datetime.utcnow().isoformat()
    }


@app.post("/model/reload")
async def reload_model():
    """Reload the model"""
    success = load_model()
    if success:
        return {"status": "Model reloaded successfully"}
    else:
        raise HTTPException(status_code=500, detail="Failed to reload model")


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(
        "services.predict.main:app",
        host=settings.predict_host,
        port=settings.predict_port,
        reload=settings.api_debug,
        log_level=settings.log_level.lower()
    )