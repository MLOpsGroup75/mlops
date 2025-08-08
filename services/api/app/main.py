import asyncio
import time
import uuid
from datetime import datetime

import httpx
import structlog
from fastapi import APIRouter, Depends, FastAPI, HTTPException, Request, status
from fastapi.responses import JSONResponse, PlainTextResponse
from slowapi import Limiter
from slowapi.errors import RateLimitExceeded
from slowapi.util import get_remote_address

from config.settings import settings
from services.common.logging_config import get_logger, setup_logging
from services.common.metrics import get_api_metrics
from services.common.middleware import LoggingMiddleware, MetricsMiddleware
from services.common.models import (
    ErrorResponse,
    HealthResponse,
    InternalPredictionRequest,
    InternalPredictionResponse,
    PredictionRequest,
    PredictionSuccess,
    ReadinessResponse,
)

# Initialize logging
setup_logging("api-service")
logger = get_logger("api-service")

# Initialize rate limiter
limiter = Limiter(key_func=get_remote_address)

# Create FastAPI app
app = FastAPI(
    title="MLOps Housing Price Prediction API",
    description="API service for housing price prediction with monitoring and health checks",
    version="1.0.0",
    docs_url="/docs",
    redoc_url="/redoc",
)

# Create API router for endpoints that should be accessible via ALB without /api prefix
api_router = APIRouter()

# Add rate limiting
app.state.limiter = limiter
# Note: Custom rate limit handler is defined below, no need for default handler

# Add middleware
app.add_middleware(LoggingMiddleware, service_name="api")
app.add_middleware(MetricsMiddleware, metrics_collector=get_api_metrics())

# HTTP client for communication with predict service
http_client = None


@app.on_event("startup")
async def startup_event():
    """Initialize services on startup"""
    global http_client
    http_client = httpx.AsyncClient(timeout=30.0)
    get_api_metrics().set_service_status(True)
    logger.info("API service started", service="api", version=settings.service_version)


@app.on_event("shutdown")
async def shutdown_event():
    """Cleanup on shutdown"""
    if http_client:
        await http_client.aclose()
    get_api_metrics().set_service_status(False)
    logger.info("API service stopped", service="api")


async def call_predict_service(
    prediction_request: InternalPredictionRequest,
) -> InternalPredictionResponse:
    """Call the predict service with the prediction request"""
    try:
        url = f"{settings.predict_url}/predict"

        start_time = time.time()
        response = await http_client.post(
            url,
            json=prediction_request.dict(),
            headers={"Content-Type": "application/json"},
        )
        duration = time.time() - start_time

        if response.status_code == 200:
            result = response.json()
            get_api_metrics().record_prediction(duration, "success")
            return InternalPredictionResponse(**result)
        else:
            get_api_metrics().record_prediction(duration, "error")
            error_detail = (
                response.text if response.text else "Unknown error from predict service"
            )
            raise HTTPException(
                status_code=response.status_code,
                detail=f"Predict service error: {error_detail}",
            )

    except httpx.RequestError as e:
        get_api_metrics().record_prediction(0, "error")
        logger.error(
            "Failed to connect to predict service", error=str(e), service="api"
        )
        raise HTTPException(status_code=503, detail="Predict service unavailable")
    except Exception as e:
        get_api_metrics().record_prediction(0, "error")
        logger.error(
            "Unexpected error calling predict service", error=str(e), service="api"
        )
        raise HTTPException(status_code=500, detail="Internal server error")


@api_router.post("/v1/predict", response_model=PredictionSuccess)
@limiter.limit(f"{settings.rate_limit_requests}/{settings.rate_limit_window}seconds")
async def predict_housing_price(
    request: Request, prediction_request: PredictionRequest
) -> PredictionSuccess:
    """Predict housing price based on input features"""

    request_id = getattr(request.state, "request_id", str(uuid.uuid4()))

    logger.info("Received prediction request", request_id=request_id, service="api")

    try:
        # Convert to internal format
        internal_request = InternalPredictionRequest.from_prediction_request(
            prediction_request, request_id
        )

        # Call predict service
        internal_response = await call_predict_service(internal_request)

        # Convert back to external format
        response = internal_response.to_prediction_success()

        logger.info(
            "Prediction completed successfully",
            request_id=request_id,
            housing_price=internal_response.housing_price,
            processing_time=internal_response.processing_time,
            service="api",
        )

        return response

    except HTTPException:
        raise
    except Exception as e:
        logger.error(
            "Unexpected error during prediction",
            request_id=request_id,
            error=str(e),
            service="api",
        )
        raise HTTPException(status_code=500, detail="Internal server error")


@app.get("/metrics", response_class=PlainTextResponse)
async def get_metrics():
    """Export Prometheus metrics"""
    return get_api_metrics().export_metrics()


@app.get("/health", response_model=HealthResponse)
async def health_check():
    """Health check endpoint"""
    return HealthResponse(
        status="OK", timestamp=datetime.utcnow().isoformat(), service="api"
    )


@app.get("/readiness", response_model=ReadinessResponse)
async def readiness_check():
    """Readiness check endpoint for Kubernetes"""
    checks = {}

    # Check predict service health
    predict_healthy = False
    try:
        url = f"{settings.predict_url}/health"
        response = await http_client.get(url, timeout=5.0)
        predict_healthy = response.status_code == 200
        checks["predictService"] = "OK" if predict_healthy else "FAIL"
    except Exception as e:
        checks["predictService"] = f"FAIL: {str(e)}"

    # Check database (SQLite log database)
    db_healthy = True
    try:
        import sqlite3

        conn = sqlite3.connect(settings.log_db_path)
        conn.execute("SELECT 1")
        conn.close()
        checks["database"] = "OK"
    except Exception as e:
        db_healthy = False
        checks["database"] = f"FAIL: {str(e)}"

    overall_ready = predict_healthy and db_healthy

    if not overall_ready:
        raise HTTPException(status_code=503, detail="Service not ready")

    return ReadinessResponse(ready=True, checks=checks)


# Custom exception handlers
@app.exception_handler(RateLimitExceeded)
async def rate_limit_handler(request: Request, exc: RateLimitExceeded):
    """Handle rate limit exceeded"""
    get_api_metrics().record_rate_limit_exceeded()
    logger.warning(
        "Rate limit exceeded", client_ip=get_remote_address(request), service="api"
    )

    return JSONResponse(
        status_code=429,
        content={
            "error": "Rate limit exceeded",
            "detail": "Rate limit exceeded. Please try again later.",
            "retry_after": exc.retry_after if hasattr(exc, "retry_after") else 60,
        },
    )


# Include the API router twice:
# 1. With /api prefix for ALB routing (ALB forwards /api/v1/predict to service)
# 2. Without prefix for local access (direct access to /v1/predict)
app.include_router(api_router, prefix="/api")
app.include_router(api_router)


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(
        "services.api.main:app",
        host=settings.api_host,
        port=settings.api_port,
        reload=settings.api_debug,
        log_level=settings.log_level.lower(),
    )
