import time
import uuid
from fastapi import Request, Response
from starlette.middleware.base import BaseHTTPMiddleware
from services.common.logging_config import get_logger
import json


class LoggingMiddleware(BaseHTTPMiddleware):
    """Middleware to log all incoming and outgoing requests/responses"""
    
    def __init__(self, app, service_name: str = "api"):
        super().__init__(app)
        self.service_name = service_name
        self.logger = get_logger(f"{service_name}-middleware")
    
    async def dispatch(self, request: Request, call_next):
        # Generate request ID
        request_id = str(uuid.uuid4())
        request.state.request_id = request_id
        
        # Start timing
        start_time = time.time()
        
        # Log incoming request (without reading body to avoid consumption issues)
        request_body = None
        content_length = request.headers.get("content-length", "0")
        if request.method in ["POST", "PUT", "PATCH"] and content_length != "0":
            request_body = f"<{request.method} body {content_length} bytes>"
        
        self.logger.info(
            "Incoming request",
            request_id=request_id,
            method=request.method,
            url=str(request.url),
            headers=dict(request.headers),
            body=request_body,
            service=self.service_name
        )
        
        # Process request
        try:
            response = await call_next(request)
        except Exception as e:
            # Calculate processing time for errors
            process_time = time.time() - start_time
            
            # Log the error
            self.logger.error(
                "Request processing error",
                request_id=request_id,
                error=str(e),
                error_type=type(e).__name__,
                duration=round(process_time, 4),
                service=self.service_name
            )
            
            # Re-raise the exception to let FastAPI handle it
            raise e
        
        # Calculate processing time
        process_time = time.time() - start_time
        
        # Log outgoing response (without reading body to avoid issues)
        response_body = f"<{response.status_code} response>"
        content_type = response.headers.get("content-type", "unknown")
        if content_type:
            response_body = f"<{response.status_code} response, {content_type}>"
        
        self.logger.info(
            "Outgoing response",
            request_id=request_id,
            status_code=response.status_code,
            headers=dict(response.headers),
            body=response_body,
            duration=round(process_time, 4),
            service=self.service_name
        )
        
        # Add custom headers
        response.headers["X-Request-ID"] = request_id
        response.headers["X-Process-Time"] = str(round(process_time, 4))
        
        return response


class MetricsMiddleware(BaseHTTPMiddleware):
    """Middleware to collect metrics for Prometheus"""
    
    def __init__(self, app, metrics_collector=None):
        super().__init__(app)
        self.metrics_collector = metrics_collector
    
    async def dispatch(self, request: Request, call_next):
        start_time = time.time()
        
        response = await call_next(request)
        
        process_time = time.time() - start_time
        
        # Collect metrics if collector is available
        if self.metrics_collector:
            self.metrics_collector.record_request(
                method=request.method,
                endpoint=request.url.path,
                status_code=response.status_code,
                duration=process_time
            )
        
        return response