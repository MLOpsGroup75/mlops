import time
import uuid
from fastapi import Request, Response
from starlette.middleware.base import BaseHTTPMiddleware
from starlette.responses import StreamingResponse
from services.common.logging_config import get_logger
from config.settings import settings
import json
import asyncio
import io
from typing import Optional, Dict, Any, Tuple


class LoggingMiddleware(BaseHTTPMiddleware):
    """Middleware to log all incoming and outgoing requests/responses"""

    # Define endpoints that should be included in logging
    LOG_ENDPOINTS = {"/api/v1/predict", "/v1/predict", "/predict/v1/predict",
                     "/api/v1/infer", "/v1/infer"}

    def __init__(self, app, service_name: str = "api"):
        super().__init__(app)
        self.service_name = service_name
        self.logger = get_logger(f"{service_name}-middleware")

    def _should_log_endpoint(self, path: str) -> bool:
        """Check if we should log this endpoint based on settings"""
        if path in self.LOG_ENDPOINTS:
            return True
        return settings.log_all_endpoints

    def _is_json_content(self, content_type: str) -> bool:
        """Check if content type is JSON"""
        return content_type and ("application/json" in content_type.lower())

    def _safe_parse_json(self, content: bytes) -> Optional[Dict[str, Any]]:
        """Safely parse JSON content with size limits"""
        try:
            if len(content) > settings.max_body_log_size:
                return {
                    "_truncated": f"Body too large ({len(content)} bytes, max {settings.max_body_log_size})"
                }

            text = content.decode("utf-8")
            return json.loads(text)
        except (json.JSONDecodeError, UnicodeDecodeError) as e:
            return {"_parse_error": f"Failed to parse JSON: {str(e)}"}
        except Exception as e:
            return {"_error": f"Unexpected error: {str(e)}"}

    async def _get_request_body(
        self, request: Request
    ) -> Tuple[Optional[Dict[str, Any]], bytes]:
        """Read and parse request body, return both parsed data and raw bytes"""
        if not settings.log_request_body:
            return None, b""

        try:
            # Read the body
            body_bytes = await request.body()

            if not body_bytes:
                return None, body_bytes

            # Check content type
            content_type = request.headers.get("content-type", "")
            if not self._is_json_content(content_type):
                return {
                    "_non_json": f"Content-Type: {content_type}, Size: {len(body_bytes)} bytes"
                }, body_bytes

            # Parse JSON
            parsed_data = self._safe_parse_json(body_bytes)
            return parsed_data, body_bytes

        except Exception as e:
            return {"_error": f"Failed to read request body: {str(e)}"}, b""

    async def _capture_response_body(
        self, response: Response
    ) -> Tuple[Optional[Dict[str, Any]], Response]:
        """Capture and parse response body, return both parsed data and updated response"""
        if not settings.log_request_body:
            return None, response

        try:
            # Check content type
            content_type = response.headers.get("content-type", "")
            if not self._is_json_content(content_type):
                return {"_non_json": f"Content-Type: {content_type}"}, response

            # Handle different response types
            if isinstance(response, StreamingResponse):
                # For streaming responses, capture the content
                body_bytes = b""
                async for chunk in response.body_iterator:
                    body_bytes += chunk

                # Parse the captured body
                parsed_data = self._safe_parse_json(body_bytes)

                # Create a new streaming response with the same content
                async def generate():
                    yield body_bytes

                new_response = StreamingResponse(
                    generate(),
                    status_code=response.status_code,
                    headers=response.headers,
                    media_type=response.media_type,
                )
                return parsed_data, new_response

            # For regular responses, try to get the body
            elif hasattr(response, "body") and response.body:
                parsed_data = self._safe_parse_json(response.body)
                return parsed_data, response

            return None, response

        except Exception as e:
            return {"_error": f"Failed to read response body: {str(e)}"}, response

    async def dispatch(self, request: Request, call_next):
        # Generate request ID
        request_id = str(uuid.uuid4())
        request.state.request_id = request_id

        # Start timing
        start_time = time.time()

        # Check if we should log this endpoint
        should_log = self._should_log_endpoint(request.url.path)

        # Read and log request body
        request_body_data = None
        if should_log and request.method in ["POST", "PUT", "PATCH"]:
            request_body_data, body_bytes = await self._get_request_body(request)

            # Replace the request's receive function to make body available again
            if body_bytes:

                async def receive():
                    return {"type": "http.request", "body": body_bytes}

                request._receive = receive
        elif request.method in ["POST", "PUT", "PATCH"]:
            # Even if not logging, we need to handle the case where body might be consumed
            # by reading it but not processing it
            try:
                body_bytes = await request.body()
                if body_bytes:

                    async def receive():
                        return {"type": "http.request", "body": body_bytes}

                    request._receive = receive
            except Exception:
                # If we can't read the body, let it continue normally
                pass

        # Log incoming request
        if should_log:
            self.logger.info(
                "Incoming request",
                request_id=request_id,
                method=request.method,
                url=str(request.url),
                headers=dict(request.headers),
                body=request_body_data,
                service=self.service_name,
            )

        # Process request
        try:
            response = await call_next(request)
        except Exception as e:
            # Calculate processing time for errors
            process_time = time.time() - start_time

            # Log the error
            if should_log:
                self.logger.error(
                    "Request processing error",
                    request_id=request_id,
                    error=str(e),
                    error_type=type(e).__name__,
                    duration=round(process_time, 4),
                    service=self.service_name,
                )

            # Re-raise the exception to let FastAPI handle it
            raise e

        # Calculate processing time
        process_time = time.time() - start_time

        # Capture and log response body
        response_body_data = None
        if should_log:
            response_body_data, response = await self._capture_response_body(response)

            # Log outgoing response
            self.logger.info(
                "Outgoing response",
                request_id=request_id,
                status_code=response.status_code,
                headers=dict(response.headers),
                body=response_body_data,
                duration=round(process_time, 4),
                service=self.service_name,
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
                duration=process_time,
            )

        return response
