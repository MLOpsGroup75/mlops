from prometheus_client import Counter, Histogram, Gauge, generate_latest, CONTENT_TYPE_LATEST
import time
from typing import Dict, Any


class MetricsCollector:
    """Prometheus metrics collector for the MLOps services"""
    
    def __init__(self, service_name: str = "mlops"):
        self.service_name = service_name
        
        # HTTP request metrics
        self.http_requests_total = Counter(
            'http_requests_total',
            'Total number of HTTP requests',
            ['method', 'endpoint', 'status_code', 'service']
        )
        
        self.http_request_duration_seconds = Histogram(
            'http_request_duration_seconds',
            'Time spent processing HTTP requests',
            ['method', 'endpoint', 'service'],
            buckets=(0.01, 0.05, 0.1, 0.25, 0.5, 1.0, 2.5, 5.0, 10.0)
        )
        
        # Prediction-specific metrics
        self.prediction_requests_total = Counter(
            'prediction_requests_total',
            'Total number of prediction requests',
            ['service', 'status']
        )
        
        self.prediction_duration_seconds = Histogram(
            'prediction_duration_seconds',
            'Time spent on predictions',
            ['service'],
            buckets=(0.01, 0.05, 0.1, 0.25, 0.5, 1.0, 2.5, 5.0)
        )
        
        # Rate limiting metrics
        self.rate_limit_exceeded_total = Counter(
            'rate_limit_exceeded_total',
            'Total number of rate limit exceeded responses',
            ['service']
        )
        
        # Service health metrics
        self.service_up = Gauge(
            'service_up',
            'Service availability',
            ['service']
        )
        
        self.active_connections = Gauge(
            'active_connections',
            'Number of active connections',
            ['service']
        )
        
        # Initialize service as up
        self.service_up.labels(service=service_name).set(1)
    
    def record_request(self, method: str, endpoint: str, status_code: int, duration: float):
        """Record an HTTP request"""
        self.http_requests_total.labels(
            method=method,
            endpoint=endpoint,
            status_code=str(status_code),
            service=self.service_name
        ).inc()
        
        self.http_request_duration_seconds.labels(
            method=method,
            endpoint=endpoint,
            service=self.service_name
        ).observe(duration)
    
    def record_prediction(self, duration: float, status: str = "success"):
        """Record a prediction request"""
        self.prediction_requests_total.labels(
            service=self.service_name,
            status=status
        ).inc()
        
        self.prediction_duration_seconds.labels(
            service=self.service_name
        ).observe(duration)
    
    def record_rate_limit_exceeded(self):
        """Record a rate limit exceeded event"""
        self.rate_limit_exceeded_total.labels(
            service=self.service_name
        ).inc()
    
    def set_service_status(self, is_up: bool):
        """Set service status"""
        self.service_up.labels(service=self.service_name).set(1 if is_up else 0)
    
    def increment_active_connections(self):
        """Increment active connections counter"""
        self.active_connections.labels(service=self.service_name).inc()
    
    def decrement_active_connections(self):
        """Decrement active connections counter"""
        self.active_connections.labels(service=self.service_name).dec()
    
    def export_metrics(self) -> str:
        """Export metrics in Prometheus format"""
        return generate_latest().decode('utf-8')
    
    def get_content_type(self) -> str:
        """Get the content type for metrics"""
        return CONTENT_TYPE_LATEST


# Global metrics collectors for each service - initialized as None
api_metrics = None
predict_metrics = None


def get_api_metrics():
    """Get or create API metrics collector"""
    global api_metrics
    if api_metrics is None:
        api_metrics = MetricsCollector("api")
    return api_metrics


def get_predict_metrics():
    """Get or create predict metrics collector"""
    global predict_metrics
    if predict_metrics is None:
        predict_metrics = MetricsCollector("predict")
    return predict_metrics