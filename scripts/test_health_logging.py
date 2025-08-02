#!/usr/bin/env python3
"""
Test script to verify that health endpoint logging can be enabled/disabled.
This script demonstrates the new log_health_endpoints flag functionality.
"""

import requests
import time
import os
import sys
from pathlib import Path

# Add the project root to the Python path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from config.settings import settings

def test_endpoint(url: str, endpoint_name: str):
    """Test an endpoint and measure response time"""
    try:
        start_time = time.time()
        response = requests.get(url, timeout=5)
        duration = time.time() - start_time
        
        print(f"✓ {endpoint_name}: Status {response.status_code}, Duration: {duration:.3f}s")
        return True
    except requests.exceptions.RequestException as e:
        print(f"✗ {endpoint_name}: Error - {e}")
        return False

def test_health_logging_feature():
    """Test the health endpoint logging feature"""
    print("Testing Health Endpoint Logging Feature")
    print("=" * 50)
    
    base_url = f"http://{settings.api_host}:{settings.api_port}"
    
    # Display current setting
    print(f"Current log_health_endpoints setting: {settings.log_health_endpoints}")
    print(f"Expected behavior: {'Logging ENABLED' if settings.log_health_endpoints else 'Logging DISABLED'} for health endpoints")
    print()
    
    # Test endpoints that should be affected by the flag
    health_endpoints = [
        ("/health", "Health Check"),
        ("/readiness", "Readiness Check"), 
        ("/metrics", "Metrics")
    ]
    
    # Test endpoints that should always be logged
    regular_endpoints = [
        ("/docs", "API Documentation"),
        ("/v1/predict", "Prediction Endpoint (POST will fail but should be logged)")
    ]
    
    print("Testing Health & Metric Endpoints (affected by log_health_endpoints flag):")
    for endpoint, name in health_endpoints:
        test_endpoint(f"{base_url}{endpoint}", name)
    
    print("\nTesting Regular Endpoints (always logged):")
    for endpoint, name in regular_endpoints:
        test_endpoint(f"{base_url}{endpoint}", name)
    
    print(f"\n📝 Check the logs to verify:")
    print(f"   - Health endpoints (/health, /readiness, /metrics) should {'APPEAR' if settings.log_health_endpoints else 'NOT APPEAR'} in logs")
    print(f"   - Regular endpoints should ALWAYS appear in logs")
    print(f"   - Log file location: {settings.log_db_path}")

def show_configuration_options():
    """Show how to configure the feature"""
    print("\n" + "=" * 50)
    print("Configuration Options:")
    print("=" * 50)
    print("To ENABLE health endpoint logging:")
    print("  1. Set environment variable: LOG_HEALTH_ENDPOINTS=true")
    print("  2. Or modify config/settings.py: log_health_endpoints = True")
    print()
    print("To DISABLE health endpoint logging (default):")
    print("  1. Set environment variable: LOG_HEALTH_ENDPOINTS=false") 
    print("  2. Or modify config/settings.py: log_health_endpoints = False")
    print()
    print("Affected endpoints:")
    print("  - /health")
    print("  - /readiness") 
    print("  - /metrics")

if __name__ == "__main__":
    print("MLOps Health Endpoint Logging Test")
    print("=" * 50)
    
    test_health_logging_feature()
    show_configuration_options()