#!/usr/bin/env python3
"""
Test script to verify JSON body logging functionality.
This script tests that request and response bodies are logged in JSON format.
"""

import requests
import json
import time
import sys
from pathlib import Path

# Add the project root to the Python path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from config.settings import settings

def test_json_body_logging():
    """Test JSON body logging for requests and responses"""
    print("Testing JSON Body Logging Functionality")
    print("=" * 50)
    
    base_url = f"http://{settings.api_host}:{settings.api_port}"
    
    # Display current settings
    print(f"JSON body logging enabled: {settings.log_request_body}")
    print(f"Max body log size: {settings.max_body_log_size} bytes")
    print(f"Testing endpoint: {base_url}/v1/predict")
    print()
    
    # Test data for prediction endpoint
    test_data = {
        "medianIncome": 8.3252,
        "housingMedianAge": 41.0,
        "totalRooms": 880.0,
        "totalBedrooms": 129.0,
        "population": 322.0,
        "households": 126.0,
        "medianHouseValue": 452600.0,
        "oceanProximity": "NEAR BAY"
    }
    
    print("Test 1: POST request with JSON body")
    print("-" * 30)
    print(f"Request data: {json.dumps(test_data, indent=2)}")
    
    try:
        response = requests.post(
            f"{base_url}/v1/predict",
            json=test_data,
            headers={"Content-Type": "application/json"},
            timeout=10
        )
        
        print(f"Response Status: {response.status_code}")
        print(f"Response Headers: {dict(response.headers)}")
        
        if response.status_code == 200:
            response_data = response.json()
            print(f"Response Body: {json.dumps(response_data, indent=2)}")
        else:
            print(f"  Response Text: {response.text}")
            
    except requests.exceptions.RequestException as e:
        print(f"Request failed: {e}")
    
    print("\n" + "=" * 50)
    
    # Test 2: GET request (no body)
    print("Test 2: GET request (no body)")
    print("-" * 30)
    
    try:
        response = requests.get(f"{base_url}/health", timeout=5)
        print(f"✓ Response Status: {response.status_code}")
        
        if response.status_code == 200:
            response_data = response.json()
            print(f"Response Body: {json.dumps(response_data, indent=2)}")
            
    except requests.exceptions.RequestException as e:
        print(f"Request failed: {e}")
    
    print("\n" + "=" * 50)
    
    # Test 3: Large JSON body (to test size limits)
    print("Test 3: Large JSON body (testing size limits)")
    print("-" * 30)
    
    large_data = {
        "large_field": "x" * (settings.max_body_log_size + 1000),  # Exceed max size
        "test": "data"
    }
    
    print(f"Request size: {len(json.dumps(large_data))} bytes")
    print(f"Max log size: {settings.max_body_log_size} bytes")
    
    try:
        response = requests.post(
            f"{base_url}/v1/predict",
            json=large_data,
            headers={"Content-Type": "application/json"},
            timeout=10
        )
        
        print(f"✓ Response Status: {response.status_code}")
        print(f"Note: Large body should be truncated in logs")
        
    except requests.exceptions.RequestException as e:
        print(f"Request failed: {e}")
    
    print("\n" + "=" * 50)
    
    # Test 4: Non-JSON content
    print("Test 4: Non-JSON content")
    print("-" * 30)
    
    try:
        response = requests.post(
            f"{base_url}/v1/predict",
            data="This is plain text, not JSON",
            headers={"Content-Type": "text/plain"},
            timeout=10
        )
        
        print(f"✓ Response Status: {response.status_code}")
        print(f"Note: Non-JSON body should be logged as metadata")
        
    except requests.exceptions.RequestException as e:
        print(f"Request failed: {e}")

def test_json_logging_configuration():
    """Test different JSON logging configurations"""
    print("\n" + "=" * 50)
    print("JSON Logging Configuration")
    print("=" * 50)
    
    print("Current Configuration:")
    print(f"  log_request_body: {settings.log_request_body}")
    print(f"  max_body_log_size: {settings.max_body_log_size}")
    print(f"  log_health_endpoints: {settings.log_health_endpoints}")
    
    print("\nConfiguration Options:")
    print("  Environment Variables:")
    print("    LOG_REQUEST_BODY=true/false")
    print("    MAX_BODY_LOG_SIZE=<bytes>")
    print("    LOG_HEALTH_ENDPOINTS=true/false")
    
    print("\n  Or in config/settings.py:")
    print("    log_request_body = True/False")
    print("    max_body_log_size = <bytes>")
    print("    log_health_endpoints = True/False")
    
    print("\nLogging Details:")
    print("  - Request bodies are logged for POST/PUT/PATCH methods")
    print("  - Response bodies are logged for all methods")
    print("  - Only JSON content is parsed and logged as structured data")
    print("  - Non-JSON content is logged with metadata only")
    print("  - Large bodies are truncated based on max_body_log_size")
    print("  - Health endpoints (/health, /readiness, /metrics) respect log_health_endpoints setting")

if __name__ == "__main__":
    print("MLOps JSON Body Logging Test")
    print("=" * 50)
    
    # Test JSON body logging
    test_json_body_logging()
    
    # Show configuration options
    test_json_logging_configuration()
    
    print(f"\nCheck the logs to verify JSON bodies are logged:")
    print(f"   - Application logs should contain 'body' field with actual JSON data")
    print(f"   - Large bodies should show truncation messages")
    print(f"   - Non-JSON bodies should show content type and size info")
    print(f"   - Log file location: {settings.log_db_path}")
    
    print("\nTest completed.")