#!/usr/bin/env python3
"""
Test script for the /v1/infer endpoint that calls Databricks.
This script tests the new inference endpoint with sample data.
"""

import requests
import json
import os
import sys
from typing import Dict, Any

# Sample data matching the format expected by the endpoint
SAMPLE_DATA = {
    "MedInc": -0.3261960037692928,
    "HouseAge": 0.3484902466663322,
    "AveRooms": -0.1749164614622689,
    "AveBedrms": -0.2083654336540427,
    "Population": 0.7682762831665109,
    "AveOccup": 0.0513760919421774,
    "Latitude": -1.3728111990669665,
    "Longitude": 1.2725865624715638
}

def test_infer_endpoint(base_url: str = "http://localhost:8000") -> None:
    """Test the /v1/infer endpoint"""
    
    endpoint = f"{base_url}/v1/infer"
    
    print(f"Testing inference endpoint: {endpoint}")
    print(f"Sample data: {json.dumps(SAMPLE_DATA, indent=2)}")
    print("-" * 50)
    
    try:
        # Make the request
        response = requests.post(
            endpoint,
            json=SAMPLE_DATA,
            headers={"Content-Type": "application/json"},
            timeout=30
        )
        
        print(f"Response Status: {response.status_code}")
        print(f"Response Headers: {dict(response.headers)}")
        
        if response.status_code == 200:
            result = response.json()
            print(f"Success! Response: {json.dumps(result, indent=2)}")
            
            # Validate response structure
            if "data" in result and "predictions" in result["data"]:
                print("✅ Response structure is valid")
                print(f"✅ Predictions received: {len(result['data']['predictions'])}")
            else:
                print("❌ Response structure is invalid")
                
        else:
            print(f"Error Response: {response.text}")
            print("❌ Request failed")
            
    except requests.exceptions.ConnectionError:
        print("❌ Connection failed. Is the API service running?")
        print(f"   Try: cd services/api && python main.py")
        
    except requests.exceptions.Timeout:
        print("❌ Request timed out. Check if Databricks endpoint is accessible.")
        
    except Exception as e:
        print(f"❌ Unexpected error: {str(e)}")

def test_health_endpoint(base_url: str = "http://localhost:8000") -> None:
    """Test the health endpoint to verify service is running"""
    
    try:
        response = requests.get(f"{base_url}/health", timeout=5)
        if response.status_code == 200:
            print("✅ API service is running")
            return True
        else:
            print("❌ API service health check failed")
            return False
    except:
        print("❌ Cannot connect to API service")
        return False

def main():
    """Main test function"""
    
    # Check if DATABRICKS_TOKEN is set
    if not os.environ.get("DATABRICKS_TOKEN"):
        print("⚠️  Warning: DATABRICKS_TOKEN environment variable is not set")
        print("   The inference endpoint may fail with 'token not configured' error")
        print("   Set it with: export DATABRICKS_TOKEN='your_token_here'")
        print()
    
    # Check if DATABRICKS_ENDPOINT_URL is set
    if not os.environ.get("DATABRICKS_ENDPOINT_URL"):
        print("ℹ️  Using default Databricks endpoint URL from settings")
        print("   To override, set: export DATABRICKS_ENDPOINT_URL='your_url_here'")
        print()
    
    # Test health first
    if not test_health_endpoint():
        print("\n❌ Cannot proceed with inference test - API service is not running")
        sys.exit(1)
    
    print("\n" + "="*60)
    print("TESTING INFERENCE ENDPOINT")
    print("="*60)
    
    # Test the inference endpoint
    test_infer_endpoint()
    
    print("\n" + "="*60)
    print("TEST COMPLETED")
    print("="*60)

if __name__ == "__main__":
    main()
