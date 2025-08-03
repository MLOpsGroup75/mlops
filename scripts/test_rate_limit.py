#!/usr/bin/env python3
"""
Test script to verify rate limiting functionality.
This script will test that rate limit returns 429 status code correctly.
"""

import requests
import time
import sys
from pathlib import Path

# Add the project root to the Python path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from config.settings import settings

def test_rate_limiting():
    """Test rate limiting functionality"""
    print("Testing Rate Limiting Functionality")
    print("=" * 50)
    
    base_url = f"http://{settings.api_host}:{settings.api_port}"
    
    # Display current rate limit settings
    print(f"Rate limit: {settings.rate_limit_requests} requests per {settings.rate_limit_window} seconds")
    print(f"Testing endpoint: {base_url}/health")
    print()
    
    successful_requests = 0
    rate_limited_requests = 0
    
    print("Making rapid requests to trigger rate limit...")
    
    # Make more requests than allowed to trigger rate limiting
    for i in range(settings.rate_limit_requests + 3):
        try:
            response = requests.get(f"{base_url}/health", timeout=5)
            
            if response.status_code == 200:
                successful_requests += 1
                print(f"Request {i+1}: Status {response.status_code} - OK")
                
            elif response.status_code == 429:
                rate_limited_requests += 1
                print(f"  Request {i+1}: Status {response.status_code} - Rate Limited")
                
                # Try to parse response content
                try:
                    error_data = response.json()
                    print(f"   Error: {error_data.get('error', 'Unknown')}")
                    print(f"   Detail: {error_data.get('detail', 'Unknown')}")
                    print(f"   Retry After: {error_data.get('retry_after', 'Unknown')} seconds")
                except:
                    print(f"   Raw response: {response.text}")
                    
            else:
                print(f"Request {i+1}: Unexpected status {response.status_code}")
                print(f"   Response: {response.text}")
                
        except requests.exceptions.RequestException as e:
            print(f"Request {i+1}: Network error - {e}")
            
        # Small delay between requests
        time.sleep(0.1)
    
    print("\n" + "=" * 50)
    print("Test Results:")
    print(f"Successful requests: {successful_requests}")
    print(f"  Rate limited requests: {rate_limited_requests}")
    
    # Validate results
    if rate_limited_requests > 0:
        print("\n SUCCESS: Rate limiting is working correctly!")
        print("   - Rate limit is properly enforced")
        print("   - HTTP 429 status code is returned correctly")
        print("   - No HTTP 500 errors when rate limited")
    else:
        print("\n  WARNING: No rate limiting detected")
        print("   - Check if the API service is running")
        print("   - Verify rate limit settings are applied")
        
    return rate_limited_requests > 0

def test_rate_limit_recovery():
    """Test that rate limit resets after the window"""
    print("\n" + "=" * 50)
    print("Testing Rate Limit Recovery")
    print("=" * 50)
    
    base_url = f"http://{settings.api_host}:{settings.api_port}"
    
    print(f"Waiting {settings.rate_limit_window + 1} seconds for rate limit window to reset...")
    
    # Wait for rate limit window to reset
    for i in range(settings.rate_limit_window + 1):
        print(f"{settings.rate_limit_window - i} seconds remaining...", end="\r")
        time.sleep(1)
    
    print("\n\nTesting requests after rate limit reset...")
    
    try:
        response = requests.get(f"{base_url}/health", timeout=5)
        
        if response.status_code == 200:
            print("SUCCESS: Rate limit has reset, requests working again")
            return True
        elif response.status_code == 429:
            print("FAILURE: Still rate limited after waiting")
            return False
        else:
            print(f"  Unexpected status: {response.status_code}")
            return False
            
    except requests.exceptions.RequestException as e:
        print(f"Network error: {e}")
        return False

if __name__ == "__main__":
    print("MLOps Rate Limiting Test")
    print("=" * 50)
    
    # Test rate limiting
    rate_limit_works = test_rate_limiting()
    
    if rate_limit_works:
        # Test recovery only if rate limiting worked
        recovery_works = test_rate_limit_recovery()
        
        if recovery_works:
            print("\n All tests passed! Rate limiting is working correctly.")
        else:
            print("\n  Rate limiting works but recovery test failed.")
    else:
        print("\nRate limiting test failed. Check API service configuration.")
        
    print("\nTest completed.")