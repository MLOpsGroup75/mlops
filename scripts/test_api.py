#!/usr/bin/env python3
"""Test script for the MLOps API"""

import requests
import json
import time

API_BASE_URL = "http://localhost:8000"


def test_health():
    """Test health endpoint"""
    print("Testing health endpoint...")
    response = requests.get(f"{API_BASE_URL}/health")
    print(f"Health Status: {response.status_code}")
    print(f"Response: {response.json()}")
    print()


def test_readiness():
    """Test readiness endpoint"""
    print("Testing readiness endpoint...")
    response = requests.get(f"{API_BASE_URL}/readiness")
    print(f"Readiness Status: {response.status_code}")
    print(f"Response: {response.json()}")
    print()


def test_metrics():
    """Test metrics endpoint"""
    print("Testing metrics endpoint...")
    response = requests.get(f"{API_BASE_URL}/metrics")
    print(f"Metrics Status: {response.status_code}")
    print(f"Metrics Preview: {response.text[:200]}...")
    print()


def test_prediction():
    """Test prediction endpoint"""
    print("Testing prediction endpoint...")

    test_data = {
        "longitude": -122.23,
        "latitude": 37.88,
        "housingMedianAge": 41.0,
        "totalRooms": 880.0,
        "totalBedrooms": 129.0,
        "population": 322.0,
        "households": 126.0,
        "medianIncome": 8.3252,
        "medianHouseValue": 452600.0,
        "oceanProximity": "NEAR BAY",
    }

    response = requests.post(
        f"{API_BASE_URL}/v1/predict",
        json=test_data,
        headers={"Content-Type": "application/json"},
    )

    print(f"Prediction Status: {response.status_code}")
    print(f"Response: {response.json()}")
    print()


def test_rate_limiting():
    """Test rate limiting"""
    print("Testing rate limiting (sending 10 requests quickly)...")

    test_data = {
        "longitude": -122.23,
        "latitude": 37.88,
        "housingMedianAge": 41.0,
        "totalRooms": 880.0,
        "totalBedrooms": 129.0,
        "population": 322.0,
        "households": 126.0,
        "medianIncome": 8.3252,
        "medianHouseValue": 452600.0,
        "oceanProximity": "NEAR BAY",
    }

    for i in range(10):
        response = requests.post(
            f"{API_BASE_URL}/v1/predict",
            json=test_data,
            headers={"Content-Type": "application/json"},
        )
        print(f"Request {i+1}: Status {response.status_code}")
        if response.status_code == 429:
            print("Rate limit triggered!")
            break
        time.sleep(0.1)
    print()


if __name__ == "__main__":
    print("MLOps API Test Suite")
    print("=" * 50)

    try:
        test_health()
        test_readiness()
        test_metrics()
        test_prediction()
        test_rate_limiting()

        print("All tests completed!")

    except requests.exceptions.ConnectionError:
        print("Error: Could not connect to API. Make sure the services are running.")
    except Exception as e:
        print(f"Error: {e}")
