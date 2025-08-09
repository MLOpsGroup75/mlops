#!/usr/bin/env python3
"""
Script to generate test traffic for monitoring dashboard validation
"""
import asyncio
import json
import random
import time
from typing import Dict, Any

import httpx


class TrafficGenerator:
    """Generate various types of HTTP traffic for monitoring"""
    
    def __init__(self, base_url: str = "http://localhost:8000"):
        self.base_url = base_url
        self.client = httpx.AsyncClient(timeout=30.0)
    
    async def __aenter__(self):
        return self
    
    async def __aexit__(self, exc_type, exc_val, exc_tb):
        await self.client.aclose()
    
    def get_valid_request(self) -> Dict[str, Any]:
        """Generate a valid prediction request"""
        return {
            "longitude": random.uniform(-125, -114),
            "latitude": random.uniform(32, 42),
            "housing_median_age": random.uniform(1, 52),
            "total_rooms": random.uniform(500, 10000),
            "total_bedrooms": random.uniform(100, 2000),
            "population": random.uniform(500, 5000),
            "households": random.uniform(100, 2000),
            "median_income": random.uniform(0.5, 15),
            "median_house_value": random.uniform(50000, 500000),
            "ocean_proximity": random.choice(["NEAR BAY", "NEAR OCEAN", "<1H OCEAN", "INLAND", "ISLAND"])
        }
    
    def get_invalid_request(self) -> Dict[str, Any]:
        """Generate an invalid prediction request (missing fields)"""
        valid = self.get_valid_request()
        # Remove random fields to cause validation errors
        fields_to_remove = random.sample(list(valid.keys()), random.randint(1, 3))
        for field in fields_to_remove:
            del valid[field]
        return valid
    
    async def make_health_request(self) -> int:
        """Make a health check request"""
        try:
            response = await self.client.get(f"{self.base_url}/health")
            return response.status_code
        except Exception as e:
            print(f"Health check failed: {e}")
            return 503
    
    async def make_prediction_request(self, valid: bool = True) -> int:
        """Make a prediction request"""
        try:
            data = self.get_valid_request() if valid else self.get_invalid_request()
            response = await self.client.post(
                f"{self.base_url}/v1/predict",
                json=data,
                headers={"Content-Type": "application/json"}
            )
            return response.status_code
        except Exception as e:
            print(f"Prediction request failed: {e}")
            return 503
    
    async def make_rate_limited_requests(self, count: int = 10) -> list:
        """Make rapid requests to trigger rate limiting"""
        tasks = []
        for _ in range(count):
            tasks.append(self.make_prediction_request())
        return await asyncio.gather(*tasks, return_exceptions=True)
    
    async def generate_mixed_traffic(self, duration_seconds: int = 60, requests_per_second: float = 2.0):
        """Generate mixed traffic patterns"""
        print(f"Generating mixed traffic for {duration_seconds} seconds at {requests_per_second} RPS")
        
        end_time = time.time() + duration_seconds
        request_count = 0
        
        while time.time() < end_time:
            start_batch = time.time()
            
            # Generate batch of requests
            batch_size = max(1, int(requests_per_second))
            tasks = []
            
            for _ in range(batch_size):
                request_type = random.choices(
                    ['valid_prediction', 'invalid_prediction', 'health'],
                    weights=[70, 20, 10]  # 70% valid, 20% invalid, 10% health
                )[0]
                
                if request_type == 'valid_prediction':
                    tasks.append(self.make_prediction_request(valid=True))
                elif request_type == 'invalid_prediction':
                    tasks.append(self.make_prediction_request(valid=False))
                else:
                    tasks.append(self.make_health_request())
            
            # Execute batch
            results = await asyncio.gather(*tasks, return_exceptions=True)
            
            # Count successful requests
            status_counts = {}
            for result in results:
                if isinstance(result, int):
                    status_counts[result] = status_counts.get(result, 0) + 1
                    request_count += 1
            
            if status_counts:
                print(f"Batch completed: {dict(status_counts)}")
            
            # Sleep to maintain rate
            batch_duration = time.time() - start_batch
            sleep_time = max(0, 1.0 - batch_duration)
            if sleep_time > 0:
                await asyncio.sleep(sleep_time)
        
        print(f"Generated {request_count} total requests")


async def main():
    """Main function to run traffic generation scenarios"""
    print("MLOps Monitoring Dashboard - Traffic Generator")
    print("=" * 50)
    
    async with TrafficGenerator() as generator:
        # Test connectivity
        print("Testing connectivity...")
        status = await generator.make_health_request()
        if status != 200:
            print(f"Warning: Health check returned {status}")
        else:
            print("✓ API is responding")
        
        print("\nChoose a traffic pattern:")
        print("1. Light traffic (30 seconds, 1 RPS)")
        print("2. Moderate traffic (60 seconds, 3 RPS)")
        print("3. Heavy traffic (60 seconds, 10 RPS)")
        print("4. Rate limit test (rapid burst)")
        print("5. Custom duration and rate")
        
        try:
            choice = input("\nEnter choice (1-5): ").strip()
            
            if choice == "1":
                await generator.generate_mixed_traffic(30, 1.0)
            elif choice == "2":
                await generator.generate_mixed_traffic(60, 3.0)
            elif choice == "3":
                await generator.generate_mixed_traffic(60, 10.0)
            elif choice == "4":
                print("Testing rate limiting...")
                results = await generator.make_rate_limited_requests(15)
                status_counts = {}
                for result in results:
                    if isinstance(result, int):
                        status_counts[result] = status_counts.get(result, 0) + 1
                print(f"Rate limit test results: {status_counts}")
            elif choice == "5":
                duration = int(input("Duration (seconds): "))
                rate = float(input("Requests per second: "))
                await generator.generate_mixed_traffic(duration, rate)
            else:
                print("Invalid choice")
                return
                
        except KeyboardInterrupt:
            print("\nTraffic generation stopped by user")
        except Exception as e:
            print(f"Error: {e}")
    
    print("\nTraffic generation complete!")
    print("Check the Grafana dashboard at http://localhost:3000")


if __name__ == "__main__":
    asyncio.run(main())
