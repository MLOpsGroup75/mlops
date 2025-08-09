"""
Pytest configuration and shared fixtures
"""

import asyncio
import os
import sqlite3
import tempfile
from unittest.mock import Mock, patch

import pytest
from fastapi.testclient import TestClient
from httpx import AsyncClient

from config.settings import Settings
from services.common.models import OceanProximity, PredictionRequest


@pytest.fixture(scope="session")
def event_loop():
    """Create an instance of the default event loop for the test session."""
    loop = asyncio.get_event_loop_policy().new_event_loop()
    yield loop
    loop.close()


@pytest.fixture
def test_settings():
    """Create test settings with temporary database"""
    with tempfile.TemporaryDirectory() as temp_dir:
        test_db_path = os.path.join(temp_dir, "test_logs.db")

        settings = Settings(
            api_host="localhost",
            api_port=8000,
            predict_host="localhost",
            predict_port=8001,
            predict_url="http://localhost:8001",
            log_level="DEBUG",
            log_to_file=True,
            log_db_path=test_db_path,
            database_url=f"sqlite:///{test_db_path}",
            rate_limit_requests=100,  # Higher limit for tests
            rate_limit_window=60,
            model_accuracy=0.85,
        )
        yield settings


@pytest.fixture
def test_db_path():
    """Create a temporary database path for testing"""
    with tempfile.NamedTemporaryFile(suffix=".db", delete=False) as temp_file:
        db_path = temp_file.name

    yield db_path

    # Cleanup
    if os.path.exists(db_path):
        os.unlink(db_path)


@pytest.fixture
def sample_prediction_request():
    """Sample prediction request for testing"""
    return PredictionRequest(
        longitude=-122.23,
        latitude=37.88,
        housingMedianAge=41.0,
        totalRooms=880.0,
        totalBedrooms=129.0,
        population=322.0,
        households=126.0,
        medianIncome=8.3252,
        medianHouseValue=452600.0,
        oceanProximity=OceanProximity.NEAR_BAY,
    )


@pytest.fixture
def sample_features():
    """Sample features dict for model testing"""
    return {
        "longitude": -122.23,
        "latitude": 37.88,
        "housing_median_age": 41.0,
        "total_rooms": 880.0,
        "total_bedrooms": 129.0,
        "population": 322.0,
        "households": 126.0,
        "median_income": 8.3252,
        "median_house_value": 452600.0,
        "ocean_proximity": "NEAR BAY",
    }


@pytest.fixture
def mock_httpx_client():
    """Mock httpx.AsyncClient for testing external API calls"""
    with patch("httpx.AsyncClient") as mock_client:
        mock_instance = Mock()
        mock_client.return_value = mock_instance
        yield mock_instance


@pytest.fixture
def mock_predict_service_response():
    """Mock response from predict service"""
    return {
        "housing_price": 450000.0,
        "accuracy": 0.85,
        "request_id": "test-123",
        "processing_time": 0.1,
    }


@pytest.fixture
async def api_client(test_settings):
    """FastAPI test client for API service"""
    with patch("config.settings.settings", test_settings):
        from app.main import app

        # Mock the HTTP client to avoid actual HTTP calls
        with patch("app.main.http_client") as mock_client:
            mock_client.post.return_value = Mock(
                status_code=200,
                json=lambda: {
                    "housing_price": 450000.0,
                    "accuracy": 0.85,
                    "request_id": "test-123",
                    "processing_time": 0.1,
                },
            )

            async with AsyncClient(app=app, base_url="http://test") as client:
                yield client


@pytest.fixture
async def predict_client(test_settings):
    """FastAPI test client for predict service"""
    with patch("config.settings.settings", test_settings):
        from app.main import app

        async with AsyncClient(app=app, base_url="http://test") as client:
            yield client


@pytest.fixture
def setup_test_database(test_db_path):
    """Setup test database with initial schema"""
    conn = sqlite3.connect(test_db_path)
    cursor = conn.cursor()

    # Create logs table
    cursor.execute(
        """
        CREATE TABLE IF NOT EXISTS logs (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            timestamp TEXT NOT NULL,
            level TEXT NOT NULL,
            logger_name TEXT,
            message TEXT,
            service TEXT,
            request_id TEXT,
            extra_data TEXT,
            created_at DATETIME DEFAULT CURRENT_TIMESTAMP
        )
    """
    )

    conn.commit()
    conn.close()

    return test_db_path
