"""
Unit tests for API service
"""
import json
from unittest.mock import AsyncMock, Mock, patch

import pytest
from fastapi import status
from httpx import Response

from services.common.models import OceanProximity, PredictionRequest


@pytest.mark.api
class TestAPIHealthEndpoints:
    """Test health-related endpoints"""

    @pytest.mark.asyncio
    async def test_health_endpoint(self, api_client):
        """Test health check endpoint"""
        response = await api_client.get("/health")

        assert response.status_code == status.HTTP_200_OK
        data = response.json()
        assert data["status"] == "OK"
        assert data["service"] == "api"
        assert "timestamp" in data

    @pytest.mark.asyncio
    async def test_readiness_endpoint_success(self, api_client):
        """Test readiness check when all dependencies are healthy"""
        with patch("app.main.http_client") as mock_client:
            # Mock predict service health check
            mock_client.get.return_value = Mock(status_code=200)

            # Mock SQLite connection
            with patch("sqlite3.connect") as mock_connect:
                mock_conn = Mock()
                mock_connect.return_value = mock_conn
                mock_conn.execute.return_value = None

                response = await api_client.get("/readiness")

                assert response.status_code == status.HTTP_200_OK
                data = response.json()
                assert data["ready"] is True
                assert data["checks"]["predictService"] == "OK"
                assert data["checks"]["database"] == "OK"

    @pytest.mark.asyncio
    async def test_readiness_endpoint_predict_service_down(self, api_client):
        """Test readiness check when predict service is down"""
        with patch("app.main.http_client") as mock_client:
            # Mock predict service failure
            mock_client.get.side_effect = Exception("Connection error")

            response = await api_client.get("/readiness")

            assert response.status_code == status.HTTP_503_SERVICE_UNAVAILABLE

    @pytest.mark.asyncio
    async def test_metrics_endpoint(self, api_client):
        """Test metrics endpoint"""
        response = await api_client.get("/metrics")

        assert response.status_code == status.HTTP_200_OK
        assert "text/plain" in response.headers["content-type"]


@pytest.mark.api
class TestAPIPredictionEndpoint:
    """Test prediction endpoint"""

    @pytest.mark.asyncio
    async def test_prediction_success(self, api_client, sample_prediction_request):
        """Test successful prediction"""
        with patch("app.main.call_predict_service") as mock_call:
            from services.common.models import InternalPredictionResponse

            mock_call.return_value = InternalPredictionResponse(
                housing_price=450000.0,
                accuracy=0.85,
                request_id="test-123",
                processing_time=0.1,
            )

            response = await api_client.post(
                "/v1/predict", json=sample_prediction_request.dict()
            )

            assert response.status_code == status.HTTP_200_OK
            data = response.json()
            assert "data" in data
            assert data["data"]["housingPrice"] == 450000.0
            assert data["data"]["accuracy"] == 0.85

    @pytest.mark.asyncio
    async def test_prediction_invalid_input(self, api_client):
        """Test prediction with invalid input"""
        invalid_request = {
            "longitude": "invalid",  # Should be float
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

        response = await api_client.post("/v1/predict", json=invalid_request)

        assert response.status_code == status.HTTP_422_UNPROCESSABLE_ENTITY

    @pytest.mark.asyncio
    async def test_prediction_missing_fields(self, api_client):
        """Test prediction with missing required fields"""
        incomplete_request = {
            "longitude": -122.23,
            "latitude": 37.88
            # Missing other required fields
        }

        response = await api_client.post("/v1/predict", json=incomplete_request)

        assert response.status_code == status.HTTP_422_UNPROCESSABLE_ENTITY

    @pytest.mark.asyncio
    async def test_prediction_predict_service_error(
        self, api_client, sample_prediction_request
    ):
        """Test prediction when predict service returns error"""
        with patch("app.main.call_predict_service") as mock_call:
            from fastapi import HTTPException

            mock_call.side_effect = HTTPException(
                status_code=503, detail="Service unavailable"
            )

            response = await api_client.post(
                "/v1/predict", json=sample_prediction_request.dict()
            )

            assert response.status_code == status.HTTP_503_SERVICE_UNAVAILABLE

    @pytest.mark.asyncio
    async def test_prediction_internal_error(
        self, api_client, sample_prediction_request
    ):
        """Test prediction with internal error"""
        with patch("app.main.call_predict_service") as mock_call:
            mock_call.side_effect = Exception("Internal error")

            response = await api_client.post(
                "/v1/predict", json=sample_prediction_request.dict()
            )

            assert response.status_code == status.HTTP_500_INTERNAL_SERVER_ERROR


@pytest.mark.api
class TestCallPredictService:
    """Test the call_predict_service function"""

    @pytest.mark.asyncio
    async def test_successful_call(self, test_settings, mock_predict_service_response):
        """Test successful call to predict service"""
        with patch("config.settings.settings", test_settings):
            from app.main import call_predict_service

            from services.common.models import InternalPredictionRequest

            mock_client = Mock()
            mock_response = Mock()
            mock_response.status_code = 200
            mock_response.json.return_value = mock_predict_service_response
            mock_client.post.return_value = mock_response

            with patch("services.api.main.http_client", mock_client):
                request = InternalPredictionRequest(
                    longitude=-122.23,
                    latitude=37.88,
                    housing_median_age=41.0,
                    total_rooms=880.0,
                    total_bedrooms=129.0,
                    population=322.0,
                    households=126.0,
                    median_income=8.3252,
                    median_house_value=452600.0,
                    ocean_proximity="NEAR BAY",
                    request_id="test-123",
                )

                result = await call_predict_service(request)

                assert result.housing_price == 450000.0
                assert result.accuracy == 0.85
                assert result.request_id == "test-123"

    @pytest.mark.asyncio
    async def test_predict_service_http_error(self, test_settings):
        """Test handling of HTTP error from predict service"""
        with patch("config.settings.settings", test_settings):
            from app.main import call_predict_service
            from fastapi import HTTPException

            from services.common.models import InternalPredictionRequest

            mock_client = Mock()
            mock_response = Mock()
            mock_response.status_code = 500
            mock_response.text = "Internal server error"
            mock_client.post.return_value = mock_response

            with patch("services.api.main.http_client", mock_client):
                request = InternalPredictionRequest(
                    longitude=-122.23,
                    latitude=37.88,
                    housing_median_age=41.0,
                    total_rooms=880.0,
                    total_bedrooms=129.0,
                    population=322.0,
                    households=126.0,
                    median_income=8.3252,
                    median_house_value=452600.0,
                    ocean_proximity="NEAR BAY",
                )

                with pytest.raises(HTTPException) as exc_info:
                    await call_predict_service(request)

                assert exc_info.value.status_code == 500

    @pytest.mark.asyncio
    async def test_predict_service_connection_error(self, test_settings):
        """Test handling of connection error to predict service"""
        with patch("config.settings.settings", test_settings):
            import httpx
            from app.main import call_predict_service
            from fastapi import HTTPException

            from services.common.models import InternalPredictionRequest

            mock_client = Mock()
            mock_client.post.side_effect = httpx.RequestError("Connection failed")

            with patch("services.api.main.http_client", mock_client):
                request = InternalPredictionRequest(
                    longitude=-122.23,
                    latitude=37.88,
                    housing_median_age=41.0,
                    total_rooms=880.0,
                    total_bedrooms=129.0,
                    population=322.0,
                    households=126.0,
                    median_income=8.3252,
                    median_house_value=452600.0,
                    ocean_proximity="NEAR BAY",
                )

                with pytest.raises(HTTPException) as exc_info:
                    await call_predict_service(request)

                assert exc_info.value.status_code == 503


@pytest.mark.api
class TestRateLimiting:
    """Test rate limiting functionality"""

    @pytest.mark.asyncio
    async def test_rate_limit_not_exceeded(self, api_client, sample_prediction_request):
        """Test normal request within rate limit"""
        with patch("app.main.call_predict_service") as mock_call:
            from services.common.models import InternalPredictionResponse

            mock_call.return_value = InternalPredictionResponse(
                housing_price=450000.0,
                accuracy=0.85,
                request_id="test-123",
                processing_time=0.1,
            )

            response = await api_client.post(
                "/v1/predict", json=sample_prediction_request.dict()
            )

            assert response.status_code == status.HTTP_200_OK
