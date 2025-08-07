"""
Unit tests for common models
"""
import pytest
from pydantic import ValidationError

from services.common.models import (
    ErrorInfo,
    ErrorResponse,
    HealthResponse,
    InternalPredictionRequest,
    InternalPredictionResponse,
    OceanProximity,
    PredictionData,
    PredictionRequest,
    PredictionSuccess,
    ReadinessResponse,
)


@pytest.mark.unit
class TestOceanProximity:
    """Test OceanProximity enum"""

    def test_ocean_proximity_values(self):
        """Test all ocean proximity enum values"""
        assert OceanProximity.LESS_THAN_1H_OCEAN == "<1H OCEAN"
        assert OceanProximity.INLAND == "INLAND"
        assert OceanProximity.ISLAND == "ISLAND"
        assert OceanProximity.NEAR_BAY == "NEAR BAY"
        assert OceanProximity.NEAR_OCEAN == "NEAR OCEAN"


@pytest.mark.unit
class TestPredictionRequest:
    """Test PredictionRequest model"""

    def test_valid_prediction_request(self):
        """Test creating valid prediction request"""
        request = PredictionRequest(
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

        assert request.longitude == -122.23
        assert request.latitude == 37.88
        assert request.oceanProximity == OceanProximity.NEAR_BAY

    def test_prediction_request_validation_errors(self):
        """Test validation errors for invalid prediction request"""
        # Test missing required fields
        with pytest.raises(ValidationError):
            PredictionRequest(longitude=-122.23)

        # Test invalid ocean proximity
        with pytest.raises(ValidationError):
            PredictionRequest(
                longitude=-122.23,
                latitude=37.88,
                housingMedianAge=41.0,
                totalRooms=880.0,
                totalBedrooms=129.0,
                population=322.0,
                households=126.0,
                medianIncome=8.3252,
                medianHouseValue=452600.0,
                oceanProximity="INVALID_PROXIMITY",
            )

        # Test invalid data types
        with pytest.raises(ValidationError):
            PredictionRequest(
                longitude="invalid",  # Should be float
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

    def test_prediction_request_json_serialization(self):
        """Test JSON serialization of prediction request"""
        request = PredictionRequest(
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

        json_data = request.dict()
        assert isinstance(json_data, dict)
        assert json_data["longitude"] == -122.23
        assert json_data["oceanProximity"] == "NEAR BAY"


@pytest.mark.unit
class TestPredictionResponse:
    """Test prediction response models"""

    def test_prediction_data(self):
        """Test PredictionData model"""
        data = PredictionData(housingPrice=450000.0, accuracy=0.85)

        assert data.housingPrice == 450000.0
        assert data.accuracy == 0.85

    def test_prediction_data_optional_accuracy(self):
        """Test PredictionData with optional accuracy"""
        data = PredictionData(housingPrice=450000.0)

        assert data.housingPrice == 450000.0
        assert data.accuracy is None

    def test_prediction_success(self):
        """Test PredictionSuccess model"""
        data = PredictionData(housingPrice=450000.0, accuracy=0.85)
        success = PredictionSuccess(data=data)

        assert success.data.housingPrice == 450000.0
        assert success.data.accuracy == 0.85


@pytest.mark.unit
class TestInternalModels:
    """Test internal communication models"""

    def test_internal_prediction_request(self):
        """Test InternalPredictionRequest model"""
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

        assert request.longitude == -122.23
        assert request.ocean_proximity == "NEAR BAY"
        assert request.request_id == "test-123"

    def test_from_prediction_request_conversion(self):
        """Test conversion from external to internal format"""
        external_request = PredictionRequest(
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

        internal_request = InternalPredictionRequest.from_prediction_request(
            external_request, "test-123"
        )

        assert internal_request.longitude == external_request.longitude
        assert internal_request.latitude == external_request.latitude
        assert internal_request.housing_median_age == external_request.housingMedianAge
        assert internal_request.total_rooms == external_request.totalRooms
        assert internal_request.ocean_proximity == external_request.oceanProximity.value
        assert internal_request.request_id == "test-123"

    def test_internal_prediction_response(self):
        """Test InternalPredictionResponse model"""
        response = InternalPredictionResponse(
            housing_price=450000.0,
            accuracy=0.85,
            request_id="test-123",
            processing_time=0.1,
        )

        assert response.housing_price == 450000.0
        assert response.accuracy == 0.85
        assert response.request_id == "test-123"
        assert response.processing_time == 0.1

    def test_to_prediction_success_conversion(self):
        """Test conversion from internal to external format"""
        internal_response = InternalPredictionResponse(
            housing_price=450000.0,
            accuracy=0.85,
            request_id="test-123",
            processing_time=0.1,
        )

        external_response = internal_response.to_prediction_success()

        assert isinstance(external_response, PredictionSuccess)
        assert external_response.data.housingPrice == internal_response.housing_price
        assert external_response.data.accuracy == internal_response.accuracy


@pytest.mark.unit
class TestHealthModels:
    """Test health check models"""

    def test_health_response(self):
        """Test HealthResponse model"""
        response = HealthResponse(
            status="OK", timestamp="2023-01-01T00:00:00", service="api"
        )

        assert response.status == "OK"
        assert response.timestamp == "2023-01-01T00:00:00"
        assert response.service == "api"

    def test_health_response_defaults(self):
        """Test HealthResponse with default values"""
        response = HealthResponse()

        assert response.status == "OK"
        assert response.timestamp is None
        assert response.service is None

    def test_readiness_response(self):
        """Test ReadinessResponse model"""
        checks = {"database": "OK", "predictService": "OK"}
        response = ReadinessResponse(ready=True, checks=checks)

        assert response.ready is True
        assert response.checks == checks

    def test_readiness_response_without_checks(self):
        """Test ReadinessResponse without checks"""
        response = ReadinessResponse(ready=False)

        assert response.ready is False
        assert response.checks is None


@pytest.mark.unit
class TestErrorModels:
    """Test error response models"""

    def test_error_info(self):
        """Test ErrorInfo model"""
        error = ErrorInfo(code="VALIDATION_ERROR", message="Invalid input")

        assert error.code == "VALIDATION_ERROR"
        assert error.message == "Invalid input"

    def test_error_response(self):
        """Test ErrorResponse model"""
        error_info = ErrorInfo(code="INTERNAL_ERROR", message="Server error")
        response = ErrorResponse(error=error_info)

        assert response.error.code == "INTERNAL_ERROR"
        assert response.error.message == "Server error"


@pytest.mark.unit
class TestModelValidation:
    """Test advanced model validation scenarios"""

    def test_prediction_request_boundary_values(self):
        """Test prediction request with boundary values"""
        # Test with very small values
        request = PredictionRequest(
            longitude=-180.0,
            latitude=-90.0,
            housingMedianAge=0.0,
            totalRooms=1.0,
            totalBedrooms=0.0,
            population=1.0,
            households=1.0,
            medianIncome=0.1,
            medianHouseValue=1.0,
            oceanProximity=OceanProximity.INLAND,
        )

        assert request.longitude == -180.0
        assert request.totalBedrooms == 0.0

        # Test with very large values
        request = PredictionRequest(
            longitude=180.0,
            latitude=90.0,
            housingMedianAge=1000.0,
            totalRooms=999999.0,
            totalBedrooms=999999.0,
            population=999999.0,
            households=999999.0,
            medianIncome=999999.0,
            medianHouseValue=999999999.0,
            oceanProximity=OceanProximity.ISLAND,
        )

        assert request.longitude == 180.0
        assert request.medianHouseValue == 999999999.0

    def test_internal_request_optional_fields(self):
        """Test internal request with optional fields"""
        # Without request_id
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

        assert request.request_id is None

        # With request_id
        request_with_id = InternalPredictionRequest(
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
            request_id="custom-id",
        )

        assert request_with_id.request_id == "custom-id"
