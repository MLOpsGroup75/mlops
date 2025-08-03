"""
Unit tests for prediction service
"""
import pytest
from unittest.mock import Mock, patch, MagicMock
from fastapi import status

from app.main import DummyHousingModel, prepare_features, load_model
from app.common.models import InternalPredictionRequest


@pytest.mark.predict
class TestDummyHousingModel:
    """Test the DummyHousingModel class"""
    
    def test_model_initialization(self):
        """Test model initialization with default accuracy"""
        model = DummyHousingModel()
        assert model.accuracy == 0.85
        assert model.base_price == 200000
        assert isinstance(model.feature_weights, dict)
    
    def test_model_initialization_custom_accuracy(self):
        """Test model initialization with custom accuracy"""
        model = DummyHousingModel(accuracy=0.9)
        assert model.accuracy == 0.9
    
    def test_get_accuracy(self):
        """Test get_accuracy method"""
        model = DummyHousingModel(accuracy=0.75)
        assert model.get_accuracy() == 0.75
    
    def test_predict_basic_features(self, sample_features):
        """Test prediction with basic features"""
        model = DummyHousingModel()
        
        # Mock numpy random to make prediction deterministic
        with patch('numpy.random.normal', return_value=0):
            price = model.predict(sample_features)
            
            assert isinstance(price, float)
            assert price > 0
            assert price >= 50000  # Minimum price check
    
    def test_predict_ocean_proximity_variations(self):
        """Test prediction with different ocean proximity values"""
        model = DummyHousingModel()
        
        base_features = {
            'longitude': -122.23,
            'latitude': 37.88,
            'housing_median_age': 41.0,
            'total_rooms': 880.0,
            'total_bedrooms': 129.0,
            'population': 322.0,
            'households': 126.0,
            'median_income': 8.3252,
            'median_house_value': 452600.0,
        }
        
        with patch('numpy.random.normal', return_value=0):
            # Test different ocean proximity values
            for proximity in ['<1H OCEAN', 'INLAND', 'ISLAND', 'NEAR BAY', 'NEAR OCEAN']:
                features = {**base_features, 'ocean_proximity': proximity}
                price = model.predict(features)
                assert isinstance(price, float)
                assert price > 0
    
    def test_predict_minimum_price_enforcement(self):
        """Test that prediction enforces minimum price"""
        model = DummyHousingModel()
        
        # Use very negative values to try to get below minimum
        features = {
            'longitude': -180,  # Very negative
            'latitude': -90,
            'housing_median_age': 100,
            'total_rooms': 1,
            'total_bedrooms': 1,
            'population': 1,
            'households': 1,
            'median_income': 0.1,
            'median_house_value': 1,
            'ocean_proximity': 'INLAND'
        }
        
        with patch('numpy.random.normal', return_value=-1000000):  # Large negative noise
            price = model.predict(features)
            assert price >= 50000  # Should enforce minimum


@pytest.mark.predict
class TestPredictServiceHelpers:
    """Test helper functions in predict service"""
    
    def test_prepare_features(self):
        """Test prepare_features function"""
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
            request_id="test-123"
        )
        
        features = prepare_features(request)
        
        assert features['longitude'] == -122.23
        assert features['latitude'] == 37.88
        assert features['housing_median_age'] == 41.0
        assert features['ocean_proximity'] == "NEAR BAY"
        assert len(features) == 10  # All features present
    
    def test_load_model_success(self, test_settings):
        """Test successful model loading"""
        with patch('app.config.settings.settings', test_settings):
            # Mock the global model variable
            with patch('app.main.model', None):
                result = load_model()
                assert result is True
    
    def test_load_model_failure(self, test_settings):
        """Test model loading failure"""
        with patch('app.config.settings.settings', test_settings):
            with patch('app.main.DummyHousingModel') as mock_model:
                mock_model.side_effect = Exception("Model loading failed")
                
                result = load_model()
                assert result is False


@pytest.mark.predict
class TestPredictServiceEndpoints:
    """Test prediction service endpoints"""
    
    @pytest.mark.asyncio
    async def test_health_endpoint_model_loaded(self, predict_client):
        """Test health endpoint when model is loaded"""
        with patch('app.main.model', Mock()):
            response = await predict_client.get("/health")
            
            assert response.status_code == status.HTTP_200_OK
            data = response.json()
            assert data["status"] == "OK"
            assert data["service"] == "predict"
    
    @pytest.mark.asyncio
    async def test_health_endpoint_model_not_loaded(self, predict_client):
        """Test health endpoint when model is not loaded"""
        with patch('app.main.model', None):
            response = await predict_client.get("/health")
            
            assert response.status_code == status.HTTP_200_OK
            data = response.json()
            assert data["status"] == "Model not loaded"
    
    @pytest.mark.asyncio
    async def test_predict_endpoint_success(self, predict_client):
        """Test successful prediction endpoint"""
        mock_model = Mock()
        mock_model.predict.return_value = 450000.0
        mock_model.get_accuracy.return_value = 0.85
        
        with patch('app.main.model', mock_model):
            request_data = {
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
                "request_id": "test-123"
            }
            
            response = await predict_client.post("/predict", json=request_data)
            
            assert response.status_code == status.HTTP_200_OK
            data = response.json()
            assert data["housing_price"] == 450000.0
            assert data["accuracy"] == 0.85
            assert data["request_id"] == "test-123"
            assert "processing_time" in data
    
    @pytest.mark.asyncio
    async def test_predict_endpoint_model_not_loaded(self, predict_client):
        """Test prediction endpoint when model is not loaded"""
        with patch('app.main.model', None):
            request_data = {
                "longitude": -122.23,
                "latitude": 37.88,
                "housing_median_age": 41.0,
                "total_rooms": 880.0,
                "total_bedrooms": 129.0,
                "population": 322.0,
                "households": 126.0,
                "median_income": 8.3252,
                "median_house_value": 452600.0,
                "ocean_proximity": "NEAR BAY"
            }
            
            response = await predict_client.post("/predict", json=request_data)
            
            assert response.status_code == status.HTTP_503_SERVICE_UNAVAILABLE
    
    @pytest.mark.asyncio
    async def test_predict_endpoint_model_error(self, predict_client):
        """Test prediction endpoint when model raises error"""
        mock_model = Mock()
        mock_model.predict.side_effect = Exception("Model prediction failed")
        
        with patch('app.main.model', mock_model):
            request_data = {
                "longitude": -122.23,
                "latitude": 37.88,
                "housing_median_age": 41.0,
                "total_rooms": 880.0,
                "total_bedrooms": 129.0,
                "population": 322.0,
                "households": 126.0,
                "median_income": 8.3252,
                "median_house_value": 452600.0,
                "ocean_proximity": "NEAR BAY"
            }
            
            response = await predict_client.post("/predict", json=request_data)
            
            assert response.status_code == status.HTTP_500_INTERNAL_SERVER_ERROR
    
    @pytest.mark.asyncio
    async def test_model_info_endpoint_success(self, predict_client):
        """Test model info endpoint when model is loaded"""
        mock_model = Mock()
        mock_model.get_accuracy.return_value = 0.85
        type(mock_model).__name__ = "DummyHousingModel"
        
        with patch('app.main.model', mock_model):
            response = await predict_client.get("/model/info")
            
            assert response.status_code == status.HTTP_200_OK
            data = response.json()
            assert "model_type" in data
            assert data["accuracy"] == 0.85
            assert "features" in data
            assert "loaded_at" in data
    
    @pytest.mark.asyncio
    async def test_model_info_endpoint_no_model(self, predict_client):
        """Test model info endpoint when model is not loaded"""
        with patch('app.main.model', None):
            response = await predict_client.get("/model/info")
            
            assert response.status_code == status.HTTP_503_SERVICE_UNAVAILABLE
    
    @pytest.mark.asyncio
    async def test_reload_model_endpoint_success(self, predict_client):
        """Test model reload endpoint success"""
        with patch('app.main.load_model', return_value=True):
            response = await predict_client.post("/model/reload")
            
            assert response.status_code == status.HTTP_200_OK
            data = response.json()
            assert data["status"] == "Model reloaded successfully"
    
    @pytest.mark.asyncio
    async def test_reload_model_endpoint_failure(self, predict_client):
        """Test model reload endpoint failure"""
        with patch('app.main.load_model', return_value=False):
            response = await predict_client.post("/model/reload")
            
            assert response.status_code == status.HTTP_500_INTERNAL_SERVER_ERROR
    
    @pytest.mark.asyncio
    async def test_metrics_endpoint(self, predict_client):
        """Test metrics endpoint"""
        response = await predict_client.get("/metrics")
        
        assert response.status_code == status.HTTP_200_OK
        assert "text/plain" in response.headers["content-type"]