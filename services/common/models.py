from pydantic import BaseModel, Field
from typing import Optional, Literal
from enum import Enum


class OceanProximity(str, Enum):
    """Enum for ocean proximity values"""

    LESS_THAN_1H_OCEAN = "<1H OCEAN"
    INLAND = "INLAND"
    ISLAND = "ISLAND"
    NEAR_BAY = "NEAR BAY"
    NEAR_OCEAN = "NEAR OCEAN"


class PredictionRequest(BaseModel):
    """Request model for housing price prediction"""

    longitude: float = Field(..., description="Longitude coordinate")
    latitude: float = Field(..., description="Latitude coordinate")
    housingMedianAge: float = Field(
        ..., description="Median age of housing in the area"
    )
    totalRooms: float = Field(..., description="Total number of rooms")
    totalBedrooms: float = Field(..., description="Total number of bedrooms")
    population: float = Field(..., description="Population in the area")
    households: float = Field(..., description="Number of households")
    medianIncome: float = Field(..., description="Median income in the area")
    medianHouseValue: float = Field(..., description="Median house value in the area")
    oceanProximity: OceanProximity = Field(..., description="Proximity to ocean")

    class Config:
        schema_extra = {
            "example": {
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
        }


class PredictionData(BaseModel):
    """Prediction result data"""

    housingPrice: float = Field(..., description="Predicted housing price")
    accuracy: Optional[float] = Field(None, description="Model accuracy (optional)")


class PredictionSuccess(BaseModel):
    """Successful prediction response"""

    data: PredictionData


class ErrorInfo(BaseModel):
    """Error information"""

    code: str = Field(..., description="Error code")
    message: str = Field(..., description="Error message")


class ErrorResponse(BaseModel):
    """Error response model"""

    error: ErrorInfo


class HealthResponse(BaseModel):
    """Health check response"""

    status: str = Field(default="OK")
    timestamp: Optional[str] = None
    service: Optional[str] = None


class ReadinessResponse(BaseModel):
    """Readiness check response"""

    ready: bool = Field(..., description="Service readiness status")
    checks: Optional[dict] = Field(None, description="Individual service checks")


# Internal communication models between API and Predict services
class InternalPredictionRequest(BaseModel):
    """Internal request model for communication between services"""

    longitude: float
    latitude: float
    housing_median_age: float
    total_rooms: float
    total_bedrooms: float
    population: float
    households: float
    median_income: float
    median_house_value: float
    ocean_proximity: str
    request_id: Optional[str] = None

    @classmethod
    def from_prediction_request(cls, req: PredictionRequest, request_id: str = None):
        """Convert external API request to internal format"""
        return cls(
            longitude=req.longitude,
            latitude=req.latitude,
            housing_median_age=req.housingMedianAge,
            total_rooms=req.totalRooms,
            total_bedrooms=req.totalBedrooms,
            population=req.population,
            households=req.households,
            median_income=req.medianIncome,
            median_house_value=req.medianHouseValue,
            ocean_proximity=req.oceanProximity.value,
            request_id=request_id,
        )


class InternalPredictionResponse(BaseModel):
    """Internal response model for communication between services"""

    housing_price: float
    accuracy: Optional[float] = None
    request_id: Optional[str] = None
    processing_time: Optional[float] = None

    def to_prediction_success(self) -> PredictionSuccess:
        """Convert internal response to external API format"""
        return PredictionSuccess(
            data=PredictionData(housingPrice=self.housing_price, accuracy=self.accuracy)
        )


# Inference models
class InferenceRequest(BaseModel):
    """Request model for inference"""
    
    MedInc: float = Field(..., description="Median income in the area")
    HouseAge: float = Field(..., description="Median age of housing in the area")
    AveRooms: float = Field(..., description="Average number of rooms")
    AveBedrms: float = Field(..., description="Average number of bedrooms")
    Population: float = Field(..., description="Population in the area")
    AveOccup: float = Field(..., description="Average occupancy")
    Latitude: float = Field(..., description="Latitude coordinate")
    Longitude: float = Field(..., description="Longitude coordinate")

    class Config:
        schema_extra = {
            "example": {
                "MedInc": -0.3261960037692928,
                "HouseAge": 0.3484902466663322,
                "AveRooms": -0.1749164614622689,
                "AveBedrms": -0.2083654336540427,
                "Population": 0.7682762831665109,
                "AveOccup": 0.0513760919421774,
                "Latitude": -1.3728111990669665,
                "Longitude": 1.2725865624715638
            }
        }


class InferenceData(BaseModel):
    """Inference result data"""
    
    predictions: list = Field(..., description="List of predictions")
    request_id: Optional[str] = Field(None, description="Request ID for tracking")


class InferenceResponse(BaseModel):
    """Successful inference response"""
    
    data: InferenceData
    processing_time: Optional[float] = Field(None, description="Processing time in seconds")
