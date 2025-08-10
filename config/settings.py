from pydantic_settings import BaseSettings
from typing import Optional
import os


class Settings(BaseSettings):
    # API Service Settings
    api_host: str = "0.0.0.0"
    api_port: int = 8000
    api_debug: bool = False

    # Predict Service Settings
    predict_host: str = "0.0.0.0"
    predict_port: int = 8001
    predict_url: str = "http://localhost:8001"

    # Rate Limiting
    rate_limit_requests: int = 150
    rate_limit_window: int = 60  # seconds

    # Logging Settings
    log_level: str = "INFO"
    log_to_file: bool = True
    log_db_path: str = "logs/app_logs.db"
    log_all_endpoints: bool = (
        False  # Enable logging for /health, /readiness, /metrics endpoints
    )
    log_request_body: bool = (
        True  # Enable logging of request/response bodies in JSON format
    )
    max_body_log_size: int = 10000  # Maximum body size to log (in bytes)

    # Database Settings
    database_url: str = "sqlite:///./logs/app_logs.db"

    # Model Settings
    model_path: str = "model/artifacts/housing_model.pkl"
    model_accuracy: Optional[float] = 0.85

    # Databricks Settings
    databricks_endpoint_url: str = "https://dbc-87ef0a7b-f01d.cloud.databricks.com/serving-endpoints/mlops/invocations"
    databricks_token: Optional[str] = None

    # Service Settings
    service_name: str = "mlops-housing-api"
    service_version: str = "1.0.0"

    class Config:
        env_file = ".env"
        case_sensitive = False

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        # Load Databricks params from environment variable
        if os.environ.get("DATABRICKS_ENDPOINT_URL") != "":
            self.databricks_endpoint_url = os.environ.get("DATABRICKS_ENDPOINT_URL")

        if not self.databricks_token:
            self.databricks_token = os.environ.get("DATABRICKS_TOKEN")


settings = Settings()
