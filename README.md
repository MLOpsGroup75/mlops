# MLOps Housing Price Prediction Service

A comprehensive MLOps solution for housing price prediction with microservices architecture, monitoring, and logging.

## Architecture

This project implements a two-service architecture:

- **API Service** (`services/api/`): External-facing REST API with rate limiting and request validation
- **Predict Service** (`services/predict/`): Internal ML model serving service

Both services communicate via HTTP and include comprehensive logging, metrics collection, and health checks.

## Features

- **FastAPI-based microservices** with automatic OpenAPI documentation
- **Rate limiting** (configurable requests per minute) with proper HTTP 429 responses
- **Structured logging** with SQLite storage and console output
- **Prometheus metrics** for monitoring
- **Health and readiness checks** for Kubernetes deployment
- **Docker containerization** with multi-service orchestration
- **Request/response logging** for all API calls
- **Dummy ML model** for housing price prediction

## API Endpoints

### API Service (Port 8000)

- `POST /v1/predict` - Housing price prediction
- `GET /metrics` - Prometheus metrics
- `GET /health` - Health check
- `GET /readiness` - Readiness check for Kubernetes

### Predict Service (Port 8001)

- `POST /predict` - Internal prediction endpoint
- `GET /metrics` - Prometheus metrics
- `GET /health` - Health check
- `GET /model/info` - Model information
- `POST /model/reload` - Reload model

## Quick Start

### Using Docker Compose (Recommended)

1. **Build and start all services:**
   ```bash
   docker-compose up --build
   ```

2. **Access the services:**
   - API Service: http://localhost:8000
   - API Documentation: http://localhost:8000/docs
   - Predict Service: http://localhost:8001
   - Prometheus: http://localhost:9090
   - Grafana: http://localhost:3000 (admin/admin)

### Using Local Python

1. **Install dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

2. **Start services:**
   ```bash
   ./scripts/run_services.sh
   ```

3. **Test the API:**
   ```bash
   python scripts/test_api.py
   ```

## Configuration

Configuration is managed through environment variables or `.env` file:

```bash
# Copy example configuration
cp .env.example .env
```

Key configuration options:
- `RATE_LIMIT_REQUESTS`: Requests per minute (default: 5)
- `RATE_LIMIT_WINDOW`: Rate limit window in seconds (default: 60)
- `LOG_LEVEL`: Logging level (default: INFO)
- `MODEL_ACCURACY`: Model accuracy for responses (default: 0.85)

## Example API Usage

### Prediction Request

```bash
curl -X POST "http://localhost:8000/v1/predict" \
  -H "Content-Type: application/json" \
  -d '{
    "longitude": -122.23,
    "latitude": 37.88,
    "housingMedianAge": 41.0,
    "totalRooms": 880.0,
    "totalBedrooms": 129.0,
    "population": 322.0,
    "households": 126.0,
    "medianIncome": 8.3252,
    "medianHouseValue": 452600.0,
    "oceanProximity": "NEAR BAY"
  }'
```

### Response

```json
{
  "data": {
    "housingPrice": 450000.0,
    "accuracy": 0.85
  }
}
```

## Project Structure

```
mlops/
├── services/
│   ├── api/                    # API service
│   ├── predict/                # Predict service
│   └── common/                 # Shared utilities
├── config/                     # Configuration files
├── data/                       # Data directories
│   ├── raw/                    # Raw data
│   ├── processed/              # Processed data
│   └── features/               # Feature data
├── model/                      # Model artifacts
│   ├── artifacts/              # Trained models
│   ├── checkpoints/            # Model checkpoints
│   └── configs/                # Model configurations
├── pipeline/                   # ML pipelines
│   ├── training/               # Training pipelines
│   ├── inference/              # Inference pipelines
│   └── monitoring/             # Model monitoring
├── docker/                     # Docker configurations
├── monitoring/                 # Prometheus/Grafana configs
├── scripts/                    # Utility scripts
├── logs/                       # Log files and database
├── swagger.yaml                # OpenAPI specification
├── docker-compose.yml          # Docker orchestration
└── requirements.txt            # Python dependencies
```

## Logging

The system uses structured logging with:

- **Console output**: Human-readable logs with timestamps
- **SQLite database**: Structured logs in `logs/app_logs.db`
- **Request tracking**: All requests have unique IDs for tracing

### Health Endpoint Logging Configuration

By default, logging for health and metric endpoints (`/health`, `/readiness`, `/metrics`) is **disabled** to reduce log noise. You can enable it using:

**Environment Variable:**
```bash
export LOG_HEALTH_ENDPOINTS=true
```

**Or in `config/settings.py`:**
```python
log_health_endpoints: bool = True  # Enable logging for health endpoints
```

**Affected endpoints:**
- `/health` - Basic health check
- `/readiness` - Kubernetes readiness probe  
- `/metrics` - Prometheus metrics export

Regular API endpoints (like `/v1/predict`) are always logged regardless of this setting.

### JSON Body Logging Configuration

Request and response bodies are logged in JSON format for better debugging and monitoring:

**Environment Variables:**
```bash
export LOG_REQUEST_BODY=true          # Enable JSON body logging (default: true)
export MAX_BODY_LOG_SIZE=10000        # Max body size to log in bytes (default: 10000)
```

**Or in `config/settings.py`:**
```python
log_request_body: bool = True         # Enable logging of request/response bodies
max_body_log_size: int = 10000        # Maximum body size to log (in bytes)
```

**Features:**
- **JSON parsing**: Request/response bodies are parsed and logged as structured JSON
- **Size limits**: Large bodies are truncated with size information
- **Content type filtering**: Only JSON content is parsed; other types show metadata
- **Request body restoration**: Request bodies are made available to endpoints after logging
- **Streaming support**: Handles both regular and streaming responses

### Log Database Schema

```sql
CREATE TABLE logs (
    id INTEGER PRIMARY KEY,
    timestamp TEXT,
    level TEXT,
    logger_name TEXT,
    message TEXT,
    service TEXT,
    request_id TEXT,
    extra_data TEXT,
    created_at DATETIME DEFAULT CURRENT_TIMESTAMP
);
```

## Monitoring

### Prometheus Metrics

- `http_requests_total`: Total HTTP requests by method, endpoint, status
- `http_request_duration_seconds`: Request duration histogram
- `prediction_requests_total`: Total prediction requests
- `prediction_duration_seconds`: Prediction duration histogram
- `rate_limit_exceeded_total`: Rate limit violations
- `service_up`: Service availability gauge

### Grafana Dashboards

Access Grafana at http://localhost:3000 (admin/admin) to view:
- Request rates and response times
- Error rates and status codes
- Prediction accuracy and performance
- Service health and availability

## Development

### Adding New Features

1. **Models**: Add Pydantic models in `services/common/models.py`
2. **Endpoints**: Add new endpoints to respective service files
3. **Metrics**: Add custom metrics in `services/common/metrics.py`
4. **Configuration**: Update `config/settings.py` for new settings

### Testing

Run the test suite:
```bash
python scripts/test_api.py
```

### Linting and Code Quality

```bash
# Install development dependencies
pip install black flake8 mypy

# Format code
black .

# Lint code
flake8 services/

# Type checking
mypy services/
```

## Deployment

### Kubernetes

The services include readiness and liveness probes for Kubernetes:

```yaml
livenessProbe:
  httpGet:
    path: /health
    port: 8000
  initialDelaySeconds: 30
  periodSeconds: 10

readinessProbe:
  httpGet:
    path: /readiness
    port: 8000
  initialDelaySeconds: 5
  periodSeconds: 5
```

### Environment Variables

For production deployment, set these environment variables:
- `API_DEBUG=false`
- `LOG_LEVEL=WARNING`
- `RATE_LIMIT_REQUESTS=100`
- `PREDICT_URL=http://predict-service:8001`

## Future Enhancements

The project structure supports future additions:

- **Real ML Models**: Replace dummy model in `services/predict/main.py`
- **Data Pipeline**: Add training/inference pipelines in `pipeline/`
- **Model Versioning**: Implement model registry and versioning
- **A/B Testing**: Add model comparison and traffic splitting
- **Advanced Monitoring**: Add drift detection and model performance tracking

## Troubleshooting

### Common Issues

1. **Port conflicts**: Ensure ports 8000, 8001, 9090, 3000 are available
2. **Permission errors**: Check file permissions for log directory
3. **Rate limiting**: Adjust `RATE_LIMIT_REQUESTS` for development

### Logs

Check service logs:
```bash
# Docker Compose
docker-compose logs api
docker-compose logs predict

# Local development
tail -f logs/app_logs.db  # View log database
```

### Health Checks

Verify service health:
```bash
curl http://localhost:8000/health
curl http://localhost:8001/health
curl http://localhost:8000/readiness
```

## License

This project is licensed under the MIT License.