# MLOps Monitoring Setup

This directory contains the monitoring configuration for the MLOps project using Prometheus and Grafana.

## Components

### Prometheus Configuration
- **File**: `prometheus.yml`
- **Purpose**: Scrapes metrics from API and Predict services every 5 seconds
- **Endpoints**:
  - API service: `http://api:8000/metrics`
  - Predict service: `http://predict:8001/metrics`

### Grafana Configuration
- **Datasources**: `grafana-datasources.yml` - Configures Prometheus as the default data source
- **Dashboard Provisioning**: `grafana-dashboards.yml` - Automatically loads dashboards
- **Request Dashboard**: `grafana-dashboard-requests.json` - Main dashboard for HTTP requests and status codes

## Dashboard Features

The **MLOps HTTP Requests and Status Codes** dashboard provides:

### Key Metrics Visualized

1. **Total Requests per Service** - Current request rate for each service
2. **Request Rate by Status Code** - Time series showing request rates colored by HTTP status (2xx=green, 4xx=yellow, 5xx=red)
3. **Request Rate per Service over Time** - Historical view of request rates per service
4. **HTTP Status Code Distribution** - Pie chart showing proportion of different status codes
5. **Error Rate by Service** - Percentage of 4xx/5xx responses per service
6. **Request Duration (95th Percentile)** - Response time performance metrics
7. **Request Details by Endpoint** - Table view of all endpoints with request rates
8. **Rate Limit Violations** - Rate limiting metrics
9. **Service Health Status** - UP/DOWN status for each service

### Available Metrics

The dashboard uses these Prometheus metrics exposed by the services:

- `http_requests_total{method, endpoint, status_code, service}` - Total HTTP requests counter
- `http_request_duration_seconds{method, endpoint, service}` - Request duration histogram
- `rate_limit_exceeded_total{service}` - Rate limit violations counter
- `service_up{service}` - Service health status gauge

## Getting Started

### 1. Start the Services
```bash
docker-compose up -d
```

### 2. Access Grafana
- URL: http://localhost:3000
- Username: `admin`
- Password: `admin`

### 3. View the Dashboard
The "MLOps HTTP Requests and Status Codes" dashboard will be automatically loaded and available in the dashboard list.

### 4. Generate Test Traffic
To see metrics in action, you can make requests to the API:

```bash
# Health check
curl http://localhost:8000/health

# Prediction request
curl -X POST http://localhost:8000/v1/predict \
  -H "Content-Type: application/json" \
  -d '{
    "longitude": -122.23,
    "latitude": 37.88,
    "housing_median_age": 21.0,
    "total_rooms": 7099.0,
    "total_bedrooms": 1106.0,
    "population": 2401.0,
    "households": 1138.0,
    "median_income": 8.3014,
    "median_house_value": 358500.0,
    "ocean_proximity": "NEAR BAY"
  }'
```

### 5. Monitor Different Status Codes
- **200 responses**: Successful predictions
- **422 responses**: Validation errors (try sending invalid data)
- **429 responses**: Rate limit exceeded (make >5 requests within 60 seconds)
- **503 responses**: Service unavailable (if predict service is down)

## Prometheus Query Examples

You can create custom queries in Grafana or directly in Prometheus at http://localhost:9090:

```promql
# Request rate per service
sum(rate(http_requests_total[5m])) by (service)

# Error rate percentage
sum(rate(http_requests_total{status_code=~"^[45].*"}[5m])) by (service) / sum(rate(http_requests_total[5m])) by (service) * 100

# 95th percentile response time
histogram_quantile(0.95, sum(rate(http_request_duration_seconds_bucket[5m])) by (le, service))

# Status code breakdown
sum(rate(http_requests_total[5m])) by (status_code)
```

## Customization

### Adding New Panels
1. Edit `grafana-dashboard-requests.json`
2. Add new panel configuration
3. Restart Grafana: `docker-compose restart grafana`

### Adding Alerts
1. Configure alerting in Grafana UI
2. Set thresholds for error rates, response times, etc.
3. Configure notification channels (email, Slack, etc.)

## Troubleshooting

### No Data in Dashboard
1. Check if services are running: `docker compose ps`
2. Verify metrics endpoints: 
   - http://localhost:8000/metrics
   - http://localhost:8001/metrics
3. Check Prometheus targets: http://localhost:9090/targets

### Dashboard Not Loading
1. Check Grafana logs: `docker compose logs grafana`
2. Verify dashboard JSON syntax
3. Ensure provisioning volumes are mounted correctly

### Common Warnings (Harmless)

#### "Could not create user agent" Warning
```
logger=base.plugin.context t=... level=warn msg="Could not create user agent" error="invalid user agent format"
```
- **Status**: Harmless warning
- **Cause**: Plugin system in Grafana v11.3.0+ has issues with user agent formatting
- **Impact**: None - dashboard works perfectly
- **Action**: Can be safely ignored

### Verification Commands
```bash
# Check Grafana health
curl http://localhost:3000/api/health

# Check if dashboard is loaded
curl -s -u admin:admin http://localhost:3000/api/search | jq '.[].title'

# Test dashboard access
curl -s -u admin:admin http://localhost:3000/api/dashboards/uid/mlops-http-requests
```
