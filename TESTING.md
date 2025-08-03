# Testing Guide

This document provides comprehensive information about testing in the MLOps Housing Price Prediction project.

## Overview

The project uses `pytest` for unit testing with comprehensive test coverage across all major components:

- **API Service Tests** - Testing FastAPI endpoints, request/response handling, error scenarios
- **Prediction Service Tests** - Testing the ML model, prediction logic, and service endpoints  
- **Common Module Tests** - Testing shared models, metrics collection, logging configuration
- **Configuration Tests** - Testing settings and environment variable handling

## Test Structure

```
tests/
├── __init__.py                 # Test package initialization
├── conftest.py                 # Pytest configuration and shared fixtures
├── test_api.py                 # API service tests
├── test_predict.py             # Prediction service tests
├── test_models.py              # Data model tests
├── test_metrics.py             # Metrics collection tests
├── test_logging.py             # Logging configuration tests
└── test_config.py              # Configuration settings tests
```

## Installation

### Install Test Dependencies

```bash
# Install all dependencies including testing
pip install -r requirements.txt

# Or install test dependencies separately
pip install pytest pytest-asyncio pytest-mock pytest-cov httpx fakefs
```

## Running Tests

### Basic Usage

```bash
# Run all tests
pytest

# Run with verbose output
pytest -v

# Run quietly
pytest -q
```

### Using the Test Runner Script

We provide a convenient test runner script with additional options:

```bash
# Run all tests
python scripts/run_tests.py

# Run with coverage report
python scripts/run_tests.py --coverage

# Run only unit tests
python scripts/run_tests.py --unit

# Run only API tests
python scripts/run_tests.py --api

# Run only prediction service tests
python scripts/run_tests.py --predict

# Run tests with HTML coverage report
python scripts/run_tests.py --coverage --html-report

# Run specific test file
python scripts/run_tests.py --file test_api.py

# Run specific test
python scripts/run_tests.py --test test_api.py::TestAPIHealthEndpoints::test_health_endpoint

# Install dependencies and run tests
python scripts/run_tests.py --install-deps --coverage
```

### Test Categories

Tests are organized using pytest markers:

```bash
# Run only unit tests
pytest -m unit

# Run only API tests  
pytest -m api

# Run only prediction service tests
pytest -m predict

# Run integration tests (when available)
pytest -m integration

# Exclude slow tests
pytest -m "not slow"

# Run specific combinations
pytest -m "api or predict"
```

### Coverage Reports

```bash
# Run tests with coverage
pytest --cov=services --cov=config

# Generate HTML coverage report
pytest --cov=services --cov=config --cov-report=html

# Coverage with missing lines
pytest --cov=services --cov=config --cov-report=term-missing
```

## Test Configuration

### pytest.ini

The `pytest.ini` file contains pytest configuration:

```ini
[tool:pytest]
minversion = 6.0
addopts = -ra -q --strict-markers --strict-config
testpaths = tests
asyncio_mode = auto
markers =
    api: marks tests as API tests
    predict: marks tests as prediction service tests
    unit: marks tests as unit tests
    integration: marks tests as integration tests
    slow: marks tests as slow running
```

### Fixtures

Common test fixtures are defined in `tests/conftest.py`:

- `test_settings` - Test configuration with temporary database
- `test_db_path` - Temporary SQLite database path
- `sample_prediction_request` - Sample prediction request data
- `api_client` - FastAPI test client for API service
- `predict_client` - FastAPI test client for prediction service
- `mock_httpx_client` - Mock HTTP client for external calls

## Test Examples

### Testing API Endpoints

```python
@pytest.mark.asyncio
async def test_prediction_success(api_client, sample_prediction_request):
    """Test successful prediction"""
    response = await api_client.post(
        "/v1/predict",
        json=sample_prediction_request.dict()
    )
    
    assert response.status_code == 200
    data = response.json()
    assert "data" in data
    assert "housingPrice" in data["data"]
```

### Testing Models

```python
def test_prediction_request_validation():
    """Test validation of prediction request"""
    with pytest.raises(ValidationError):
        PredictionRequest(longitude="invalid")  # Should be float
```

### Testing with Mocks

```python
@patch('services.api.main.call_predict_service')
async def test_prediction_with_mock(mock_call, api_client):
    """Test prediction with mocked service call"""
    mock_call.return_value = InternalPredictionResponse(
        housing_price=450000.0,
        accuracy=0.85
    )
    
    response = await api_client.post("/v1/predict", json=request_data)
    assert response.status_code == 200
```

## Test Categories Explained

### Unit Tests (`@pytest.mark.unit`)

Test individual components in isolation:
- Model validation and serialization
- Business logic functions  
- Configuration handling
- Utility functions

### API Tests (`@pytest.mark.api`)

Test API endpoints and HTTP handling:
- Request/response validation
- Error handling
- Status codes
- Authentication (if implemented)

### Predict Tests (`@pytest.mark.predict`)

Test prediction service functionality:
- Model loading and prediction
- Feature processing
- Service endpoints
- Error scenarios

## Continuous Integration

### GitHub Actions Example

```yaml
name: Tests
on: [push, pull_request]

jobs:
  test:
    runs-on: ubuntu-latest
    steps:
    - uses: actions/checkout@v2
    - name: Set up Python
      uses: actions/setup-python@v2
      with:
        python-version: 3.9
    - name: Install dependencies
      run: pip install -r requirements.txt
    - name: Run tests
      run: python scripts/run_tests.py --coverage
    - name: Upload coverage
      uses: codecov/codecov-action@v1
```

## Best Practices

### Writing Tests

1. **Use descriptive test names** - Test names should explain what is being tested
2. **One assertion per test** - Keep tests focused and simple
3. **Use fixtures for setup** - Avoid code duplication in test setup
4. **Test edge cases** - Include boundary conditions and error scenarios
5. **Mock external dependencies** - Isolate units under test

### Test Organization

1. **Group related tests** - Use test classes to group related functionality
2. **Use markers** - Tag tests with appropriate markers for filtering
3. **Separate test types** - Keep unit tests separate from integration tests
4. **Consistent naming** - Follow consistent naming conventions

### Performance

1. **Use appropriate fixtures scope** - Session, module, or function scope as needed
2. **Mock expensive operations** - Don't make real HTTP calls or database operations in unit tests
3. **Parallel execution** - Consider using pytest-xdist for parallel test execution

## Debugging Tests

### Running Individual Tests

```bash
# Run specific test with verbose output
pytest tests/test_api.py::TestAPIHealthEndpoints::test_health_endpoint -v

# Run with pdb debugger
pytest tests/test_api.py::test_specific_function --pdb

# Run with print statements visible
pytest tests/test_api.py -s
```

### Common Issues

1. **Import Errors** - Ensure PYTHONPATH includes project root
2. **Async Test Issues** - Use `@pytest.mark.asyncio` for async tests
3. **Mock Problems** - Verify mock paths match actual import paths
4. **Database Issues** - Use temporary databases in tests

## Coverage Goals

- **Minimum Coverage**: 80%
- **Target Coverage**: 90%+
- **Critical Paths**: 100% (authentication, payments, data validation)

### Checking Coverage

```bash
# View coverage report
pytest --cov=services --cov=config --cov-report=term-missing

# Generate HTML report for detailed analysis
pytest --cov=services --cov=config --cov-report=html
open htmlcov/index.html
```

## Integration with Development Workflow

### Pre-commit Hooks

Add to `.pre-commit-config.yaml`:

```yaml
repos:
  - repo: local
    hooks:
      - id: pytest
        name: pytest
        entry: python scripts/run_tests.py --quiet
        language: system
        pass_filenames: false
        always_run: true
```

### IDE Integration

Most IDEs support pytest integration:

- **VS Code**: Python extension with pytest discovery
- **PyCharm**: Built-in pytest runner
- **Vim**: pytest.vim plugin

## Troubleshooting

### Common Test Failures

1. **Import Errors**: Check PYTHONPATH and module structure
2. **Database Errors**: Ensure test database is properly isolated
3. **Mock Errors**: Verify mock patch targets are correct
4. **Async Errors**: Ensure proper async/await usage

### Performance Issues

1. **Slow Tests**: Use mocks instead of real external calls
2. **Memory Usage**: Clean up resources in test teardown
3. **Parallel Execution**: Consider pytest-xdist for faster execution

## Contributing

When adding new features:

1. Write tests first (TDD approach)
2. Ensure all tests pass
3. Maintain or improve coverage
4. Add appropriate test markers
5. Update documentation if needed

For bug fixes:

1. Write a test that reproduces the bug
2. Fix the bug
3. Ensure the test passes
4. Check that no other tests are broken