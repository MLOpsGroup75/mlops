# MLOps CI/CD Pipeline Architecture

This repository implements a sophisticated CI/CD pipeline system that automatically triggers different workflows based on the files that change. This path-based triggering ensures efficient resource usage and faster feedback loops.

## 🏗️ Pipeline Architecture

### Path-Based Pipeline Triggering

Our CI/CD system uses GitHub Actions path filters to trigger specific pipelines based on which directories contain changes:

```
Repository Structure:
├── services/         → Triggers Services Pipeline
├── data/             → Triggers Data Pipeline  
├── model/            → Triggers Model Pipeline
└── infrastructure/   → Triggers Main Pipeline (fallback)
```

## 🚀 Pipeline Details

### 1. CI/CD Pipeline with Docker (`service-pipeline.yml`)

**Triggers when:** Changes in `services/**` directory

**Includes:**
- ✅ **Testing**: Python 3.11 testing with pytest
- 🔍 **Code Quality**: Linting with flake8, coverage reporting
- 🐳 **Build**: Docker image building and pushing to Docker Hub
- 🔗 **Integration**: Comprehensive integration testing with health checks
- 🔒 **Security**: Trivy vulnerability scanning with SARIF reports
- 🚀 **Deploy**: Automated EKS deployment via ArgoCD with GitOps

**Services Included:**
- API Service (`services/api/`) - Port 8000
- Predict Service (`services/predict/`) - Port 8001

**Advanced Features:**
- Multi-service Docker builds with proper tagging
- Container health monitoring and retry logic
- GitOps deployment with manifest updates
- ArgoCD automatic synchronization

### 2. Data Pipeline (`data-pipeline.yml`)

**Triggers when:** Changes in `data/**` directory or data-related scripts

**Includes:**
- ✅ **Data Validation**: Schema validation and quality checks
- 🔍 **Code Quality**: Linting of data processing code
- ⚙️ **Data Processing**: Data preprocessing and feature engineering
- 📊 **Quality Assurance**: Data integrity and range validation
- 📤 **Artifact Management**: Data versioning and storage
- 📢 **Notifications**: Downstream pipeline notifications

**Current Status:** Placeholder implementation with comprehensive logging

### 3. Model Pipeline (`model-pipeline.yml`)

**Triggers when:** Changes in `model/**` directory or training scripts

**Includes:**
- ✅ **Model Validation**: Code validation and architecture checks
- 🏋️ **Training**: Model training with hyperparameter optimization
- 🧪 **Testing**: Comprehensive model unit testing
- 📝 **Registration**: Model artifact registration and versioning
- 🚀 **Deployment Readiness**: Pre-deployment validation
- 📢 **Notifications**: Monitoring and downstream notifications

**Current Status:** Placeholder implementation with detailed workflow steps

### 4. Main Pipeline (`main.yml`)

**Triggers when:** Changes to infrastructure, documentation, or other files

**Includes:**
- ℹ️ **Pipeline Information**: Overview of the pipeline system
- 🏗️ **Infrastructure Checks**: Terraform and Kubernetes manifest validation
- 📚 **Documentation**: Documentation completeness checks
- 🔍 **General Linting**: YAML and root-level Python file validation

## 🔧 Configuration

### Required Secrets

Add these secrets to your GitHub repository:

```bash
# Docker Hub
DOCKER_USERNAME          # Docker Hub username
DOCKER_PASSWORD          # Docker Hub password/token

# AWS Configuration  
AWS_ACCESS_KEY_ID       # AWS access key
AWS_SECRET_ACCESS_KEY   # AWS secret key
AWS_REGION              # AWS region (e.g., ap-south-1)
EKS_CLUSTER_NAME        # EKS cluster name for deployment
```

### Environment Variables

Each pipeline uses consistent environment variables:

```yaml
env:
  PYTHON_VERSION: '3.9'          # Python version for all jobs
  MLFLOW_TRACKING_URI: 'http://localhost:5000'  # MLflow tracking (model pipeline)
```

## 📁 Directory Structure Impact

### Services Directory Changes
```bash
services/
├── api/
│   ├── app/
│   ├── tests/
│   ├── Dockerfile
│   └── requirements.txt
└── predict/
    ├── app/
    ├── tests/
    ├── Dockerfile
    └── requirements.txt
```
**Result:** Full CI/CD pipeline with testing, building, and deployment

### Data Directory Changes
```bash
data/
├── raw/
├── processed/
├── features/
└── src/
    └── preprocess.py
```
**Result:** Data validation, quality checks, and preprocessing pipeline

### Model Directory Changes
```bash
model/
├── src/
├── artifacts/
├── configs/
└── *.py training scripts
```
**Result:** Model training, validation, and registration pipeline

## 🎯 Pipeline Optimization Features

### 1. **Caching Strategy**
- Pip dependencies cached by directory type
- Docker layer caching for faster builds
- Separate cache keys for different pipeline types

### 2. **Parallel Execution**
- Independent jobs run in parallel within each pipeline
- Matrix builds for multi-version testing
- Concurrent security scanning

### 3. **Conditional Execution**
- Deployment only on main branch
- Different validation levels for PR vs push
- Smart path filtering to avoid unnecessary runs

### 4. **Resource Efficiency**
- Path-based triggering reduces compute usage
- Only relevant tests run for specific changes
- Targeted linting and validation

## 🔄 Workflow Examples

### Example 1: API Service Update
```bash
# Change made to services/api/app/main.py
git add services/api/app/main.py
git commit -m "Update API endpoint"
git push
```
**Result:** Only `service-pipeline.yml` pipeline runs

### Example 2: Data Processing Update  
```bash
# Change made to data/src/preprocess.py
git add data/src/preprocess.py
git commit -m "Improve data preprocessing"
git push
```
**Result:** Only `data-pipeline.yml` pipeline runs

### Example 3: Model Training Update
```bash
# Change made to model/housing_models.py
git add model/housing_models.py  
git commit -m "Add new model architecture"
git push
```
**Result:** Only `model-pipeline.yml` pipeline runs

## 🚨 Troubleshooting

### Pipeline Not Triggering
1. Check if file changes match the path filters
2. Verify branch is `main` or `develop` for push triggers
3. Check if files are in ignored paths

### Multiple Pipelines Triggering
- This is expected if changes span multiple directories
- Each pipeline runs independently and in parallel

### Deployment Failures
1. Verify all required secrets are configured
2. Check AWS credentials and EKS cluster access
3. Ensure Docker Hub credentials are valid

## 📈 Future Enhancements

### Planned Features
- [ ] MLflow integration for model tracking
- [ ] DVC integration for data versioning  
- [ ] Advanced monitoring and alerting
- [ ] Cross-pipeline dependencies
- [ ] Automatic rollback capabilities
- [ ] Performance benchmarking

### Integration Opportunities
- [ ] Slack/Teams notifications
- [ ] Jira ticket integration
- [ ] Automated PR comments with pipeline results
- [ ] Integration with monitoring dashboards

## 📊 Monitoring and Observability

Each pipeline includes comprehensive logging and status reporting:

- **Real-time Status**: GitHub Actions provides live status updates
- **Detailed Logs**: Each step includes descriptive output
- **Artifact Tracking**: Build artifacts and test results are preserved
- **Security Reports**: Vulnerability scans uploaded to Security tab
- **Coverage Reports**: Test coverage integrated with Codecov

This architecture provides a robust, scalable foundation for MLOps workflows with clear separation of concerns and efficient resource utilization.
