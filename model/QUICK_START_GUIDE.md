# 🚀 Quick Start Guide - California Housing MLOps Training

## Option 1: With Databricks Token (Recommended)
```bash
# Replace YOUR_TOKEN_HERE with your actual token
./model/scripts/run_training.sh YOUR_TOKEN_HERE
```

## Option 2: Using Python Script Directly
```bash
# Set environment variables
export DATABRICKS_HOST="https://dbc-87ef0a7b-f01d.cloud.databricks.com"
export DATABRICKS_TOKEN="YOUR_TOKEN_HERE"

# Run training
python model/scripts/train_with_databricks.py
```

## Option 3: Local MLflow Only
```bash
# Run without Databricks (uses local MLflow)
./model/scripts/run_training.sh local
```

## Generated Files:
- **`training_results/`** - Complete training results and reports
- **`plots/`** - Performance visualizations 
- **`model/artifacts/`** - Saved model files
- **`mlruns/`** - MLflow experiment tracking data

## MLflow Artifacts:
- Experiment tracking with all parameters and metrics
- Model artifacts with versioning
- Performance plots and comparisons
- Best model registration in Model Registry

## Viewing Results

### 1. MLflow UI (Local)
```bash
mlflow ui --backend-store-uri file:./mlruns
# Open: http://localhost:5000
```

### 2. Databricks MLflow
- Go to your Databricks workspace
- Navigate to "Machine Learning" -> "Experiments"
- Look for experiments starting with "california_housing"

### 3. Generated Reports
- **HTML Report:** `model_comparison_report.html`
- **CSV Results:** `model_comparison.csv`
- **Best Model Info:** `best_model_info.json`

## Timing Expectations
- **Training:** ~15-20 minutes (7 models)