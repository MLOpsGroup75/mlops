#!/bin/bash

# California Housing MLOps Training Script with Databricks Fix
# This script handles Databricks experiment path requirements

echo "California Housing MLOps Training Pipeline (Databricks Fixed)"
echo "=============================================================="

# Set Databricks host
export DATABRICKS_HOST="https://dbc-87ef0a7b-f01d.cloud.databricks.com"

# Check if token is provided as argument
if [ "$1" != "" ]; then
    export DATABRICKS_TOKEN="$1"
    echo "Databricks token set from command line argument"
else
    # Check if token is already in environment
    if [ "$DATABRICKS_TOKEN" = "" ]; then
        echo "Databricks token not provided"
        echo ""
        echo "Usage options:"
        echo "cd model/scripts"
        echo ""
        echo "1. Pass token as argument:     ./run_training.sh YOUR_TOKEN_HERE"
        echo "2. Set environment variable:  export DATABRICKS_TOKEN=YOUR_TOKEN_HERE && ./run_training.sh"
        echo "3. Run without Databricks:    ./run_training.sh local"
        echo ""
        exit 1
    fi
fi

# Check for special modes
if [ "$1" = "local" ]; then
    echo " Running with local MLflow tracking only"
    python train_with_databricks.py --local-only
elif [ "$2" = "test" ] || [ "$1" = "test" ]; then
    echo " Testing Databricks connection only"
    python test_databricks_connection.py
elif [ "$2" = "quick" ] || [ "$1" = "quick" ]; then
    echo "Running in quick mode (fewer models for testing)"
    echo "Using fixed Databricks experiment paths"
    python train_with_databricks.py --quick
else
    echo "Running comprehensive training with Databricks MLflow"
    echo "Using fixed Databricks experiment paths"
    echo " Host: $DATABRICKS_HOST"
    echo " Token: [HIDDEN]"
    echo ""
    echo " Note: Using /Shared/mlops/ experiment paths for Databricks compatibility"
    echo ""
    python train_with_databricks.py
fi

# Check exit status
if [ $? -eq 0 ]; then
    echo ""
    echo "Training completed successfully!"
    echo ""
    echo " Next steps:"
    echo "1. View results in Databricks MLflow: Machine Learning → Experiments"
    echo "2. Look for experiments under '/Shared/mlops/'"
    echo "3. Check generated reports in training_results/ or quick_results/"
else
    echo ""
    echo "Training failed. Common issues:"
    echo ""
    echo " If you see experiment path errors:"
    echo "cd model/scripts"
    echo ""
    echo "1. Run: ./run_training.sh YOUR_TOKEN test"
    echo "2. Create '/Shared/mlops' folder in your Databricks workspace"
    echo "3. Or try: /Users/<your-username>/mlops paths"
    echo ""
    echo " If you see permission errors:"
    echo "1. Verify your token has experiment creation permissions"
    echo "2. Try running locally first: ./run_training.sh local"
    echo ""
fi
