#!/usr/bin/env python3
"""
Test script to verify the environment is properly set up.
"""

import sys
import os
from pathlib import Path


def test_imports():
    """Test that all required packages can be imported."""
    print("Testing package imports...")

    try:
        import pandas as pd

        print(f" pandas {pd.__version__}")
    except ImportError as e:
        print(f" pandas: {e}")
        return False

    try:
        import numpy as np

        print(f" numpy {np.__version__}")
    except ImportError as e:
        print(f" numpy: {e}")
        return False

    try:
        import sklearn

        print(f" scikit-learn {sklearn.__version__}")
    except ImportError as e:
        print(f" scikit-learn: {e}")
        return False

    try:
        import mlflow

        print(f" mlflow {mlflow.__version__}")
    except ImportError as e:
        print(f" mlflow: {e}")
        return False

    try:
        import matplotlib

        print(f" matplotlib {matplotlib.__version__}")
    except ImportError as e:
        print(f" matplotlib: {e}")
        return False

    return True


def test_data_files():
    """Test that required data files exist."""
    print("\nTesting data files...")

    data_dir = Path("data/processed")
    required_files = ["X_train.csv", "X_test.csv", "y_train.csv", "y_test.csv"]

    if not data_dir.exists():
        print(f" Data directory {data_dir} does not exist")
        return False

    all_exist = True
    for file_name in required_files:
        file_path = data_dir / file_name
        if file_path.exists():
            size = file_path.stat().st_size
            print(f" {file_name} ({size:,} bytes)")
        else:
            print(f" {file_name} missing")
            all_exist = False

    return all_exist


def test_mlflow_setup():
    """Test MLflow setup."""
    print("\nTesting MLflow setup...")

    try:
        import mlflow

        # Test local tracking
        mlflow.set_tracking_uri("file:./test_mlruns")
        mlflow.set_experiment("test_experiment")

        with mlflow.start_run():
            mlflow.log_param("test_param", "test_value")
            mlflow.log_metric("test_metric", 0.123)

        print(" Local MLflow tracking works")

        # Test Databricks connection if credentials are provided
        databricks_host = os.environ.get("DATABRICKS_HOST")
        databricks_token = os.environ.get("DATABRICKS_TOKEN")

        if databricks_host and databricks_token:
            print(f" Databricks Host: {databricks_host}")
            print(" Databricks Token: [SET]")

            try:
                mlflow.set_tracking_uri("databricks")
                print(" Databricks MLflow setup successful")
            except Exception as e:
                print(f" Databricks MLflow setup warning: {e}")
        else:
            print(" Databricks credentials not set (will use local MLflow)")
            if not databricks_host:
                print("  Set DATABRICKS_HOST environment variable")
            if not databricks_token:
                print("  Set DATABRICKS_TOKEN environment variable")

        return True

    except Exception as e:
        print(f" MLflow setup failed: {e}")
        return False


def test_model_imports():
    """Test that model training modules can be imported."""
    print("\nTesting model imports...")

    try:
        sys.path.insert(0, str(Path(__file__).parent))

        from model.src.base_trainer import BaseTrainer

        print(" BaseTrainer")

        from model.src.linear_regression_trainer import LinearRegressionTrainer

        print(" LinearRegressionTrainer")

        from model.src.decision_tree_trainer import DecisionTreeTrainer

        print(" DecisionTreeTrainer")

        from model.src.model_comparison import ModelComparator

        print(" ModelComparator")

        from model.src.model_registry import ModelRegistry

        print(" ModelRegistry")

        return True

    except Exception as e:
        print(f" Model import failed: {e}")
        return False


def test_data_loading():
    """Test that data can be loaded properly."""
    print("\nTesting data loading...")

    try:
        sys.path.insert(0, str(Path(__file__).parent))
        from model.src.linear_regression_trainer import LinearRegressionTrainer

        trainer = LinearRegressionTrainer(tracking_uri="file:./test_mlruns")
        X_train, X_test, y_train, y_test = trainer.load_data("data/processed")

        print(f" Training data: {X_train.shape}")
        print(f" Test data: {X_test.shape}")
        print(f" Training targets: {y_train.shape}")
        print(f" Test targets: {y_test.shape}")

        return True

    except Exception as e:
        print(f" Data loading failed: {e}")
        return False


def main():
    """Run all tests."""
    print("Environment Test for MLOps Training Pipeline")
    print("=" * 50)

    tests = [
        test_imports,
        test_data_files,
        test_mlflow_setup,
        test_model_imports,
        test_data_loading,
    ]

    passed = 0
    total = len(tests)

    for test in tests:
        try:
            if test():
                passed += 1
        except Exception as e:
            print(f" Test failed with exception: {e}")

    print("\n" + "=" * 50)
    print(f"Tests passed: {passed}/{total}")

    if passed == total:
        print("All tests passed! Ready to run training pipeline.")
        print("\nNext steps:")
        print("1. Set Databricks credentials (optional):")
        print("   export DATABRICKS_TOKEN='your-token-here'")
        print("2. Run training:")
        print("   python train_housing_models_comprehensive.py")
    else:
        print("Some tests failed. Please fix the issues above.")

    return passed == total


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
