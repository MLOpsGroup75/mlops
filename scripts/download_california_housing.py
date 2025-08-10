#!/usr/bin/env python3
"""
Script to download California Housing dataset and save it to data/raw folder.
"""

import os
import pandas as pd
import numpy as np
import mlflow.sklearn
from sklearn.datasets import fetch_california_housing
import json
from pathlib import Path


def download_california_housing_dataset():
    """
    Download California Housing dataset and save to data/raw folder.
    """
    # Get the project root directory
    script_dir = Path(__file__).parent
    project_root = script_dir.parent
    raw_data_dir = project_root / "data" / "raw"

    # Create raw data directory if it doesn't exist
    raw_data_dir.mkdir(parents=True, exist_ok=True)

    print("Downloading California Housing dataset...")

    # Fetch the dataset
    housing = fetch_california_housing(as_frame=True)

    # Get features and target
    X = housing.data
    y = housing.target

    # Combine features and target into a single DataFrame
    housing_df = X.copy()
    housing_df["target"] = y

    # Save as CSV
    csv_path = raw_data_dir / "california_housing.csv"
    housing_df.to_csv(csv_path, index=False)
    print(f"✓ Dataset saved as CSV: {csv_path}")

    # Save features separately
    features_path = raw_data_dir / "california_housing_features.csv"
    X.to_csv(features_path, index=False)
    print(f"✓ Features saved: {features_path}")

    # Save target separately
    target_path = raw_data_dir / "california_housing_target.csv"
    y.to_csv(target_path, index=False, header=["MedHouseVal"])
    print(f"✓ Target values saved: {target_path}")

    # Save dataset metadata
    metadata = {
        "dataset_name": "California Housing",
        "description": housing.DESCR,
        "feature_names": list(housing.feature_names),
        "target_name": "MedHouseVal",
        "n_samples": len(housing_df),
        "n_features": len(housing.feature_names),
        "data_types": housing_df.dtypes.astype(str).to_dict(),
        "file_paths": {
            "full_dataset": str(csv_path.relative_to(project_root)),
            "features_only": str(features_path.relative_to(project_root)),
            "target_only": str(target_path.relative_to(project_root)),
        },
    }

    metadata_path = raw_data_dir / "california_housing_metadata.json"
    with open(metadata_path, "w") as f:
        json.dump(metadata, f, indent=2)
    print(f"✓ Metadata saved: {metadata_path}")

    # Print dataset summary
    print("\n" + "=" * 50)
    print("CALIFORNIA HOUSING DATASET SUMMARY")
    print("=" * 50)
    print(f"Dataset shape: {housing_df.shape}")
    print(f"Features: {', '.join(housing.feature_names)}")
    print(
        f"Target: {housing.target_names[0] if hasattr(housing, 'target_names') and housing.target_names else 'MedHouseVal'}"
    )
    print(f"\nFirst few rows:")
    print(housing_df.head())
    print(f"\nDataset statistics:")
    print(housing_df.describe())

    return housing_df, metadata


if __name__ == "__main__":
    try:
        dataset, metadata = download_california_housing_dataset()
        print("\n✅ California Housing dataset successfully downloaded!")
        print(f"📁 Files saved in: data/raw/")
        print(
            f"📊 Dataset contains {metadata['n_samples']} samples with {metadata['n_features']} features"
        )
    except Exception as e:
        print(f"❌ Error downloading dataset: {e}")
        raise
