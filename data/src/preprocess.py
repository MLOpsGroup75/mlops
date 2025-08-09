#!/usr/bin/env python3
"""
Simple Data Preprocessing Script for California Housing Dataset

This script performs basic data preprocessing:
- Load data from raw CSV files
- Split into train/test sets
- Scale features
- Save to processed folder
"""

import os
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import json


def load_data(raw_data_path):
    """Load features and target data from CSV files"""
    print("Loading data...")

    features_path = os.path.join(raw_data_path, "california_housing_features.csv")
    target_path = os.path.join(raw_data_path, "california_housing_target.csv")

    # Load data
    features_df = pd.read_csv(features_path)
    target_df = pd.read_csv(target_path)
    target_series = target_df.iloc[:, 0]

    print(f"Loaded {len(features_df)} samples with {len(features_df.columns)} features")
    return features_df, target_series


def preprocess_data(features_df, target_series):
    """Simple preprocessing: train/test split and scaling"""
    print("Preprocessing data...")

    # Train-test split
    X_train, X_test, y_train, y_test = train_test_split(
        features_df, target_series, test_size=0.2, random_state=42
    )

    # Scale features
    scaler = StandardScaler()
    X_train_scaled = pd.DataFrame(
        scaler.fit_transform(X_train), columns=X_train.columns, index=X_train.index
    )
    X_test_scaled = pd.DataFrame(
        scaler.transform(X_test), columns=X_test.columns, index=X_test.index
    )

    print(f"Train set: {len(X_train_scaled)} samples")
    print(f"Test set: {len(X_test_scaled)} samples")

    return X_train_scaled, X_test_scaled, y_train, y_test


def save_processed_data(X_train, X_test, y_train, y_test, output_path):
    """Save processed data to CSV files"""
    print("Saving processed data...")

    # Create directories
    processed_dir = os.path.join(output_path, "processed")
    features_dir = os.path.join(output_path, "features")
    os.makedirs(processed_dir, exist_ok=True)
    os.makedirs(features_dir, exist_ok=True)

    # Save processed data
    X_train.to_csv(os.path.join(processed_dir, "X_train.csv"), index=False)
    X_test.to_csv(os.path.join(processed_dir, "X_test.csv"), index=False)
    y_train.to_csv(
        os.path.join(processed_dir, "y_train.csv"), index=False, header=["target"]
    )
    y_test.to_csv(
        os.path.join(processed_dir, "y_test.csv"), index=False, header=["target"]
    )

    # Save feature info
    feature_info = {
        "feature_names": X_train.columns.tolist(),
        "n_features": len(X_train.columns),
    }
    with open(os.path.join(features_dir, "feature_info.json"), "w") as f:
        json.dump(feature_info, f, indent=2)

    print(f"Data saved to {processed_dir}")
    print(f"Feature info saved to {features_dir}")


def main():
    """Main preprocessing function"""
    # Set paths relative to script location
    script_dir = os.path.dirname(os.path.abspath(__file__))
    data_dir = os.path.dirname(script_dir)
    raw_data_path = os.path.join(data_dir, "raw")
    output_path = data_dir

    # Load data
    features_df, target_series = load_data(raw_data_path)

    # Preprocess
    X_train, X_test, y_train, y_test = preprocess_data(features_df, target_series)

    # Save results
    save_processed_data(X_train, X_test, y_train, y_test, output_path)

    print("\nPreprocessing completed successfully!")


if __name__ == "__main__":
    main()
