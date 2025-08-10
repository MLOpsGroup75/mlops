#!/usr/bin/env python3
"""
Test script for data pipeline functionality
"""

import os
import json
import pandas as pd
from pathlib import Path

def test_data_summary_generation():
    """Test the data summary generation functionality"""

    # Create artifacts directory if it doesn't exist
    artifacts_dir = Path("artifacts")
    artifacts_dir.mkdir(exist_ok=True)

    # Generate data summary
    summary = {}
    data_dir = Path("data")

    if data_dir.exists():
        for root, dirs, files in os.walk(data_dir):
            for file in files:
                if file.endswith('.csv'):
                    filepath = os.path.join(root, file)
                    try:
                        df = pd.read_csv(filepath)
                        summary[filepath] = {
                            'rows': len(df),
                            'columns': len(df.columns),
                            'size_mb': round(os.path.getsize(filepath) / (1024*1024), 2)
                        }
                        print(f"Processed: {filepath}")
                    except Exception as e:
                        summary[filepath] = {'error': str(e)}
                        print(f"❌ Error processing {filepath}: {e}")

    # Save summary to artifacts
    summary_file = artifacts_dir / "data_summary.json"
    with open(summary_file, 'w') as f:
        json.dump(summary, f, indent=2)

    print(f"\nData Summary Generated:")
    print(f"Saved to: {summary_file}")
    print(f"Total datasets: {len(summary)}")

    for filepath, info in summary.items():
        if 'error' not in info:
            print(f"  • {filepath}: {info['rows']} rows, {info['columns']} columns, {info['size_mb']} MB")
        else:
            print(f"  • {filepath}: ERROR - {info['error']}")

    return summary

def test_dvc_commands():
    """Test DVC command availability"""
    import subprocess

    try:
        # Check if DVC is available
        result = subprocess.run(['dvc', '--version'], capture_output=True, text=True)
        if result.returncode == 0:
            print(f"DVC is available: {result.stdout.strip()}")
        else:
            print("❌ DVC command failed")
    except FileNotFoundError:
        print("❌ DVC not found in PATH")

    try:
        # Check DVC config
        result = subprocess.run(['dvc', 'config', '--list'], capture_output=True, text=True)
        if result.returncode == 0:
            print("DVC config accessible")
            print(result.stdout)
        else:
            print("❌ DVC config command failed")
    except Exception as e:
        print(f"❌ Error checking DVC config: {e}")

if __name__ == "__main__":
    print("🧪 Testing Data Pipeline Functionality")
    print("=" * 50)

    print("\n1. Testing data summary generation...")
    summary = test_data_summary_generation()

    print("\n2. Testing DVC availability...")
    test_dvc_commands()

    print("\nData pipeline tests completed!")
