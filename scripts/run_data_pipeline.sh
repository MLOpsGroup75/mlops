#!/bin/bash

# Data Pipeline Manual Execution Script
# This script demonstrates the steps that the GitHub Actions workflow automates

set -e  # Exit on any error

echo "🚀 Starting Data Pipeline..."
echo "================================"

# Check if DVC is available
if ! command -v dvc &> /dev/null; then
    echo "❌ DVC is not installed. Please install DVC first."
    exit 1
fi

echo "✅ DVC is available: $(dvc --version)"

# Check DVC configuration
echo "📋 Checking DVC configuration..."
dvc config --list

# Pull existing data from remote
echo "📥 Pulling existing data from DVC remote..."
dvc pull --all-branches || echo "No existing data to pull"

# Run data processing (if script exists)
if [ -f "data/src/preprocess.py" ]; then
    echo "⚙️ Running data processing..."
    cd data/src
    python preprocess.py
    cd ../..
    echo "✅ Data processing completed!"
else
    echo "⚠️ No preprocessing script found at data/src/preprocess.py"
fi

# Add and version datasets with DVC
echo "📊 Versioning datasets with DVC..."

# Add raw data
if [ -d "data/raw" ]; then
    echo "📁 Processing raw data..."
    cd data/raw
    for file in *.csv; do
        if [ -f "$file" ]; then
            echo "  Adding $file to DVC..."
            dvc add "$file"
        fi
    done
    cd ../..
fi

# Add processed data
if [ -d "data/processed" ]; then
    echo "📁 Processing processed data..."
    cd data/processed
    for file in *.csv; do
        if [ -f "$file" ]; then
            echo "  Adding $file to DVC..."
            dvc add "$file"
        fi
    done
    cd ../..
fi

echo "✅ All datasets versioned with DVC!"

# Check DVC status
echo "📋 Checking DVC status..."
dvc status

# Push data to remote (if configured)
echo "📤 Pushing data to DVC remote..."
dvc push --all-branches
echo "✅ Data pushed to DVC remote!"

# Generate data summary
echo "📋 Generating data summary..."
python scripts/test_data_pipeline.py

# Show git status for .dvc files
echo "🔍 Checking git status for .dvc files..."
git status --porcelain | grep "\.dvc" || echo "No .dvc file changes detected"

echo ""
echo "🎉 Data Pipeline Completed Successfully!"
echo "========================================"
echo "✅ Data processed and versioned"
echo "✅ Data pushed to DVC remote"
echo "✅ Data summary generated"
echo ""
echo "📝 Next steps:"
echo "  1. Review .dvc file changes: git status"
echo "  2. Commit .dvc files: git add data/**/*.dvc"
echo "  3. Push to git: git commit -m 'Update dataset versions' && git push"
echo ""
echo "💡 Tip: The GitHub Actions workflow automates these steps!"
