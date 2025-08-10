# Data Pipeline

This directory contains the data pipeline for the MLOps project, including raw data, processed data, and data processing scripts.

## Structure

```
data/
├── raw/                    # Raw datasets (versioned with DVC)
│   ├── california_housing.csv
├── processed/              # Processed datasets (versioned with DVC)
│   ├── X_train.csv
│   ├── y_train.csv
│   ├── X_test.csv
│   └── y_test.csv
├── features/               # Feature information
│   └── feature_info.json
└── src/                    # Data processing scripts
    └── preprocess.py
```

## Data Versioning with DVC

All datasets in the `raw/` and `processed/` folders are versioned using DVC (Data Version Control). The data pipeline automatically:

1. **Pulls existing data** from the DVC remote (S3)
2. **Runs data processing** using scripts in `src/`
3. **Versions new/updated datasets** with DVC
4. **Pushes data** to the DVC remote (S3)
5. **Commits .dvc files** to git

## Automated Data Pipeline

The data pipeline is triggered automatically when:
- Changes are made to files in the `data/` directory
- Changes are made to `scripts/download_california_housing.py`
- Changes are made to the workflow file itself
- Manual triggering via GitHub Actions

## DVC Configuration

- **Remote Storage**: S3 bucket `s3://mlops-housing-dev-datasets`
- **Region**: us-east-1
- **Authentication**: AWS credentials via GitHub Secrets

## Manual Data Operations

### Add a new dataset
```bash
# Add raw data
cd data/raw
dvc add new_dataset.csv

# Add processed data
cd data/processed
dvc add processed_dataset.csv

# Commit .dvc files
git add *.dvc
git commit -m "Add new dataset"
git push
```

### Update existing dataset
```bash
# After modifying a dataset
dvc add dataset.csv  # This updates the .dvc file
git add dataset.csv.dvc
git commit -m "Update dataset"
git push
```

### Pull latest data
```bash
dvc pull  # Pulls all datasets
# or
dvc pull dataset.csv  # Pulls specific dataset
```

### Check data status
```bash
dvc status  # Shows which datasets are out of sync
dvc list .  # Lists all tracked datasets
```

## Data Processing

The data processing pipeline is defined in `src/preprocess.py` and includes:
- Data loading and validation
- Feature engineering
- Train/test splitting
- Data cleaning and preprocessing

## Monitoring

The pipeline generates a data summary artifact (`artifacts/data_summary.json`) containing:
- Dataset statistics (rows, columns, file size)
- Processing status
- Error information (if any)

## Dependencies

- Python 3.9+
- DVC with S3 support
- pandas, scikit-learn, numpy
- AWS credentials configured

## Troubleshooting

### DVC authentication issues
- Ensure AWS credentials are properly configured in GitHub Secrets
- Check S3 bucket permissions
- Verify DVC remote configuration

### Data sync issues
- Run `dvc status` to check sync status
- Use `dvc pull --all-branches` to pull all data
- Check `.dvc` files are committed to git

### Pipeline failures
- Check GitHub Actions logs for detailed error messages
- Verify all required secrets are configured
- Ensure data processing scripts are working correctly