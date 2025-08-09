# Data Preprocessing

This module contains a simple script for preprocessing the California Housing dataset.

## Files

- `preprocess.py` - Simple preprocessing script that handles basic data preparation

## What it does

1. **Loads data** from CSV files in the `raw` folder
2. **Splits data** into train/test sets (80/20 split)
3. **Scales features** using StandardScaler
4. **Saves processed data** to the `processed` folder
5. **Saves feature info** to the `features` folder

## Usage

```bash
cd data/src
python preprocess.py
```

## Output Structure

After running the script:

```
data/
├── processed/
│   ├── X_train.csv     # Training features (16,512 samples)
│   ├── X_test.csv      # Testing features (4,128 samples)  
│   ├── y_train.csv     # Training targets
│   └── y_test.csv      # Testing targets
├── features/
│   └── feature_info.json  # Feature names and count
└── src/
    ├── preprocess.py
    └── README.md
```

Simple and straightforward - just what you need for basic data preprocessing!