#!/usr/bin/env python3
"""
Comprehensive training script for California housing price prediction models.
This script runs the complete MLOps pipeline with MLflow tracking.
"""

import sys
import os
from pathlib import Path

# Add project root to Python path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

# Import the training pipeline
from model.src.train_models import MLTrainingPipeline
import logging

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler('training.log')
    ]
)

logger = logging.getLogger(__name__)


def main():
    """Run the comprehensive training pipeline."""
    logger.info("Starting California Housing MLOps Training Pipeline")
    
    # Configuration for the training pipeline
    config = {
        'data_dir': 'data/processed',
        'experiment_name': 'california_housing_comprehensive',
        'models_to_train': {
            'linear_regression': {
                'enabled': True,
                'variants': ['linear', 'ridge', 'lasso', 'elastic'],
                'hyperparameter_tuning': True,
                'search_type': 'grid',
                'n_trials': 30
            },
            'decision_tree': {
                'enabled': True,
                'variants': ['decision_tree', 'random_forest', 'gradient_boosting'],
                'hyperparameter_tuning': True,
                'search_type': 'random',
                'n_trials': 50
            }
        },
        'advanced_tuning': {
            'enabled': False,  # Set to True if you have Optuna installed
            'n_trials': 100,
            'timeout': 1800  # 30 minutes
        },
        'model_selection': {
            'primary_metric': 'test_r2_score',
            'min_r2_threshold': 0.5,  # Minimum acceptable R² score
            'weights': {
                'performance': 0.7,
                'simplicity': 0.2,
                'robustness': 0.1
            }
        },
        'model_registry': {
            'register_best_model': True,
            'model_name': 'california_housing_predictor',
            'promote_to_production': True
        },
        'output': {
            'save_results': True,
            'generate_report': True,
            'create_visualizations': True,
            'results_dir': 'training_results'
        }
    }
    
    try:
        # Check if processed data exists
        data_dir = Path(config['data_dir'])
        required_files = ['X_train.csv', 'X_test.csv', 'y_train.csv', 'y_test.csv']
        
        missing_files = []
        for file_name in required_files:
            if not (data_dir / file_name).exists():
                missing_files.append(file_name)
        
        if missing_files:
            logger.error(f"Missing required data files: {missing_files}")
            logger.error(f"Please ensure processed data is available in {data_dir}")
            logger.info("You can process the data using the preprocessing scripts in data/src/")
            return
        
        # Initialize and run the training pipeline
        pipeline = MLTrainingPipeline(config=config, tracking_uri="databricks")
        
        # Set up Databricks credentials if available
        if "DATABRICKS_HOST" in os.environ and "DATABRICKS_TOKEN" in os.environ:
            logger.info("Databricks credentials found, using Databricks SaaS for MLflow")
        else:
            logger.info("No Databricks credentials found, using local MLflow tracking")
            logger.info("To use Databricks SaaS, set DATABRICKS_HOST and DATABRICKS_TOKEN environment variables")
        
        # Run the complete pipeline
        logger.info("="*60)
        logger.info("STARTING COMPREHENSIVE MODEL TRAINING")
        logger.info("="*60)
        
        pipeline.run_complete_pipeline()
        
        logger.info("="*60)
        logger.info("TRAINING PIPELINE COMPLETED SUCCESSFULLY")
        logger.info("="*60)
        
        # Print summary
        if pipeline.best_model_info:
            logger.info("BEST MODEL SUMMARY:")
            logger.info(f"  Model Type: {pipeline.best_model_info['model_type']}")
            logger.info(f"  R² Score: {pipeline.best_model_info['performance']:.4f}")
            logger.info(f"  Rank: {pipeline.best_model_info['rank']}")
            
            # Print all test metrics if available
            for key, value in pipeline.best_model_info.items():
                if key.startswith('test_') and isinstance(value, (int, float)):
                    logger.info(f"  {key}: {value:.4f}")
        
        logger.info(f"Total models trained: {len(pipeline.training_results)}")
        
        # Print results location
        results_dir = Path(config['output']['results_dir'])
        if results_dir.exists():
            logger.info(f"Results saved to: {results_dir.absolute()}")
            logger.info("Generated files:")
            for file_path in results_dir.rglob('*'):
                if file_path.is_file():
                    logger.info(f"  - {file_path.relative_to(results_dir)}")
        
        # MLflow UI information
        logger.info("\nMLflow Tracking:")
        logger.info("  - View experiments in MLflow UI")
        logger.info("  - Local UI: mlflow ui --backend-store-uri file:./mlruns")
        logger.info("  - Databricks: Check your Databricks workspace MLflow section")
        
    except Exception as e:
        logger.error(f"Training pipeline failed: {e}")
        logger.error("Check the logs above for details")
        raise


if __name__ == "__main__":
    main()
