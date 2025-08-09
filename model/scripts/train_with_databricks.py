#!/usr/bin/env python3
"""
Training script with improved Databricks integration and error handling.
"""

import sys
import os
import argparse
import getpass
from pathlib import Path

# Add project root to Python path
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

from model.src.train_models import MLTrainingPipeline
import logging

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler('training.log')
    ]
)

logger = logging.getLogger(__name__)


def setup_databricks_credentials(host=None, token=None, interactive=True):
    """
    Set up Databricks credentials.
    
    Args:
        host: Databricks host URL
        token: Databricks token
        interactive: Whether to prompt for missing credentials
    
    Returns:
        bool: True if credentials are set up
    """
    # Check if already set in environment
    env_host = os.environ.get("DATABRICKS_HOST")
    env_token = os.environ.get("DATABRICKS_TOKEN")
    
    if env_host and env_token:
        logger.info(f"Using existing Databricks credentials")
        logger.info(f"Host: {env_host}")
        return True
    
    # Use provided arguments
    if host:
        os.environ["DATABRICKS_HOST"] = host
        logger.info(f"Set Databricks host: {host}")
    
    if token:
        os.environ["DATABRICKS_TOKEN"] = token
        logger.info("Set Databricks token: [HIDDEN]")
    
    # Check if we have what we need
    final_host = os.environ.get("DATABRICKS_HOST")
    final_token = os.environ.get("DATABRICKS_TOKEN")
    
    if not final_host and interactive:
        final_host = input("Enter Databricks host URL (or press Enter to use local MLflow): ").strip()
        if final_host:
            os.environ["DATABRICKS_HOST"] = final_host
    
    if final_host and not final_token and interactive:
        final_token = getpass.getpass("Enter Databricks token (hidden input): ").strip()
        if final_token:
            os.environ["DATABRICKS_TOKEN"] = final_token
    
    # Final check
    if os.environ.get("DATABRICKS_HOST") and os.environ.get("DATABRICKS_TOKEN"):
        logger.info("✓ Databricks credentials configured")
        return True
    else:
        logger.info("ℹ Using local MLflow tracking (no Databricks credentials)")
        return False


def run_training_pipeline(use_databricks=True, quick_mode=False):
    """
    Run the training pipeline with proper error handling.
    
    Args:
        use_databricks: Whether to attempt Databricks integration
        quick_mode: Whether to run in quick mode with fewer models
    """
    
    # Configuration
    if quick_mode:
        config = {
            'data_dir': '../../data/processed',
            'experiment_name': 'california_housing_quick',
            'models_to_train': {
                'linear_regression': {
                    'enabled': True,
                    'variants': ['ridge', 'lasso'],  # Only 2 models
                    'hyperparameter_tuning': True,
                    'search_type': 'grid',
                    'n_trials': 10
                },
                'decision_tree': {
                    'enabled': True,
                    'variants': ['decision_tree'],  # Only 1 model
                    'hyperparameter_tuning': True,
                    'search_type': 'random',
                    'n_trials': 20
                }
            },
            'advanced_tuning': {'enabled': False},
            'model_selection': {
                'primary_metric': 'test_r2_score',
                'min_r2_threshold': 0.4,
                'weights': {'performance': 0.8, 'simplicity': 0.2, 'robustness': 0.0}
            },
            'model_registry': {
                'register_best_model': True,  # Disabled in quick mode
                'model_name': 'california_housing_quick',
                'promote_to_production': False,
                'unity_catalog': {
                    'enabled': True,  # Enable Unity Catalog support
                    'catalog': 'workspace',
                    'schema': 'default'
                }
            },
            'output': {
                'save_results': True,
                'generate_report': True,
                'create_visualizations': True,
                'results_dir': '../../quick_results'
            }
        }
    else:
        config = {
            'data_dir': '../../data/processed',
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
            'advanced_tuning': {'enabled': False},
            'model_selection': {
                'primary_metric': 'test_r2_score',
                'min_r2_threshold': 0.5,
                'weights': {'performance': 0.7, 'simplicity': 0.2, 'robustness': 0.1}
            },
            'model_registry': {
                'register_best_model': True,
                'model_name': 'california_housing_predictor',
                'promote_to_production': False,
                'unity_catalog': {
                    'enabled': True,  # Enable Unity Catalog support
                    'catalog': 'workspace',
                    'schema': 'default'
                },
            },
            'output': {
                'save_results': True,
                'generate_report': True,
                'create_visualizations': True,
                'results_dir': '../../training_results'
            }
        }
    
    try:
        # Check data availability
        data_dir = Path(config['data_dir'])
        required_files = ['X_train.csv', 'X_test.csv', 'y_train.csv', 'y_test.csv']
        
        missing_files = []
        for file_name in required_files:
            if not (data_dir / file_name).exists():
                missing_files.append(file_name)
        
        if missing_files:
            logger.error(f"Missing required data files: {missing_files}")
            logger.error(f"Please ensure processed data is available in {data_dir}")
            return False
        
        # Determine tracking URI
        tracking_uri = "databricks" if use_databricks and os.environ.get("DATABRICKS_TOKEN") else "file:./mlruns"
        
        logger.info("="*60)
        logger.info("STARTING CALIFORNIA HOUSING MODEL TRAINING")
        logger.info("="*60)
        logger.info(f"Mode: {'Quick' if quick_mode else 'Comprehensive'}")
        logger.info(f"MLflow tracking: {tracking_uri}")
        logger.info("="*60)
        
        # Initialize and run pipeline
        pipeline = MLTrainingPipeline(config=config, tracking_uri=tracking_uri)
        pipeline.run_complete_pipeline()
        
        logger.info("="*60)
        logger.info("TRAINING COMPLETED SUCCESSFULLY!")
        logger.info("="*60)
        
        # Print summary
        if pipeline.best_model_info:
            logger.info("BEST MODEL SUMMARY:")
            logger.info(f"  Type: {pipeline.best_model_info['model_type']}")
            logger.info(f"  R² Score: {pipeline.best_model_info['performance']:.4f}")
            if 'rank' in pipeline.best_model_info:
                logger.info(f"  Rank: {pipeline.best_model_info['rank']}")
            
            # Show additional metrics if available
            for key, value in pipeline.best_model_info.items():
                if key.startswith('test_') and isinstance(value, (int, float)) and key != 'test_r2_score':
                    logger.info(f"  {key}: {value:.4f}")
        
        logger.info(f"Total models trained: {len(pipeline.training_results)}")
        
        # Results location
        results_dir = Path(config['output']['results_dir'])
        if results_dir.exists():
            logger.info(f"Results saved to: {results_dir.absolute()}")
        
        # MLflow UI info
        logger.info("\nView results:")
        if tracking_uri.startswith("file:"):
            logger.info("  Local MLflow: mlflow ui --backend-store-uri file:./mlruns")
        else:
            logger.info("  Databricks MLflow: Check your Databricks workspace")
        
        return True
        
    except Exception as e:
        logger.error(f"Training failed: {e}")
        logger.error("Check the logs above for details")
        import traceback
        traceback.print_exc()
        return False


def main():
    """Main function with command line interface."""
    parser = argparse.ArgumentParser(description='Train California housing models with MLflow')
    
    parser.add_argument('--databricks-host', type=str,
                       help='Databricks workspace URL')
    parser.add_argument('--databricks-token', type=str,
                       help='Databricks access token')
    parser.add_argument('--local-only', action='store_true',
                       help='Use only local MLflow tracking (skip Databricks)')
    parser.add_argument('--quick', action='store_true',
                       help='Quick mode with fewer models for testing')
    parser.add_argument('--non-interactive', action='store_true',
                       help='Non-interactive mode (no prompts)')
    
    args = parser.parse_args()
    
    # Set up credentials
    use_databricks = not args.local_only
    
    if use_databricks:
        setup_databricks_credentials(
            host=args.databricks_host,
            token=args.databricks_token,
            interactive=not args.non_interactive
        )
    
    # Run training
    success = run_training_pipeline(
        use_databricks=use_databricks,
        quick_mode=args.quick
    )
    
    sys.exit(0 if success else 1)


if __name__ == "__main__":
    main()
