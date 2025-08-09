#!/usr/bin/env python3
"""
Test script to verify Databricks MLflow connection and experiment creation.
"""

import sys
import os
import logging
from pathlib import Path

# Add project root to Python path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


def test_databricks_connection():
    """Test Databricks MLflow connection."""
    logger.info("Testing Databricks MLflow connection...")
    
    # Check environment variables
    databricks_host = os.environ.get("DATABRICKS_HOST")
    databricks_token = os.environ.get("DATABRICKS_TOKEN")
    
    if not databricks_host:
        logger.error("DATABRICKS_HOST environment variable not set")
        return False
    
    if not databricks_token:
        logger.error("DATABRICKS_TOKEN environment variable not set")
        return False
    
    logger.info(f"Databricks Host: {databricks_host}")
    logger.info("Databricks Token: [HIDDEN]")
    
    try:
        import mlflow
        
        # Set up Databricks MLflow
        mlflow.set_tracking_uri("databricks")
        
        # Test with proper Databricks experiment path
        test_experiment_name = "/Shared/mlops/test_connection"
        
        logger.info(f"Testing experiment creation: {test_experiment_name}")
        
        # Try to get or create experiment
        try:
            experiment = mlflow.get_experiment_by_name(test_experiment_name)
            if experiment is None:
                experiment_id = mlflow.create_experiment(test_experiment_name)
                logger.info(f"✅ Created new experiment: {test_experiment_name}")
            else:
                experiment_id = experiment.experiment_id
                logger.info(f"✅ Found existing experiment: {test_experiment_name}")
            
            # Set the experiment
            mlflow.set_experiment(test_experiment_name)
            
            # Test a simple run
            with mlflow.start_run(run_name="connection_test"):
                mlflow.log_param("test_param", "connection_test")
                mlflow.log_metric("test_metric", 1.0)
                run_id = mlflow.active_run().info.run_id
                logger.info(f"✅ Successfully logged test run: {run_id}")
            
            logger.info("✅ Databricks MLflow connection test PASSED")
            return True
            
        except Exception as e:
            logger.error(f"❌ Error with experiment operations: {e}")
            
            # Try alternative experiment path formats
            alternative_paths = [
                "/Users/mlops/test_connection",
                f"/Users/{os.environ.get('USER', 'user')}/test_connection",
                "/Workspace/Shared/mlops/test_connection"
            ]
            
            for alt_path in alternative_paths:
                try:
                    logger.info(f"Trying alternative path: {alt_path}")
                    experiment = mlflow.get_experiment_by_name(alt_path)
                    if experiment is None:
                        experiment_id = mlflow.create_experiment(alt_path)
                        logger.info(f"✅ Created experiment with alternative path: {alt_path}")
                        return True
                    else:
                        logger.info(f"✅ Found experiment with alternative path: {alt_path}")
                        return True
                except Exception as alt_e:
                    logger.warning(f"Alternative path {alt_path} failed: {alt_e}")
                    continue
            
            return False
            
    except Exception as e:
        logger.error(f"❌ Failed to connect to Databricks MLflow: {e}")
        return False


def suggest_experiment_paths():
    """Suggest proper experiment paths for Databricks."""
    logger.info("\n📝 Suggested Databricks experiment paths:")
    
    user = os.environ.get('USER', 'user')
    paths = [
        f"/Users/{user}/mlops/california_housing",
        "/Shared/mlops/california_housing",
        "/Workspace/Shared/mlops/california_housing",
        f"/Users/{user}/experiments/california_housing"
    ]
    
    for i, path in enumerate(paths, 1):
        logger.info(f"  {i}. {path}")
    
    logger.info("\n💡 Tips:")
    logger.info("  - Use absolute paths starting with '/'")
    logger.info("  - /Shared/ is accessible to all workspace users")
    logger.info("  - /Users/<username>/ is user-specific")
    logger.info("  - Create parent folders in Databricks workspace if needed")


def main():
    """Main test function."""
    logger.info("🧪 Databricks MLflow Connection Test")
    logger.info("=" * 50)
    
    # Test connection
    success = test_databricks_connection()
    
    if not success:
        logger.info("\n" + "=" * 50)
        suggest_experiment_paths()
        logger.info("\n" + "=" * 50)
        logger.info("❌ Connection test failed.")
        logger.info("\n🔧 Next steps:")
        logger.info("1. Verify your Databricks token has proper permissions")
        logger.info("2. Check if the workspace allows experiment creation")
        logger.info("3. Try creating a folder '/Shared/mlops' in your Databricks workspace")
        logger.info("4. Or use a /Users/<your-username>/ path instead")
        return False
    else:
        logger.info("\n" + "=" * 50)
        logger.info("✅ All tests passed! Ready for MLflow training with Databricks.")
        logger.info("\n🚀 You can now run:")
        logger.info("  ./run_training.sh YOUR_TOKEN quick")
        return True


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
