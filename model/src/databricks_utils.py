"""
Databricks-specific utilities for MLflow integration.
"""

import os
import logging
import mlflow
from typing import Optional, List

logger = logging.getLogger(__name__)


class DatabricksExperimentManager:
    """Manages Databricks experiment creation with proper path handling."""

    def __init__(self):
        self.is_databricks = self._is_databricks_environment()

    def _is_databricks_environment(self) -> bool:
        """Check if we're running in a Databricks environment."""
        return (
            "DATABRICKS_HOST" in os.environ
            and "DATABRICKS_TOKEN" in os.environ
            and mlflow.get_tracking_uri() == "databricks"
        )

    def get_experiment_path_candidates(self, base_name: str) -> List[str]:
        """
        Get list of potential experiment paths to try for Databricks.

        Args:
            base_name: Base experiment name (e.g., 'california_housing_linear')

        Returns:
            List of experiment paths to try, in order of preference
        """
        if not self.is_databricks:
            return [base_name]  # Use simple name for non-Databricks

        # Get username for user-specific paths
        user = os.environ.get("DATABRICKS_USER", os.environ.get("USERNAME", "user"))

        # Try different path formats that commonly work in Databricks
        candidates = [
            f"/Shared/mlops/{base_name}",
            f"/Users/{user}/mlops/{base_name}",
            f"/{base_name}",  # Root level (sometimes works)
        ]

        return candidates

    def create_experiment_with_fallback(self, base_name: str) -> Optional[str]:
        """
        Create experiment with automatic fallback to alternative paths.

        Args:
            base_name: Base experiment name

        Returns:
            Actual experiment name that was created/found, or None if all failed
        """
        candidates = self.get_experiment_path_candidates(base_name)

        for experiment_path in candidates:
            try:
                # Check if experiment already exists
                experiment = mlflow.get_experiment_by_name(experiment_path)

                if experiment is not None:
                    logger.info(f"✅ Found existing experiment: {experiment_path}")
                    mlflow.set_experiment(experiment_path)
                    return experiment_path

                # Try to create new experiment
                experiment_id = mlflow.create_experiment(experiment_path)
                logger.info(f"✅ Created new experiment: {experiment_path}")
                mlflow.set_experiment(experiment_path)
                return experiment_path

            except Exception as e:
                logger.warning(
                    f"❌ Failed to use experiment path '{experiment_path}': {e}"
                )
                continue

        # If all candidates failed, log detailed error
        logger.error("❌ All experiment path candidates failed!")
        logger.error("💡 Suggestions:")
        logger.error("  1. Create '/Shared/mlops' folder in your Databricks workspace")
        logger.error("  2. Ensure your token has experiment creation permissions")
        logger.error("  3. Try using a /Users/<your-username>/ path")

        return None

    def setup_experiment(self, base_name: str) -> bool:
        """
        Set up experiment with proper error handling and fallbacks.

        Args:
            base_name: Base experiment name

        Returns:
            True if experiment was set up successfully
        """
        try:
            experiment_name = self.create_experiment_with_fallback(base_name)

            if experiment_name is None:
                logger.error(f"Failed to set up any experiment for '{base_name}'")
                return False

            # Verify the experiment is properly set
            current_experiment = mlflow.get_experiment_by_name(experiment_name)
            if current_experiment is None:
                logger.error(f"Failed to verify experiment '{experiment_name}'")
                return False

            logger.info(f"✅ Experiment setup completed: {experiment_name}")
            return True

        except Exception as e:
            logger.error(f"❌ Critical error in experiment setup: {e}")
            return False


def setup_databricks_mlflow(experiment_name: str) -> Optional[str]:
    """
    Convenience function to set up Databricks MLflow with proper experiment handling.

    Args:
        experiment_name: Base experiment name

    Returns:
        Actual experiment name used, or None if setup failed
    """
    try:
        # Set tracking URI
        mlflow.set_tracking_uri("databricks")

        # Use experiment manager for proper path handling
        exp_manager = DatabricksExperimentManager()

        if exp_manager.setup_experiment(experiment_name):
            # Get the current experiment name
            current_exp = mlflow.get_experiment(
                mlflow.get_experiment_by_name(experiment_name).experiment_id
            )
            return current_exp.name
        else:
            return None

    except Exception as e:
        logger.error(f"Failed to setup Databricks MLflow: {e}")
        return None


def test_databricks_permissions() -> bool:
    """
    Test if the current Databricks setup has proper permissions.

    Returns:
        True if permissions are sufficient
    """
    try:
        # Try to create a test experiment
        test_name = "mlops_permission_test"
        exp_manager = DatabricksExperimentManager()

        test_experiment = exp_manager.create_experiment_with_fallback(test_name)

        if test_experiment is None:
            return False

        # Try to log a test run
        with mlflow.start_run(run_name="permission_test"):
            mlflow.log_param("test", "permissions")
            mlflow.log_metric("test_metric", 1.0)

        logger.info("✅ Databricks permissions test passed")
        return True

    except Exception as e:
        logger.error(f"❌ Databricks permissions test failed: {e}")
        return False
