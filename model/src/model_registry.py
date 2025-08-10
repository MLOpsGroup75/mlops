"""
MLflow model registry and management utilities.
"""

import mlflow
import mlflow.sklearn
import pandas as pd
import numpy as np
from typing import Dict, Any, List, Optional, Tuple
from mlflow.tracking import MlflowClient
from mlflow.entities import ViewType
import logging
import cloudpickle
import joblib
import os
from pathlib import Path

logger = logging.getLogger(__name__)


class ModelRegistry:
    """MLflow model registry for managing trained models."""

    def __init__(
        self, tracking_uri: str = "databricks", registry_uri: Optional[str] = None
    ):
        """
        Initialize model registry.

        Args:
            tracking_uri: MLflow tracking URI
            registry_uri: MLflow model registry URI (optional, defaults to tracking_uri)
        """
        self.tracking_uri = tracking_uri
        self.registry_uri = registry_uri or tracking_uri

        # Set up MLflow
        self._setup_mlflow()

        # Initialize client
        self.client = MlflowClient(
            tracking_uri=self.tracking_uri, registry_uri=self.registry_uri
        )

    def _setup_mlflow(self):
        """Set up MLflow tracking and registry."""
        try:
            # For Databricks SaaS, set tracking URI to databricks
            if self.tracking_uri == "databricks":
                if "DATABRICKS_HOST" in os.environ and "DATABRICKS_TOKEN" in os.environ:
                    mlflow.set_tracking_uri("databricks")
                    if self.registry_uri == "databricks":
                        mlflow.set_registry_uri("databricks")
                    logger.info(
                        "Using Databricks SaaS for MLflow tracking and registry"
                    )
                else:
                    # Fallback to local tracking
                    mlflow.set_tracking_uri("file:./mlruns")
                    logger.warning(
                        "Databricks credentials not found, using local MLflow tracking"
                    )
            else:
                mlflow.set_tracking_uri(self.tracking_uri)
                if self.registry_uri:
                    mlflow.set_registry_uri(self.registry_uri)

        except Exception as e:
            logger.error(f"Error setting up MLflow: {e}")
            # Set local tracking as ultimate fallback
            mlflow.set_tracking_uri("file:./mlruns")

    def get_best_models_from_experiments(
        self,
        experiment_names: List[str],
        metric_name: str = "test_r2_score",
        ascending: bool = False,
        max_results: int = 10,
    ) -> List[Dict[str, Any]]:
        """
        Get best models from multiple experiments.

        Args:
            experiment_names: List of experiment names to search
            metric_name: Metric to optimize for
            ascending: Whether to sort in ascending order (True for metrics like RMSE)
            max_results: Maximum number of results to return

        Returns:
            List of best model information dictionaries
        """
        best_models = []

        for experiment_name in experiment_names:
            try:
                experiment = self.client.get_experiment_by_name(experiment_name)
                if experiment is None:
                    logger.warning(f"Experiment '{experiment_name}' not found")
                    continue

                # Search runs in the experiment
                runs = self.client.search_runs(
                    experiment_ids=[experiment.experiment_id],
                    filter_string="",
                    run_view_type=ViewType.ACTIVE_ONLY,
                    max_results=1000,
                    order_by=[
                        f"metrics.{metric_name} {'ASC' if ascending else 'DESC'}"
                    ],
                )

                for run in runs:
                    if metric_name in run.data.metrics:
                        model_info = {
                            "run_id": run.info.run_id,
                            "experiment_name": experiment_name,
                            "experiment_id": experiment.experiment_id,
                            "model_uri": f"runs:/{run.info.run_id}/model",
                            "metric_value": run.data.metrics[metric_name],
                            "run_name": run.data.tags.get("mlflow.runName", "Unknown"),
                            "model_type": run.data.tags.get("model_type", "Unknown"),
                            "start_time": run.info.start_time,
                            "params": run.data.params,
                            "metrics": run.data.metrics,
                            "tags": run.data.tags,
                        }
                        best_models.append(model_info)

            except Exception as e:
                logger.error(f"Error searching experiment '{experiment_name}': {e}")

        # Sort all models by the metric
        best_models.sort(key=lambda x: x["metric_value"], reverse=not ascending)

        return best_models[:max_results]

    def compare_models(
        self, model_infos: List[Dict[str, Any]], comparison_metrics: List[str] = None
    ) -> pd.DataFrame:
        """
        Compare multiple models using specified metrics.

        Args:
            model_infos: List of model information dictionaries
            comparison_metrics: List of metrics to compare

        Returns:
            DataFrame with model comparison results
        """
        if comparison_metrics is None:
            comparison_metrics = [
                "test_r2_score",
                "test_rmse",
                "test_mae",
                "cv_r2_mean",
            ]

        comparison_data = []

        for model_info in model_infos:
            row = {
                "run_id": model_info["run_id"],
                "model_type": model_info.get("model_type", "Unknown"),
                "run_name": model_info.get("run_name", "Unknown"),
                "experiment_name": model_info.get("experiment_name", "Unknown"),
            }

            # Add metrics
            for metric in comparison_metrics:
                row[metric] = model_info["metrics"].get(metric, None)

            # Add key parameters
            params = model_info.get("params", {})
            for param_key in [
                "model_type",
                "hyperparameter_optimization",
                "search_type",
            ]:
                if param_key in params:
                    row[f"param_{param_key}"] = params[param_key]

            comparison_data.append(row)

        comparison_df = pd.DataFrame(comparison_data)

        # Sort by the first metric in the list
        if comparison_metrics and comparison_metrics[0] in comparison_df.columns:
            ascending = (
                "rmse" in comparison_metrics[0].lower()
                or "mae" in comparison_metrics[0].lower()
            )
            comparison_df = comparison_df.sort_values(
                comparison_metrics[0], ascending=ascending
            )

        return comparison_df

    def register_best_model(
        self,
        model_info: Dict[str, Any],
        model_name: str = "california_housing_predictor",
        description: str = None,
        tags: Dict[str, str] = None,
    ) -> str:
        """
        Register the best model in MLflow Model Registry.

        Args:
            model_info: Model information dictionary
            model_name: Name for the registered model
            description: Model description
            tags: Additional tags for the model

        Returns:
            Model version string
        """
        try:
            # Create registered model if it doesn't exist
            try:
                self.client.get_registered_model(model_name)
                logger.info(f"Using existing registered model: {model_name}")
            except Exception:
                self.client.create_registered_model(
                    name=model_name,
                    description=description
                    or f"California housing price prediction model",
                )
                logger.info(f"Created new registered model: {model_name}")

            # Create model version
            model_uri = model_info["model_uri"]
            model_version = mlflow.register_model(model_uri=model_uri, name=model_name)

            # Add tags if provided
            if tags:
                for key, value in tags.items():
                    self.client.set_model_version_tag(
                        name=model_name,
                        version=model_version.version,
                        key=key,
                        value=value,
                    )

            # Add performance metrics as tags
            metrics = model_info.get("metrics", {})
            for metric_name, metric_value in metrics.items():
                if isinstance(metric_value, (int, float)):
                    self.client.set_model_version_tag(
                        name=model_name,
                        version=model_version.version,
                        key=f"metric_{metric_name}",
                        value=str(metric_value),
                    )

            logger.info(
                f"Registered model version {model_version.version} for {model_name}"
            )
            return model_version.version

        except Exception as e:
            logger.error(f"Error registering model: {e}")
            raise

    def promote_model_to_production(
        self, model_name: str, version: str, archive_existing: bool = True
    ) -> bool:
        """
        Promote a model version to production stage.

        Args:
            model_name: Name of the registered model
            version: Version to promote
            archive_existing: Whether to archive existing production models

        Returns:
            True if successful
        """
        try:
            # Archive existing production models if requested
            if archive_existing:
                production_versions = self.client.get_latest_versions(
                    name=model_name, stages=["Production"]
                )

                for prod_version in production_versions:
                    self.client.transition_model_version_stage(
                        name=model_name, version=prod_version.version, stage="Archived"
                    )
                    logger.info(f"Archived model version {prod_version.version}")

            # Promote new version to production
            self.client.transition_model_version_stage(
                name=model_name, version=version, stage="Production"
            )

            logger.info(f"Promoted model version {version} to Production")
            return True

        except Exception as e:
            logger.error(f"Error promoting model to production: {e}")
            return False

    def load_model_from_registry(self, model_name: str, stage: str = "Production"):
        """
        Load a model from the registry.

        Args:
            model_name: Name of the registered model
            stage: Model stage (Production, Staging, etc.)

        Returns:
            Loaded model
        """
        try:
            model_uri = f"models:/{model_name}/{stage}"
            model = mlflow.sklearn.load_model(model_uri)
            logger.info(f"Loaded model {model_name} from {stage} stage")
            return model

        except Exception as e:
            logger.error(f"Error loading model from registry: {e}")
            raise

    def get_model_performance_summary(self, model_name: str) -> pd.DataFrame:
        """
        Get performance summary for all versions of a registered model.

        Args:
            model_name: Name of the registered model

        Returns:
            DataFrame with performance summary
        """
        try:
            # Get all versions of the model
            model_versions = self.client.search_model_versions(f"name='{model_name}'")

            summary_data = []

            for version in model_versions:
                version_info = {
                    "version": version.version,
                    "stage": version.current_stage,
                    "creation_timestamp": version.creation_timestamp,
                    "run_id": version.run_id,
                    "status": version.status,
                }

                # Get metrics from tags
                for tag_key, tag_value in version.tags.items():
                    if tag_key.startswith("metric_"):
                        metric_name = tag_key.replace("metric_", "")
                        try:
                            version_info[metric_name] = float(tag_value)
                        except:
                            version_info[metric_name] = tag_value

                summary_data.append(version_info)

            summary_df = pd.DataFrame(summary_data)

            # Sort by version number
            if not summary_df.empty:
                summary_df["version_num"] = summary_df["version"].astype(int)
                summary_df = summary_df.sort_values("version_num", ascending=False)
                summary_df = summary_df.drop("version_num", axis=1)

            return summary_df

        except Exception as e:
            logger.error(f"Error getting model performance summary: {e}")
            return pd.DataFrame()

    def cleanup_old_models(
        self, model_name: str, keep_versions: int = 5, exclude_stages: List[str] = None
    ) -> int:
        """
        Clean up old model versions, keeping only the most recent ones.

        Args:
            model_name: Name of the registered model
            keep_versions: Number of versions to keep
            exclude_stages: Stages to exclude from cleanup

        Returns:
            Number of versions deleted
        """
        if exclude_stages is None:
            exclude_stages = ["Production", "Staging"]

        try:
            # Get all versions
            model_versions = self.client.search_model_versions(f"name='{model_name}'")

            # Filter out excluded stages
            deletable_versions = [
                v for v in model_versions if v.current_stage not in exclude_stages
            ]

            # Sort by creation time (newest first)
            deletable_versions.sort(key=lambda x: x.creation_timestamp, reverse=True)

            # Delete old versions
            deleted_count = 0
            for version in deletable_versions[keep_versions:]:
                try:
                    self.client.delete_model_version(
                        name=model_name, version=version.version
                    )
                    deleted_count += 1
                    logger.info(f"Deleted model version {version.version}")
                except Exception as e:
                    logger.warning(f"Could not delete version {version.version}: {e}")

            logger.info(f"Cleaned up {deleted_count} old model versions")
            return deleted_count

        except Exception as e:
            logger.error(f"Error during model cleanup: {e}")
            return 0
