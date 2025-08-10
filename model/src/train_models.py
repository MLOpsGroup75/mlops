"""
Main training script for California housing price prediction models.
"""

import os
import sys
import logging
import argparse
import json
import time
from pathlib import Path
from typing import Dict, Any, List

# Add project root to path
sys.path.append(str(Path(__file__).parent.parent.parent))

# Import model trainers
from model.src.linear_regression_trainer import LinearRegressionTrainer
from model.src.decision_tree_trainer import DecisionTreeTrainer
from model.src.model_comparison import ModelComparator
from model.src.model_registry import ModelRegistry
from model.src.hyperparameter_tuning import HyperparameterTuner

# Import model classes for advanced tuning
import mlflow.sklearn
from sklearn.linear_model import Ridge, Lasso, ElasticNet
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor

# Set up logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


class MLTrainingPipeline:
    """Complete ML training pipeline for California housing prediction."""

    def __init__(self, config: Dict[str, Any] = None, tracking_uri: str = "databricks"):
        """
        Initialize training pipeline.

        Args:
            config: Configuration dictionary
            tracking_uri: MLflow tracking URI
        """
        self.config = config or self._get_default_config()
        self.tracking_uri = tracking_uri
        self.training_results = []
        self.best_model_info = None

        # Initialize components
        self.model_comparator = ModelComparator(tracking_uri)
        self.model_registry = ModelRegistry(tracking_uri)
        self.hyperparameter_tuner = HyperparameterTuner(tracking_uri)

        logger.info("MLTrainingPipeline initialized")

    def _get_default_config(self) -> Dict[str, Any]:
        """Get default configuration."""
        return {
            "data_dir": "data/processed",
            "experiment_name": "california_housing_experiment",
            "models_to_train": {
                "linear_regression": {
                    "enabled": True,
                    "variants": ["linear", "ridge", "lasso", "elastic"],
                    "hyperparameter_tuning": True,
                    "search_type": "grid",
                    "n_trials": 50,
                },
                "decision_tree": {
                    "enabled": True,
                    "variants": ["decision_tree", "random_forest", "gradient_boosting"],
                    "hyperparameter_tuning": True,
                    "search_type": "random",
                    "n_trials": 50,
                },
            },
            "advanced_tuning": {
                "enabled": False,  # Requires Optuna
                "n_trials": 100,
                "timeout": 3600,  # 1 hour
            },
            "model_selection": {
                "primary_metric": "test_r2_score",
                "min_r2_threshold": 0.6,
                "weights": {"performance": 0.7, "simplicity": 0.2, "robustness": 0.1},
            },
            "model_registry": {
                "register_best_model": False,  # Disabled by default due to Unity Catalog requirements
                "model_name": "california_housing_predictor",
                "promote_to_production": False,
                "unity_catalog": {
                    "enabled": False,  # Enable if you have Unity Catalog access
                    "catalog": "main",
                    "schema": "mlops",
                },
            },
            "output": {
                "save_results": True,
                "generate_report": True,
                "create_visualizations": True,
                "results_dir": "results",
            },
        }

    def load_data(self):
        """Load training and test data."""
        logger.info("Loading data...")

        # Use any trainer to load data (they all have the same method)
        dummy_trainer = LinearRegressionTrainer()
        X_train, X_test, y_train, y_test = dummy_trainer.load_data(
            self.config["data_dir"]
        )

        logger.info(
            f"Data loaded: {X_train.shape[0]} train samples, {X_test.shape[0]} test samples"
        )
        return X_train, X_test, y_train, y_test

    def train_linear_regression_models(
        self, X_train, y_train, X_test, y_test
    ) -> List[Dict[str, Any]]:
        """Train linear regression models."""
        logger.info("Training linear regression models...")

        linear_config = self.config["models_to_train"]["linear_regression"]
        results = []

        for variant in linear_config["variants"]:
            logger.info(f"Training {variant} linear regression...")

            start_time = time.time()

            # Initialize trainer
            trainer = LinearRegressionTrainer(
                model_type=variant,
                experiment_name=f"{self.config['experiment_name']}_linear",
                tracking_uri=self.tracking_uri,
            )

            try:
                if linear_config["hyperparameter_tuning"] and variant != "linear":
                    # Train with hyperparameter tuning
                    result = trainer.train_with_hyperparameter_tuning(
                        X_train,
                        y_train,
                        X_test,
                        y_test,
                        search_type=linear_config["search_type"],
                        n_iter=linear_config["n_trials"],
                    )
                else:
                    # Simple training
                    result = trainer.train(X_train, y_train, X_test, y_test)

                # Add timing information
                result["training_time"] = time.time() - start_time
                result["model_variant"] = variant
                results.append(result)

                logger.info(
                    f"{variant} training completed in {result['training_time']:.2f} seconds"
                )

            except Exception as e:
                logger.error(f"Error training {variant}: {e}")
                continue

        return results

    def train_decision_tree_models(
        self, X_train, y_train, X_test, y_test
    ) -> List[Dict[str, Any]]:
        """Train decision tree models."""
        logger.info("Training decision tree models...")

        tree_config = self.config["models_to_train"]["decision_tree"]
        results = []

        for variant in tree_config["variants"]:
            logger.info(f"Training {variant} model...")

            start_time = time.time()

            # Initialize trainer
            trainer = DecisionTreeTrainer(
                model_type=variant,
                experiment_name=f"{self.config['experiment_name']}_tree",
                tracking_uri=self.tracking_uri,
            )

            try:
                if tree_config["hyperparameter_tuning"]:
                    # Train with hyperparameter tuning
                    result = trainer.train_with_hyperparameter_tuning(
                        X_train,
                        y_train,
                        X_test,
                        y_test,
                        search_type=tree_config["search_type"],
                        n_iter=tree_config["n_trials"],
                    )
                else:
                    # Simple training
                    result = trainer.train(X_train, y_train, X_test, y_test)

                # Add timing information
                result["training_time"] = time.time() - start_time
                result["model_variant"] = variant
                results.append(result)

                logger.info(
                    f"{variant} training completed in {result['training_time']:.2f} seconds"
                )

            except Exception as e:
                logger.error(f"Error training {variant}: {e}")
                continue

        return results

    def run_advanced_hyperparameter_tuning(
        self, X_train, y_train
    ) -> List[Dict[str, Any]]:
        """Run advanced hyperparameter tuning with Optuna."""
        if not self.config["advanced_tuning"]["enabled"]:
            logger.info("Advanced tuning disabled")
            return []

        logger.info("Running advanced hyperparameter tuning with Optuna...")

        advanced_results = []

        # Define models to tune
        models_to_tune = [
            (Ridge, "ridge"),
            (Lasso, "lasso"),
            (ElasticNet, "elastic"),
            (DecisionTreeRegressor, "decision_tree"),
            (RandomForestRegressor, "random_forest"),
            (GradientBoostingRegressor, "gradient_boosting"),
        ]

        for model_class, model_type in models_to_tune:
            logger.info(f"Advanced tuning for {model_type}...")

            try:
                result = self.hyperparameter_tuner.tune_model_with_optuna(
                    model_class=model_class,
                    model_type=model_type,
                    X=X_train,
                    y=y_train,
                    n_trials=self.config["advanced_tuning"]["n_trials"],
                    timeout=self.config["advanced_tuning"]["timeout"],
                )

                if result:
                    result["advanced_tuning"] = True
                    advanced_results.append(result)
                    logger.info(f"Advanced tuning for {model_type} completed")

            except Exception as e:
                logger.error(f"Error in advanced tuning for {model_type}: {e}")
                continue

        return advanced_results

    def compare_and_select_best_model(self) -> Dict[str, Any]:
        """Compare all trained models and select the best one."""
        logger.info("Comparing models and selecting the best...")

        if not self.training_results:
            logger.error("No training results available for comparison")
            return {}

        # Compare models
        best_model_info = self.model_comparator.select_best_model(
            self.training_results,
            selection_criteria={
                "primary_metric": self.config["model_selection"]["primary_metric"],
                "min_r2_threshold": self.config["model_selection"]["min_r2_threshold"],
            },
            weights=self.config["model_selection"]["weights"],
        )

        self.best_model_info = best_model_info

        logger.info(
            f"Best model selected: {best_model_info['model_type']} "
            f"with {self.config['model_selection']['primary_metric']} score: {best_model_info['performance']:.4f}"
        )

        return best_model_info

    def register_best_model(self) -> str:
        """Register the best model in MLflow Model Registry with Unity Catalog support."""
        if not self.best_model_info:
            logger.error("No best model selected. Run model comparison first.")
            return ""

        if not self.config["model_registry"]["register_best_model"]:
            logger.info(
                "ℹ️ Model registration disabled (recommended for Unity Catalog workspaces)"
            )
            logger.info(
                "💡 Model artifacts are available in MLflow runs for manual registration"
            )
            return "registration_disabled"

        logger.info("Attempting model registration...")

        try:
            # Check if Unity Catalog is configured
            unity_config = self.config["model_registry"].get("unity_catalog", {})

            if unity_config.get("enabled", False):
                logger.info("Using Unity Catalog for model registration")
                from .unity_catalog_utils import register_model_with_unity_catalog

                # Find the best model's run ID from training results
                best_run_id = None
                for result in self.training_results:
                    if result.get("model_type") == self.best_model_info["model_type"]:
                        # Get the actual run ID from training results
                        best_run_id = result.get("run_id")
                        if best_run_id:
                            logger.info(
                                f"Found run ID for best model ({self.best_model_info['model_type']}): {best_run_id}"
                            )
                            break

                if best_run_id:
                    model_uri = f"runs:/{best_run_id}/model"

                    version = register_model_with_unity_catalog(
                        model_uri=model_uri,
                        model_name=self.config["model_registry"]["model_name"],
                        catalog=unity_config.get("catalog", "main"),
                        schema=unity_config.get("schema", "mlops"),
                        description=f"Best {self.best_model_info['model_type']} model",
                        tags={
                            "model_type": self.best_model_info["model_type"],
                            "performance": str(self.best_model_info["performance"]),
                        },
                    )

                    if version and not version.startswith("not_registered"):
                        logger.info(
                            f"✅ Model registered in Unity Catalog as version {version}"
                        )
                        return version
                    else:
                        logger.warning(f"⚠️ Model registration skipped: {version}")
                        return version or "unity_catalog_failed"
                else:
                    logger.error("Could not find run ID for best model")
                    logger.error("Available training results:")
                    for i, result in enumerate(self.training_results):
                        logger.error(
                            f"  Result {i}: model_type={result.get('model_type')}, run_id={result.get('run_id', 'MISSING')}"
                        )
                    return "run_id_not_found"

            else:
                logger.info(
                    "Using legacy model registry (may fail in Unity Catalog workspaces)"
                )

                # Legacy registration approach
                mock_model_info = {
                    "model_uri": f"models:/{self.config['model_registry']['model_name']}/1",
                    "run_id": "best_model_run",
                    "metrics": {
                        self.config["model_selection"][
                            "primary_metric"
                        ]: self.best_model_info["performance"]
                    },
                }

                version = self.model_registry.register_best_model(
                    model_info=mock_model_info,
                    model_name=self.config["model_registry"]["model_name"],
                    description=f"Best {self.best_model_info['model_type']} model for California housing prediction",
                    tags={
                        "model_type": self.best_model_info["model_type"],
                        "performance_metric": str(self.best_model_info["performance"]),
                        "training_timestamp": str(time.time()),
                    },
                )

                if version:
                    logger.info(f"Model registered as version {version}")
                    return version
                else:
                    logger.warning("Legacy model registration failed")
                    return "legacy_registration_failed"

        except Exception as e:
            error_msg = str(e)
            if "legacy workspace model registry is disabled" in error_msg:
                logger.warning(
                    "⚠️ Legacy model registry disabled - this is expected in Unity Catalog workspaces"
                )
                logger.info(
                    "💡 Model artifacts are logged and available in MLflow runs"
                )
                logger.info(
                    "💡 You can manually register models in Unity Catalog through the UI"
                )
                return "legacy_disabled_unity_catalog"
            else:
                logger.error(f"Error registering model: {e}")
                return "registration_error"

    def generate_outputs(self):
        """Generate outputs, reports, and visualizations."""
        if not self.config["output"]["save_results"]:
            return

        logger.info("Generating outputs...")

        # Create results directory
        results_dir = Path(self.config["output"]["results_dir"])
        results_dir.mkdir(exist_ok=True)

        # Save training results
        if self.training_results:
            results_file = results_dir / "training_results.json"
            with open(results_file, "w") as f:
                # Convert numpy types to native Python types for JSON serialization
                serializable_results = []
                for result in self.training_results:
                    serializable_result = {}
                    for key, value in result.items():
                        if isinstance(value, dict):
                            serializable_result[key] = {
                                k: float(v) if isinstance(v, (int, float)) else v
                                for k, v in value.items()
                            }
                        else:
                            serializable_result[key] = value
                    serializable_results.append(serializable_result)

                json.dump(serializable_results, f, indent=2, default=str)

            logger.info(f"Training results saved to {results_file}")

        # Save comparison results
        if not self.model_comparator.comparison_results.empty:
            comparison_file = results_dir / "model_comparison.csv"
            self.model_comparator.save_comparison_results(str(comparison_file))

        # Generate visualizations
        if self.config["output"]["create_visualizations"]:
            plot_dir = results_dir / "visualizations"
            self.model_comparator.create_comparison_visualizations(str(plot_dir))

        # Generate HTML report
        if self.config["output"]["generate_report"]:
            report_file = results_dir / "model_comparison_report.html"
            self.model_comparator.generate_model_report(str(report_file))

        # Save best model info
        if self.best_model_info:
            best_model_file = results_dir / "best_model_info.json"
            with open(best_model_file, "w") as f:
                json.dump(self.best_model_info, f, indent=2, default=str)

            logger.info(f"Best model info saved to {best_model_file}")

    def run_complete_pipeline(self):
        """Run the complete training pipeline."""
        logger.info("Starting complete ML training pipeline...")

        pipeline_start_time = time.time()

        try:
            # 1. Load data
            X_train, X_test, y_train, y_test = self.load_data()

            # 2. Train linear regression models
            if self.config["models_to_train"]["linear_regression"]["enabled"]:
                linear_results = self.train_linear_regression_models(
                    X_train, y_train, X_test, y_test
                )
                self.training_results.extend(linear_results)

            # 3. Train decision tree models
            if self.config["models_to_train"]["decision_tree"]["enabled"]:
                tree_results = self.train_decision_tree_models(
                    X_train, y_train, X_test, y_test
                )
                self.training_results.extend(tree_results)

            # 4. Run advanced hyperparameter tuning (if enabled)
            advanced_results = self.run_advanced_hyperparameter_tuning(X_train, y_train)
            if advanced_results:
                self.training_results.extend(advanced_results)

            # 5. Compare models and select the best
            self.compare_and_select_best_model()

            # 6. Register best model
            self.register_best_model()

            # 7. Generate outputs
            self.generate_outputs()

            pipeline_time = time.time() - pipeline_start_time
            logger.info(f"Complete pipeline finished in {pipeline_time:.2f} seconds")
            logger.info(f"Total models trained: {len(self.training_results)}")

            if self.best_model_info:
                logger.info(
                    f"Best model: {self.best_model_info['model_type']} "
                    f"(R² Score: {self.best_model_info['performance']:.4f})"
                )

        except Exception as e:
            logger.error(f"Pipeline failed: {e}")
            raise


def main():
    """Main function to run the training pipeline."""
    parser = argparse.ArgumentParser(
        description="Train California housing prediction models"
    )

    parser.add_argument("--config", type=str, help="Path to configuration file")
    parser.add_argument(
        "--tracking-uri", type=str, default="databricks", help="MLflow tracking URI"
    )
    parser.add_argument(
        "--experiment-name",
        type=str,
        default="california_housing_experiment",
        help="MLflow experiment name",
    )
    parser.add_argument(
        "--data-dir",
        type=str,
        default="data/processed",
        help="Directory containing processed data",
    )
    parser.add_argument(
        "--models",
        nargs="+",
        choices=[
            "linear",
            "ridge",
            "lasso",
            "elastic",
            "decision_tree",
            "random_forest",
            "gradient_boosting",
        ],
        help="Specific models to train",
    )
    parser.add_argument(
        "--no-hyperparameter-tuning",
        action="store_true",
        help="Disable hyperparameter tuning",
    )
    parser.add_argument(
        "--advanced-tuning",
        action="store_true",
        help="Enable advanced hyperparameter tuning with Optuna",
    )
    parser.add_argument(
        "--no-registry", action="store_true", help="Disable model registry"
    )

    args = parser.parse_args()

    # Load configuration
    config = None
    if args.config and os.path.exists(args.config):
        with open(args.config, "r") as f:
            config = json.load(f)
        logger.info(f"Loaded configuration from {args.config}")

    # Create pipeline
    pipeline = MLTrainingPipeline(config, args.tracking_uri)

    # Override config with command line arguments
    if args.experiment_name:
        pipeline.config["experiment_name"] = args.experiment_name

    if args.data_dir:
        pipeline.config["data_dir"] = args.data_dir

    if args.models:
        # Disable all models and enable only specified ones
        for model_type in pipeline.config["models_to_train"]:
            pipeline.config["models_to_train"][model_type]["enabled"] = False

        if any(m in args.models for m in ["linear", "ridge", "lasso", "elastic"]):
            pipeline.config["models_to_train"]["linear_regression"]["enabled"] = True
            if args.models:
                pipeline.config["models_to_train"]["linear_regression"]["variants"] = [
                    m
                    for m in args.models
                    if m in ["linear", "ridge", "lasso", "elastic"]
                ]

        if any(
            m in args.models
            for m in ["decision_tree", "random_forest", "gradient_boosting"]
        ):
            pipeline.config["models_to_train"]["decision_tree"]["enabled"] = True
            if args.models:
                pipeline.config["models_to_train"]["decision_tree"]["variants"] = [
                    m
                    for m in args.models
                    if m in ["decision_tree", "random_forest", "gradient_boosting"]
                ]

    if args.no_hyperparameter_tuning:
        for model_type in pipeline.config["models_to_train"]:
            pipeline.config["models_to_train"][model_type][
                "hyperparameter_tuning"
            ] = False

    if args.advanced_tuning:
        pipeline.config["advanced_tuning"]["enabled"] = True

    if args.no_registry:
        pipeline.config["model_registry"]["register_best_model"] = False

    # Run pipeline
    pipeline.run_complete_pipeline()


if __name__ == "__main__":
    main()
