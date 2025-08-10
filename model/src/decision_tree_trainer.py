"""
Decision tree trainer with hyperparameter tuning and MLflow integration.
"""

import pandas as pd
import numpy as np
from typing import Dict, Any, Tuple
import mlflow.sklearn
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.model_selection import GridSearchCV, RandomizedSearchCV
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
import logging

from .base_trainer import BaseTrainer

logger = logging.getLogger(__name__)


class DecisionTreeTrainer(BaseTrainer):
    """Decision tree trainer with multiple tree-based algorithms."""

    def __init__(
        self,
        model_type: str = "decision_tree",
        experiment_name: str = "california_housing_tree",
        tracking_uri: str = "databricks",
        random_state: int = 42,
    ):
        """
        Initialize decision tree trainer.

        Args:
            model_type: Type of tree model ('decision_tree', 'random_forest', 'gradient_boosting')
            experiment_name: MLflow experiment name
            tracking_uri: MLflow tracking URI
            random_state: Random state for reproducibility
        """
        super().__init__(experiment_name, tracking_uri, random_state)
        self.model_type = model_type
        self.model_name = (
            f"decision_tree" if model_type == "decision_tree" else model_type
        )
        self.pipeline = None
        self.best_params = {}

        # Validate model type
        valid_types = ["decision_tree", "random_forest", "gradient_boosting"]
        if model_type not in valid_types:
            raise ValueError(f"model_type must be one of {valid_types}")

    def _get_base_model(self):
        """Get the base model based on model type."""
        if self.model_type == "decision_tree":
            return DecisionTreeRegressor(random_state=self.random_state)
        elif self.model_type == "random_forest":
            return RandomForestRegressor(random_state=self.random_state, n_jobs=-1)
        elif self.model_type == "gradient_boosting":
            return GradientBoostingRegressor(random_state=self.random_state)

    def _get_param_grid(self, search_type: str = "grid") -> Dict[str, Any]:
        """
        Get parameter grid for hyperparameter tuning.

        Args:
            search_type: Type of search ('grid' or 'random')

        Returns:
            Parameter grid for the model
        """
        if self.model_type == "decision_tree":
            if search_type == "grid":
                return {
                    "model__max_depth": [3, 5, 7, 10, 15, None],
                    "model__min_samples_split": [2, 5, 10, 20],
                    "model__min_samples_leaf": [1, 2, 5, 10],
                    "model__max_features": ["sqrt", "log2", None],
                }
            else:
                return {
                    "model__max_depth": [None] + list(range(3, 21)),
                    "model__min_samples_split": range(2, 21),
                    "model__min_samples_leaf": range(1, 11),
                    "model__max_features": ["sqrt", "log2", None],
                    "model__ccp_alpha": np.linspace(0.0, 0.1, 20),
                }

        elif self.model_type == "random_forest":
            if search_type == "grid":
                return {
                    "model__n_estimators": [50, 100, 200],
                    "model__max_depth": [5, 10, 15, None],
                    "model__min_samples_split": [2, 5, 10],
                    "model__min_samples_leaf": [1, 2, 4],
                    "model__max_features": ["sqrt", "log2"],
                }
            else:
                return {
                    "model__n_estimators": range(50, 301, 50),
                    "model__max_depth": [None] + list(range(5, 21)),
                    "model__min_samples_split": range(2, 21),
                    "model__min_samples_leaf": range(1, 11),
                    "model__max_features": ["sqrt", "log2"],
                    "model__bootstrap": [True, False],
                }

        elif self.model_type == "gradient_boosting":
            if search_type == "grid":
                return {
                    "model__n_estimators": [50, 100, 200],
                    "model__learning_rate": [0.01, 0.1, 0.2],
                    "model__max_depth": [3, 5, 7],
                    "model__min_samples_split": [2, 5, 10],
                    "model__min_samples_leaf": [1, 2, 4],
                }
            else:
                return {
                    "model__n_estimators": range(50, 301, 25),
                    "model__learning_rate": np.logspace(-3, 0, 20),
                    "model__max_depth": range(3, 11),
                    "model__min_samples_split": range(2, 21),
                    "model__min_samples_leaf": range(1, 11),
                    "model__subsample": np.linspace(0.6, 1.0, 10),
                }

    def train(
        self,
        X_train: pd.DataFrame,
        y_train: pd.Series,
        X_test: pd.DataFrame,
        y_test: pd.Series,
        use_scaling: bool = False,
        **kwargs,
    ) -> Dict[str, Any]:
        """
        Train the decision tree model.

        Args:
            X_train: Training features
            y_train: Training targets
            X_test: Test features
            y_test: Test targets
            use_scaling: Whether to use feature scaling (generally not needed for trees)
            **kwargs: Additional training parameters

        Returns:
            Dictionary containing training results
        """
        logger.info(f"Training {self.model_type} model")

        # Create pipeline
        steps = []
        if use_scaling:
            steps.append(("scaler", StandardScaler()))
        steps.append(("model", self._get_base_model()))

        self.pipeline = Pipeline(steps)

        # Train the model
        self.pipeline.fit(X_train, y_train)
        self.model = self.pipeline  # For compatibility with base class

        # Make predictions
        y_train_pred = self.pipeline.predict(X_train)
        y_test_pred = self.pipeline.predict(X_test)

        # Calculate metrics
        train_metrics = self.calculate_metrics(y_train, y_train_pred)
        test_metrics = self.calculate_metrics(y_test, y_test_pred)

        # Cross-validation
        cv_metrics = self.cross_validate_model(X_train, y_train, cv=5)

        # Feature importance (trees have built-in feature importance)
        if hasattr(self.pipeline.named_steps["model"], "feature_importances_"):
            feature_importance = self.pipeline.named_steps["model"].feature_importances_
            feature_importance_dict = dict(zip(X_train.columns, feature_importance))
        else:
            feature_importance_dict = {}

        # Create visualizations
        plot_paths = self.create_visualizations(
            y_test,
            y_test_pred,
            feature_names=X_train.columns.tolist(),
            model_name=self.model_name,
        )

        # Prepare results
        results = {
            "model_type": self.model_type,
            "use_scaling": use_scaling,
            "train_metrics": train_metrics,
            "test_metrics": test_metrics,
            "cv_metrics": cv_metrics,
            "feature_importance": feature_importance_dict,
            "plot_paths": plot_paths,
            "model_params": self.pipeline.get_params(),
            "feature_names": X_train.columns.tolist(),
            "n_features": X_train.shape[1],
            "n_samples_train": X_train.shape[0],
            "n_samples_test": X_test.shape[0],
        }

        # Log to MLflow
        mlflow_params = {
            "model_type": self.model_type,
            "use_scaling": use_scaling,
            "n_features": X_train.shape[1],
            "n_samples_train": X_train.shape[0],
            "n_samples_test": X_test.shape[0],
        }

        # Add model-specific parameters
        model_params = self.pipeline.named_steps["model"].get_params()
        for key, value in model_params.items():
            if isinstance(value, (int, float, str, bool)):
                mlflow_params[f"model__{key}"] = value

        mlflow_metrics = {
            **{f"train_{k}": v for k, v in train_metrics.items()},
            **{f"test_{k}": v for k, v in test_metrics.items()},
            **cv_metrics,
        }

        # Add feature importance metrics
        for feature_name, importance in feature_importance_dict.items():
            mlflow_metrics[f"feature_importance_{feature_name}"] = importance

        # Log to MLflow and capture run ID
        run_id = self.log_to_mlflow(
            mlflow_params, mlflow_metrics, plot_paths, self.model_name, X_train
        )

        # Add run ID to results for model registration
        results["run_id"] = run_id
        results["model_uri"] = f"runs:/{run_id}/model"

        # Save model
        model_path = f"../../model/artifacts/{self.model_name}_model.joblib"
        self.save_model(model_path)

        logger.info(
            f"Training completed. Test R² score: {test_metrics['r2_score']:.4f}"
        )

        return results

    def tune_hyperparameters(
        self,
        X_train: pd.DataFrame,
        y_train: pd.Series,
        cv: int = 5,
        search_type: str = "random",
        n_iter: int = 50,
        use_scaling: bool = False,
        **kwargs,
    ) -> Dict[str, Any]:
        """
        Tune hyperparameters for the decision tree model.

        Args:
            X_train: Training features
            y_train: Training targets
            cv: Number of cross-validation folds
            search_type: Type of search ('grid' or 'random')
            n_iter: Number of iterations for random search
            use_scaling: Whether to use feature scaling
            **kwargs: Additional tuning parameters

        Returns:
            Dictionary containing best parameters and scores
        """
        logger.info(
            f"Tuning hyperparameters for {self.model_type} using {search_type} search"
        )

        # Create pipeline
        steps = []
        if use_scaling:
            steps.append(("scaler", StandardScaler()))
        steps.append(("model", self._get_base_model()))

        pipeline = Pipeline(steps)

        # Get parameter grid
        param_grid = self._get_param_grid(search_type)

        # Choose search strategy
        if search_type == "grid":
            search = GridSearchCV(
                pipeline, param_grid, cv=cv, scoring="r2", n_jobs=-1, verbose=1
            )
        else:
            search = RandomizedSearchCV(
                pipeline,
                param_grid,
                cv=cv,
                scoring="r2",
                n_iter=n_iter,
                n_jobs=-1,
                verbose=1,
                random_state=self.random_state,
            )

        # Perform search
        search.fit(X_train, y_train)

        # Store best model and parameters
        self.pipeline = search.best_estimator_
        self.model = self.pipeline
        self.best_params = search.best_params_

        # Get predictions from best model
        y_train_pred = self.pipeline.predict(X_train)

        # Calculate metrics
        train_metrics = self.calculate_metrics(y_train, y_train_pred)
        cv_metrics = self.cross_validate_model(X_train, y_train, cv=cv)

        # Feature importance
        if hasattr(self.pipeline.named_steps["model"], "feature_importances_"):
            feature_importance = self.pipeline.named_steps["model"].feature_importances_
            feature_importance_dict = dict(zip(X_train.columns, feature_importance))
        else:
            feature_importance_dict = {}

        # Prepare results
        results = {
            "best_params": self.best_params,
            "best_score": search.best_score_,
            "cv_results": search.cv_results_,
            "train_metrics": train_metrics,
            "cv_metrics": cv_metrics,
            "feature_importance": feature_importance_dict,
            "search_type": search_type,
            "n_iter": (
                n_iter if search_type == "random" else len(search.cv_results_["params"])
            ),
            "model_type": self.model_type,
            "use_scaling": use_scaling,
        }

        # Log hyperparameter tuning to MLflow
        with_mlflow = kwargs.get("with_mlflow", True)
        if with_mlflow:
            mlflow_params = {
                "hyperparameter_optimization": True,
                "search_type": search_type,
                "cv_folds": cv,
                "model_type": self.model_type,
                "use_scaling": use_scaling,
                "n_features": X_train.shape[1],
                "n_samples": X_train.shape[0],
            }

            # Add best parameters
            for key, value in self.best_params.items():
                if isinstance(value, (int, float, str, bool)):
                    mlflow_params[f"best_{key}"] = value

            mlflow_metrics = {
                "best_cv_score": search.best_score_,
                **{f"train_{k}": v for k, v in train_metrics.items()},
                **cv_metrics,
            }

            # Add feature importance metrics
            for feature_name, importance in feature_importance_dict.items():
                mlflow_metrics[f"feature_importance_{feature_name}"] = importance

            run_id = self.log_to_mlflow(
                mlflow_params,
                mlflow_metrics,
                {},
                f"{self.model_name}_hyperparameter_tuning",
                X_train,
            )

            # Add run ID to tuning results
            results["run_id"] = run_id
            results["model_uri"] = f"runs:/{run_id}/model"

        logger.info(
            f"Hyperparameter tuning completed. Best CV score: {search.best_score_:.4f}"
        )
        logger.info(f"Best parameters: {self.best_params}")

        return results

    def train_with_hyperparameter_tuning(
        self,
        X_train: pd.DataFrame,
        y_train: pd.Series,
        X_test: pd.DataFrame,
        y_test: pd.Series,
        cv: int = 5,
        search_type: str = "random",
        n_iter: int = 50,
        use_scaling: bool = False,
        **kwargs,
    ) -> Dict[str, Any]:
        """
        Train model with hyperparameter tuning and full evaluation.

        Args:
            X_train: Training features
            y_train: Training targets
            X_test: Test features
            y_test: Test targets
            cv: Number of cross-validation folds
            search_type: Type of search ('grid' or 'random')
            n_iter: Number of iterations for random search
            use_scaling: Whether to use feature scaling
            **kwargs: Additional parameters

        Returns:
            Dictionary containing all results
        """
        # First tune hyperparameters
        tuning_results = self.tune_hyperparameters(
            X_train,
            y_train,
            cv=cv,
            search_type=search_type,
            n_iter=n_iter,
            use_scaling=use_scaling,
            with_mlflow=False,
        )

        # Now evaluate on test set with best model
        if self.pipeline is not None:
            y_test_pred = self.pipeline.predict(X_test)
            test_metrics = self.calculate_metrics(y_test, y_test_pred)

            # Create visualizations
            plot_paths = self.create_visualizations(
                y_test,
                y_test_pred,
                feature_names=X_train.columns.tolist(),
                model_name=f"{self.model_name}_tuned",
            )

            # Combine results
            final_results = {
                **tuning_results,
                "test_metrics": test_metrics,
                "plot_paths": plot_paths,
                "feature_names": X_train.columns.tolist(),
            }

            # Log final results to MLflow
            mlflow_params = {
                "hyperparameter_optimization": True,
                "search_type": search_type,
                "cv_folds": cv,
                "model_type": self.model_type,
                "use_scaling": use_scaling,
                "n_features": X_train.shape[1],
                "n_samples_train": X_train.shape[0],
                "n_samples_test": X_test.shape[0],
            }

            # Add best parameters
            for key, value in self.best_params.items():
                if isinstance(value, (int, float, str, bool)):
                    mlflow_params[f"final_{key}"] = value

            mlflow_metrics = {
                "optimization_best_score": tuning_results["best_score"],
                **{f"train_{k}": v for k, v in tuning_results["train_metrics"].items()},
                **{f"test_{k}": v for k, v in test_metrics.items()},
                **tuning_results["cv_metrics"],
            }

            # Add feature importance metrics
            for feature_name, importance in tuning_results[
                "feature_importance"
            ].items():
                mlflow_metrics[f"feature_importance_{feature_name}"] = importance

            run_id = self.log_to_mlflow(
                mlflow_params, mlflow_metrics, plot_paths, self.model_name, X_train
            )

            # Add run ID to final results
            final_results["run_id"] = run_id
            final_results["model_uri"] = f"runs:/{run_id}/model"

            # Save final model
            model_path = f"../../model/artifacts/{self.model_name}_tuned.joblib"
            self.save_model(model_path)

            logger.info(f"Training with hyperparameter tuning completed.")
            logger.info(f"Final test R² score: {test_metrics['r2_score']:.4f}")

            return final_results

        return tuning_results
