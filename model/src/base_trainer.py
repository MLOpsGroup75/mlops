"""
Base trainer class for machine learning models with MLflow integration.
"""

import os
import pickle
import pandas as pd
import numpy as np
import mlflow
import mlflow.sklearn
from abc import ABC, abstractmethod
from typing import Dict, Any, Tuple, Optional, Union
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.model_selection import cross_val_score
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import logging

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class BaseTrainer(ABC):
    """Base class for model trainers with MLflow integration."""
    
    def __init__(self, experiment_name: str = "california_housing_experiment", 
                 tracking_uri: str = "databricks", random_state: int = 42):
        """
        Initialize base trainer.
        
        Args:
            experiment_name: MLflow experiment name
            tracking_uri: MLflow tracking URI (use "databricks" for Databricks SaaS)
            random_state: Random state for reproducibility
        """
        self.experiment_name = experiment_name
        self.tracking_uri = tracking_uri
        self.random_state = random_state
        self.model = None
        self.model_name = None
        self.model_params = {}
        
        # Set up MLflow
        self._setup_mlflow()
        
    def _setup_mlflow(self):
        """Set up MLflow tracking with improved Databricks support."""
        try:
            # For Databricks SaaS, set tracking URI to databricks
            if self.tracking_uri == "databricks":
                # Set Databricks credentials if available in environment
                if "DATABRICKS_HOST" in os.environ and "DATABRICKS_TOKEN" in os.environ:
                    mlflow.set_tracking_uri("databricks")
                    logger.info("Using Databricks SaaS for MLflow tracking")
                    
                    # Use Databricks utilities for proper experiment setup
                    from .databricks_utils import DatabricksExperimentManager
                    exp_manager = DatabricksExperimentManager()
                    
                    if exp_manager.setup_experiment(self.experiment_name):
                        # Get the actual experiment name that was used
                        current_exp = mlflow.get_experiment_by_name(self.experiment_name)
                        if current_exp is None:
                            # Find the experiment that was actually created
                            candidates = exp_manager.get_experiment_path_candidates(self.experiment_name)
                            for candidate in candidates:
                                exp = mlflow.get_experiment_by_name(candidate)
                                if exp is not None:
                                    self.experiment_name = candidate
                                    break
                        logger.info(f"✅ Databricks experiment ready: {self.experiment_name}")
                    else:
                        logger.error("❌ Failed to set up Databricks experiment")
                        # Fallback to local tracking
                        mlflow.set_tracking_uri("file:./mlruns")
                        logger.warning("🔄 Falling back to local MLflow tracking")
                else:
                    # Fallback to local tracking
                    mlflow.set_tracking_uri("file:./mlruns")
                    logger.warning("Databricks credentials not found, using local MLflow tracking")
            else:
                mlflow.set_tracking_uri(self.tracking_uri)
                
            # For non-Databricks or fallback cases, use simple experiment setup
            if not (self.tracking_uri == "databricks" and "DATABRICKS_HOST" in os.environ):
                try:
                    experiment = mlflow.get_experiment_by_name(self.experiment_name)
                    if experiment is None:
                        experiment_id = mlflow.create_experiment(self.experiment_name)
                        logger.info(f"Created new experiment: {self.experiment_name}")
                    else:
                        experiment_id = experiment.experiment_id
                        logger.info(f"Using existing experiment: {self.experiment_name}")
                        
                    mlflow.set_experiment(self.experiment_name)
                    
                except Exception as e:
                    logger.error(f"Error setting up experiment: {e}")
                    # Create a fallback experiment name
                    fallback_name = f"fallback_{pd.Timestamp.now().strftime('%Y%m%d_%H%M%S')}"
                    mlflow.set_experiment(fallback_name)
                    logger.warning(f"Using fallback experiment: {fallback_name}")
                
        except Exception as e:
            logger.error(f"Error setting up MLflow: {e}")
            # Set local tracking as ultimate fallback
            mlflow.set_tracking_uri("file:./mlruns")
            # Clean experiment name for local use
            clean_name = self.experiment_name.split("/")[-1] if "/" in self.experiment_name else self.experiment_name
            mlflow.set_experiment(clean_name)
    
    def load_data(self, data_dir: str = "data/processed") -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        """
        Load preprocessed training and test data.
        
        Args:
            data_dir: Directory containing processed data files
            
        Returns:
            Tuple of (X_train, X_test, y_train, y_test)
        """
        try:
            X_train = pd.read_csv(f"{data_dir}/X_train.csv")
            X_test = pd.read_csv(f"{data_dir}/X_test.csv")
            y_train = pd.read_csv(f"{data_dir}/y_train.csv")
            y_test = pd.read_csv(f"{data_dir}/y_test.csv")
            
            # Convert target to series if it's a DataFrame
            if isinstance(y_train, pd.DataFrame):
                y_train = y_train.iloc[:, 0]
            if isinstance(y_test, pd.DataFrame):
                y_test = y_test.iloc[:, 0]
                
            logger.info(f"Data loaded successfully:")
            logger.info(f"  Training set: {X_train.shape[0]} samples, {X_train.shape[1]} features")
            logger.info(f"  Test set: {X_test.shape[0]} samples, {X_test.shape[1]} features")
            
            return X_train, X_test, y_train, y_test
            
        except Exception as e:
            logger.error(f"Error loading data: {e}")
            raise
    
    def calculate_metrics(self, y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, float]:
        """
        Calculate regression metrics.
        
        Args:
            y_true: True target values
            y_pred: Predicted target values
            
        Returns:
            Dictionary of metrics
        """
        metrics = {
            'mse': mean_squared_error(y_true, y_pred),
            'rmse': np.sqrt(mean_squared_error(y_true, y_pred)),
            'mae': mean_absolute_error(y_true, y_pred),
            'r2_score': r2_score(y_true, y_pred),
            'mape': np.mean(np.abs((y_true - y_pred) / y_true)) * 100
        }
        
        # Add additional metrics
        metrics['max_error'] = np.max(np.abs(y_true - y_pred))
        metrics['mean_error'] = np.mean(y_true - y_pred)
        
        return metrics
    
    def cross_validate_model(self, X: pd.DataFrame, y: pd.Series, cv: int = 5) -> Dict[str, float]:
        """
        Perform cross-validation on the model.
        
        Args:
            X: Feature matrix
            y: Target vector
            cv: Number of cross-validation folds
            
        Returns:
            Dictionary of cross-validation scores
        """
        if self.model is None:
            raise ValueError("Model must be fitted before cross-validation")
            
        # Perform cross-validation
        cv_scores = cross_val_score(self.model, X, y, cv=cv, scoring='r2')
        
        cv_metrics = {
            'cv_r2_mean': cv_scores.mean(),
            'cv_r2_std': cv_scores.std(),
            'cv_r2_scores': cv_scores.tolist()
        }
        
        logger.info(f"Cross-validation R² score: {cv_metrics['cv_r2_mean']:.4f} (+/- {cv_metrics['cv_r2_std'] * 2:.4f})")
        
        return cv_metrics
    
    def create_visualizations(self, y_true: np.ndarray, y_pred: np.ndarray, 
                            feature_names: Optional[list] = None,
                            model_name: str = "model") -> Dict[str, str]:
        """
        Create model visualization plots.
        
        Args:
            y_true: True target values
            y_pred: Predicted target values
            feature_names: List of feature names for importance plot
            model_name: Name of the model for plot titles
            
        Returns:
            Dictionary of plot file paths
        """
        plots_dir = Path("../../plots")
        plots_dir.mkdir(exist_ok=True)
        
        plot_paths = {}
        
        # 1. Actual vs Predicted plot
        plt.figure(figsize=(10, 8))
        plt.scatter(y_true, y_pred, alpha=0.6)
        plt.plot([y_true.min(), y_true.max()], [y_true.min(), y_true.max()], 'r--', lw=2)
        plt.xlabel('Actual Values')
        plt.ylabel('Predicted Values')
        plt.title(f'{model_name.title()} - Actual vs Predicted Values')
        plt.grid(True, alpha=0.3)
        
        # Add R² score to plot
        r2 = r2_score(y_true, y_pred)
        plt.text(0.05, 0.95, f'R² = {r2:.4f}', transform=plt.gca().transAxes, 
                bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
        
        actual_vs_pred_path = plots_dir / f"{model_name}_actual_vs_predicted.png"
        plt.savefig(actual_vs_pred_path, dpi=300, bbox_inches='tight')
        plt.close()
        plot_paths['actual_vs_predicted'] = str(actual_vs_pred_path)
        
        # 2. Residuals plot
        residuals = y_true - y_pred
        plt.figure(figsize=(10, 6))
        plt.scatter(y_pred, residuals, alpha=0.6)
        plt.axhline(y=0, color='r', linestyle='--')
        plt.xlabel('Predicted Values')
        plt.ylabel('Residuals')
        plt.title(f'{model_name.title()} - Residuals Plot')
        plt.grid(True, alpha=0.3)
        
        residuals_path = plots_dir / f"{model_name}_residuals.png"
        plt.savefig(residuals_path, dpi=300, bbox_inches='tight')
        plt.close()
        plot_paths['residuals'] = str(residuals_path)
        
        # 3. Residuals distribution
        plt.figure(figsize=(10, 6))
        plt.hist(residuals, bins=50, alpha=0.7, edgecolor='black')
        plt.xlabel('Residuals')
        plt.ylabel('Frequency')
        plt.title(f'{model_name.title()} - Residuals Distribution')
        plt.axvline(x=0, color='r', linestyle='--')
        plt.grid(True, alpha=0.3)
        
        residuals_dist_path = plots_dir / f"{model_name}_residuals_distribution.png"
        plt.savefig(residuals_dist_path, dpi=300, bbox_inches='tight')
        plt.close()
        plot_paths['residuals_distribution'] = str(residuals_dist_path)
        
        # 4. Feature importance (if available and feature names provided)
        if hasattr(self.model, 'feature_importances_') and feature_names:
            importances = self.model.feature_importances_
            indices = np.argsort(importances)[::-1]
            
            plt.figure(figsize=(12, 8))
            plt.bar(range(len(importances)), importances[indices])
            plt.xlabel('Features')
            plt.ylabel('Importance')
            plt.title(f'{model_name.title()} - Feature Importance')
            plt.xticks(range(len(importances)), [feature_names[i] for i in indices], rotation=45)
            plt.grid(True, alpha=0.3)
            
            feature_importance_path = plots_dir / f"{model_name}_feature_importance.png"
            plt.savefig(feature_importance_path, dpi=300, bbox_inches='tight')
            plt.close()
            plot_paths['feature_importance'] = str(feature_importance_path)
            
        elif hasattr(self.model, 'coef_') and feature_names:
            # For linear models, plot coefficients
            coef = np.abs(self.model.coef_)
            indices = np.argsort(coef)[::-1]
            
            plt.figure(figsize=(12, 8))
            plt.bar(range(len(coef)), coef[indices])
            plt.xlabel('Features')
            plt.ylabel('Coefficient Magnitude')
            plt.title(f'{model_name.title()} - Feature Coefficients')
            plt.xticks(range(len(coef)), [feature_names[i] for i in indices], rotation=45)
            plt.grid(True, alpha=0.3)
            
            feature_importance_path = plots_dir / f"{model_name}_feature_importance.png"
            plt.savefig(feature_importance_path, dpi=300, bbox_inches='tight')
            plt.close()
            plot_paths['feature_importance'] = str(feature_importance_path)
        
        logger.info(f"Created {len(plot_paths)} visualization plots")
        return plot_paths
    
    def log_to_mlflow(self, params: Dict[str, Any], metrics: Dict[str, float], 
                     artifacts: Dict[str, str], model_name: str, 
                     X_sample: pd.DataFrame = None):
        """
        Log parameters, metrics, and artifacts to MLflow.
        
        Args:
            params: Model parameters
            metrics: Model metrics
            artifacts: Dictionary of artifact file paths
            model_name: Name of the model
        """
        with mlflow.start_run(run_name=f"{model_name}_{pd.Timestamp.now().strftime('%Y%m%d_%H%M%S')}"):
            # Log parameters
            for key, value in params.items():
                mlflow.log_param(key, value)
            
            # Log metrics
            for key, value in metrics.items():
                if isinstance(value, (int, float)) and not isinstance(value, bool):
                    mlflow.log_metric(key, value)
            
            # Log artifacts (plots, models, etc.)
            for artifact_name, artifact_path in artifacts.items():
                if os.path.exists(artifact_path):
                    mlflow.log_artifact(artifact_path)
            
            # Log the model (without automatic registration for Unity Catalog compatibility)
            if self.model is not None:
                # Create input example for model signature (suppresses warnings)
                input_example = None
                if X_sample is not None and not X_sample.empty:
                    try:
                        input_example = X_sample.head(3)
                    except Exception:
                        input_example = None
                
                mlflow.sklearn.log_model(
                    sk_model=self.model,
                    name="model",
                    input_example=input_example
                )
            
            # Log additional metadata
            mlflow.set_tag("model_type", model_name)
            mlflow.set_tag("framework", "scikit-learn")
            mlflow.set_tag("dataset", "california_housing")
            
            # Get run info
            run = mlflow.active_run()
            logger.info(f"MLflow run logged: {run.info.run_id}")
            
            return run.info.run_id
    
    def save_model(self, model_path: str):
        """
        Save the trained model to disk.
        
        Args:
            model_path: Path to save the model
        """
        if self.model is None:
            raise ValueError("No model to save. Train a model first.")
        
        # Create directory if it doesn't exist
        os.makedirs(os.path.dirname(model_path), exist_ok=True)
        
        # Save model
        with open(model_path, 'wb') as f:
            pickle.dump(self.model, f)
        
        logger.info(f"Model saved to {model_path}")
    
    @abstractmethod
    def train(self, X_train: pd.DataFrame, y_train: pd.Series, 
              X_test: pd.DataFrame, y_test: pd.Series, **kwargs) -> Dict[str, Any]:
        """
        Train the model. Must be implemented by subclasses.
        
        Args:
            X_train: Training features
            y_train: Training targets
            X_test: Test features
            y_test: Test targets
            **kwargs: Additional training parameters
            
        Returns:
            Dictionary containing training results
        """
        pass
    
    @abstractmethod
    def tune_hyperparameters(self, X_train: pd.DataFrame, y_train: pd.Series, 
                           cv: int = 5, **kwargs) -> Dict[str, Any]:
        """
        Tune hyperparameters. Must be implemented by subclasses.
        
        Args:
            X_train: Training features
            y_train: Training targets
            cv: Number of cross-validation folds
            **kwargs: Additional tuning parameters
            
        Returns:
            Dictionary containing best parameters and scores
        """
        pass
