"""
Advanced hyperparameter tuning utilities with Optuna integration.
"""

import pandas as pd
import numpy as np
from typing import Dict, Any, Callable, Optional, Union, List
import logging
import mlflow
import mlflow.sklearn
from sklearn.model_selection import cross_val_score
from sklearn.base import BaseEstimator
import warnings

# Optional Optuna import
try:
    import optuna
    from optuna.integration.mlflow import MLflowCallback
    OPTUNA_AVAILABLE = True
except ImportError:
    OPTUNA_AVAILABLE = False
    optuna = None

logger = logging.getLogger(__name__)


class HyperparameterTuner:
    """Advanced hyperparameter tuning with Optuna and MLflow integration."""
    
    def __init__(self, tracking_uri: str = "databricks", random_state: int = 42):
        """
        Initialize hyperparameter tuner.
        
        Args:
            tracking_uri: MLflow tracking URI
            random_state: Random state for reproducibility
        """
        self.tracking_uri = tracking_uri
        self.random_state = random_state
        self.study = None
        self.best_params = {}
        self.best_score = None
        
        if not OPTUNA_AVAILABLE:
            logger.warning("Optuna not available. Install with: pip install optuna")
    
    def create_objective_function(self, model_class: type, X: pd.DataFrame, y: pd.Series,
                                param_distributions: Dict[str, Any],
                                cv: int = 5, scoring: str = 'r2',
                                use_pipeline: bool = True,
                                preprocessing_steps: List = None) -> Callable:
        """
        Create objective function for Optuna optimization.
        
        Args:
            model_class: Model class to optimize
            X: Training features
            y: Training targets
            param_distributions: Parameter distributions for optimization
            cv: Number of cross-validation folds
            scoring: Scoring metric
            use_pipeline: Whether to use sklearn pipeline
            preprocessing_steps: List of preprocessing steps
            
        Returns:
            Objective function for Optuna
        """
        def objective(trial):
            # Sample hyperparameters
            params = {}
            for param_name, param_config in param_distributions.items():
                if param_config['type'] == 'float':
                    if param_config.get('log', False):
                        params[param_name] = trial.suggest_float(
                            param_name, param_config['low'], param_config['high'], log=True
                        )
                    else:
                        params[param_name] = trial.suggest_float(
                            param_name, param_config['low'], param_config['high']
                        )
                elif param_config['type'] == 'int':
                    if param_config.get('log', False):
                        params[param_name] = trial.suggest_int(
                            param_name, param_config['low'], param_config['high'], log=True
                        )
                    else:
                        params[param_name] = trial.suggest_int(
                            param_name, param_config['low'], param_config['high']
                        )
                elif param_config['type'] == 'categorical':
                    params[param_name] = trial.suggest_categorical(
                        param_name, param_config['choices']
                    )
            
            # Create model with sampled parameters
            try:
                if use_pipeline:
                    from sklearn.pipeline import Pipeline
                    steps = preprocessing_steps or []
                    steps.append(('model', model_class(**params, random_state=self.random_state)))
                    model = Pipeline(steps)
                else:
                    model = model_class(**params, random_state=self.random_state)
                
                # Perform cross-validation
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore")
                    cv_scores = cross_val_score(model, X, y, cv=cv, scoring=scoring, n_jobs=-1)
                
                # Return mean score
                return cv_scores.mean()
                
            except Exception as e:
                logger.warning(f"Trial failed with params {params}: {e}")
                # Return a bad score to discard this trial
                return float('-inf') if scoring in ['r2', 'accuracy'] else float('inf')
        
        return objective
    
    def optimize_with_optuna(self, objective_func: Callable, n_trials: int = 100,
                           study_name: str = None, direction: str = 'maximize',
                           timeout: Optional[int] = None,
                           mlflow_logging: bool = True) -> Dict[str, Any]:
        """
        Optimize hyperparameters using Optuna.
        
        Args:
            objective_func: Objective function to optimize
            n_trials: Number of optimization trials
            study_name: Name of the Optuna study
            direction: Optimization direction ('maximize' or 'minimize')
            timeout: Timeout in seconds
            mlflow_logging: Whether to log to MLflow
            
        Returns:
            Dictionary with optimization results
        """
        if not OPTUNA_AVAILABLE:
            raise ImportError("Optuna is required for advanced hyperparameter tuning")
        
        # Create study
        study_name = study_name or f"study_{pd.Timestamp.now().strftime('%Y%m%d_%H%M%S')}"
        
        # Set up MLflow callback if requested
        callbacks = []
        if mlflow_logging:
            try:
                mlflow_callback = MLflowCallback(
                    tracking_uri=self.tracking_uri,
                    metric_name="objective_value"
                )
                callbacks.append(mlflow_callback)
            except Exception as e:
                logger.warning(f"Could not set up MLflow callback: {e}")
        
        self.study = optuna.create_study(
            direction=direction,
            study_name=study_name,
            sampler=optuna.samplers.TPESampler(seed=self.random_state)
        )
        
        # Optimize
        logger.info(f"Starting Optuna optimization with {n_trials} trials")
        self.study.optimize(
            objective_func,
            n_trials=n_trials,
            timeout=timeout,
            callbacks=callbacks,
            show_progress_bar=True
        )
        
        # Store results
        self.best_params = self.study.best_params
        self.best_score = self.study.best_value
        
        results = {
            'best_params': self.best_params,
            'best_score': self.best_score,
            'n_trials': len(self.study.trials),
            'study': self.study,
            'optimization_history': [trial.value for trial in self.study.trials if trial.value is not None]
        }
        
        logger.info(f"Optimization completed. Best score: {self.best_score:.4f}")
        logger.info(f"Best params: {self.best_params}")
        
        return results
    
    def get_linear_regression_param_distributions(self, model_type: str) -> Dict[str, Any]:
        """
        Get parameter distributions for linear regression models.
        
        Args:
            model_type: Type of linear model ('ridge', 'lasso', 'elastic')
            
        Returns:
            Parameter distributions dictionary
        """
        if model_type == 'ridge':
            return {
                'alpha': {
                    'type': 'float',
                    'low': 1e-5,
                    'high': 1e3,
                    'log': True
                }
            }
        elif model_type == 'lasso':
            return {
                'alpha': {
                    'type': 'float',
                    'low': 1e-5,
                    'high': 1e2,
                    'log': True
                }
            }
        elif model_type == 'elastic':
            return {
                'alpha': {
                    'type': 'float',
                    'low': 1e-5,
                    'high': 1e2,
                    'log': True
                },
                'l1_ratio': {
                    'type': 'float',
                    'low': 0.01,
                    'high': 0.99
                }
            }
        else:
            return {}
    
    def get_decision_tree_param_distributions(self, model_type: str) -> Dict[str, Any]:
        """
        Get parameter distributions for tree-based models.
        
        Args:
            model_type: Type of tree model ('decision_tree', 'random_forest', 'gradient_boosting')
            
        Returns:
            Parameter distributions dictionary
        """
        if model_type == 'decision_tree':
            return {
                'max_depth': {
                    'type': 'int',
                    'low': 3,
                    'high': 20
                },
                'min_samples_split': {
                    'type': 'int',
                    'low': 2,
                    'high': 20
                },
                'min_samples_leaf': {
                    'type': 'int',
                    'low': 1,
                    'high': 10
                },
                'max_features': {
                    'type': 'categorical',
                    'choices': ['sqrt', 'log2', None]
                },
                'ccp_alpha': {
                    'type': 'float',
                    'low': 0.0,
                    'high': 0.1
                }
            }
        elif model_type == 'random_forest':
            return {
                'n_estimators': {
                    'type': 'int',
                    'low': 50,
                    'high': 300
                },
                'max_depth': {
                    'type': 'int',
                    'low': 5,
                    'high': 20
                },
                'min_samples_split': {
                    'type': 'int',
                    'low': 2,
                    'high': 20
                },
                'min_samples_leaf': {
                    'type': 'int',
                    'low': 1,
                    'high': 10
                },
                'max_features': {
                    'type': 'categorical',
                    'choices': ['sqrt', 'log2']
                },
                'bootstrap': {
                    'type': 'categorical',
                    'choices': [True, False]
                }
            }
        elif model_type == 'gradient_boosting':
            return {
                'n_estimators': {
                    'type': 'int',
                    'low': 50,
                    'high': 300
                },
                'learning_rate': {
                    'type': 'float',
                    'low': 0.01,
                    'high': 0.3,
                    'log': True
                },
                'max_depth': {
                    'type': 'int',
                    'low': 3,
                    'high': 10
                },
                'min_samples_split': {
                    'type': 'int',
                    'low': 2,
                    'high': 20
                },
                'min_samples_leaf': {
                    'type': 'int',
                    'low': 1,
                    'high': 10
                },
                'subsample': {
                    'type': 'float',
                    'low': 0.6,
                    'high': 1.0
                }
            }
        else:
            return {}
    
    def tune_model_with_optuna(self, model_class: type, model_type: str,
                             X: pd.DataFrame, y: pd.Series,
                             n_trials: int = 100, cv: int = 5,
                             use_preprocessing: bool = True,
                             **kwargs) -> Dict[str, Any]:
        """
        Tune a model using Optuna with predefined parameter distributions.
        
        Args:
            model_class: Model class to tune
            model_type: Type of model for parameter selection
            X: Training features
            y: Training targets
            n_trials: Number of Optuna trials
            cv: Number of cross-validation folds
            use_preprocessing: Whether to include preprocessing in pipeline
            **kwargs: Additional arguments for optimization
            
        Returns:
            Dictionary with tuning results
        """
        # Get parameter distributions
        if 'linear' in model_type.lower() or model_type in ['ridge', 'lasso', 'elastic']:
            param_distributions = self.get_linear_regression_param_distributions(model_type)
        else:
            param_distributions = self.get_decision_tree_param_distributions(model_type)
        
        if not param_distributions:
            logger.warning(f"No parameter distributions defined for model type: {model_type}")
            return {}
        
        # Set up preprocessing if requested
        preprocessing_steps = []
        if use_preprocessing and 'tree' not in model_type.lower():
            from sklearn.preprocessing import StandardScaler
            preprocessing_steps.append(('scaler', StandardScaler()))
        
        # Create objective function
        objective_func = self.create_objective_function(
            model_class=model_class,
            X=X,
            y=y,
            param_distributions=param_distributions,
            cv=cv,
            scoring='r2',
            use_pipeline=len(preprocessing_steps) > 0,
            preprocessing_steps=preprocessing_steps
        )
        
        # Optimize
        results = self.optimize_with_optuna(
            objective_func=objective_func,
            n_trials=n_trials,
            study_name=f"{model_type}_optimization",
            direction='maximize',
            **kwargs
        )
        
        # Add model information
        results['model_type'] = model_type
        results['model_class'] = model_class.__name__
        results['preprocessing_used'] = len(preprocessing_steps) > 0
        
        return results
    
    def plot_optimization_history(self, save_path: str = None):
        """
        Plot optimization history.
        
        Args:
            save_path: Path to save the plot
        """
        if self.study is None:
            logger.error("No study available. Run optimization first.")
            return
        
        try:
            import matplotlib.pyplot as plt
            
            # Get optimization history
            trials = [trial for trial in self.study.trials if trial.value is not None]
            values = [trial.value for trial in trials]
            
            plt.figure(figsize=(12, 6))
            
            # Plot optimization history
            plt.subplot(1, 2, 1)
            plt.plot(values)
            plt.xlabel('Trial')
            plt.ylabel('Objective Value')
            plt.title('Optimization History')
            plt.grid(True, alpha=0.3)
            
            # Plot best value progression
            plt.subplot(1, 2, 2)
            best_values = []
            current_best = float('-inf')
            for value in values:
                if value > current_best:
                    current_best = value
                best_values.append(current_best)
            
            plt.plot(best_values)
            plt.xlabel('Trial')
            plt.ylabel('Best Objective Value')
            plt.title('Best Value Progression')
            plt.grid(True, alpha=0.3)
            
            plt.tight_layout()
            
            if save_path:
                plt.savefig(save_path, dpi=300, bbox_inches='tight')
                logger.info(f"Optimization history plot saved to {save_path}")
            
            plt.show()
            
        except Exception as e:
            logger.error(f"Error plotting optimization history: {e}")
    
    def get_parameter_importance(self, target_param: str = None) -> Dict[str, float]:
        """
        Get parameter importance from the study.
        
        Args:
            target_param: Target parameter for importance calculation
            
        Returns:
            Dictionary of parameter importances
        """
        if self.study is None:
            logger.error("No study available. Run optimization first.")
            return {}
        
        try:
            if target_param:
                importance = optuna.importance.get_param_importances(
                    self.study, target=lambda t: t.params.get(target_param, 0)
                )
            else:
                importance = optuna.importance.get_param_importances(self.study)
            
            return importance
            
        except Exception as e:
            logger.error(f"Error calculating parameter importance: {e}")
            return {}
