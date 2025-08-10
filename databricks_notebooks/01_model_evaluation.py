# Databricks notebook source
# MAGIC %md
# MAGIC # Model Evaluation Task
# MAGIC 
# MAGIC This notebook defines the evaluation task for the California Housing MLOps pipeline.
# MAGIC It uses `mlflow.evaluate` to produce comprehensive validation metrics for model versions.
# MAGIC 
# MAGIC ## Features:
# MAGIC - Automated model evaluation using mlflow.evaluate
# MAGIC - Custom metrics and visualizations
# MAGIC - Evaluation results logging
# MAGIC - Model performance comparison

# COMMAND ----------

# MAGIC %md
# MAGIC ## 1. Setup and Configuration

# COMMAND ----------

# Import required libraries
import mlflow
import mlflow.sklearn
import mlflow.genai.evaluate
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from mlflow.tracking import MlflowClient
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from typing import Dict, Any, List, Optional
import logging
import warnings
warnings.filterwarnings('ignore')

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# COMMAND ----------

# MAGIC %md
# MAGIC ## 2. Configuration Parameters

# COMMAND ----------

# Configuration
dbutils.widgets.text("model_name", "california_housing_predictor", "Registered Model Name")
dbutils.widgets.text("model_version", "latest", "Model Version (or 'latest')")
dbutils.widgets.text("evaluation_experiment", "/Shared/mlops/model_evaluation", "Evaluation Experiment Path")
dbutils.widgets.text("data_path", "/databricks-datasets/learning-spark-v2/sf-airbnb/sf-airbnb-clean.parquet", "Test Data Path")
dbutils.widgets.dropdown("evaluation_type", "comprehensive", ["quick", "comprehensive"], "Evaluation Type")

# Get parameters
MODEL_NAME = dbutils.widgets.get("model_name")
MODEL_VERSION = dbutils.widgets.get("model_version") 
EVALUATION_EXPERIMENT = dbutils.widgets.get("evaluation_experiment")
DATA_PATH = dbutils.widgets.get("data_path")
EVALUATION_TYPE = dbutils.widgets.get("evaluation_type")

print(f"Model Name: {MODEL_NAME}")
print(f"Model Version: {MODEL_VERSION}")
print(f"Evaluation Experiment: {EVALUATION_EXPERIMENT}")
print(f"Evaluation Type: {EVALUATION_TYPE}")

# COMMAND ----------

# MAGIC %md
# MAGIC ## 3. Data Loading and Preparation

# COMMAND ----------

def load_test_data():
    """Load test data for evaluation."""
    try:
        # For California Housing dataset, try to load from DBFS or local source
        # This is a placeholder - adapt based on your actual data location
        
        # Option 1: Load from DBFS if uploaded
        try:
            df = spark.read.csv("/databricks-datasets/california-housing/cal_housing.data", 
                              header=False, inferSchema=True)
            # California Housing column names
            columns = ['longitude', 'latitude', 'housing_median_age', 'total_rooms',
                      'total_bedrooms', 'population', 'households', 'median_income', 'median_house_value']
            
            for i, col_name in enumerate(columns):
                df = df.withColumnRenamed(f"_c{i}", col_name)
            
            # Convert to Pandas for sklearn compatibility
            df_pandas = df.toPandas()
            
            # Prepare features and target
            X = df_pandas.drop('median_house_value', axis=1)
            y = df_pandas['median_house_value']
            
            print(f"Loaded test data: {X.shape[0]} samples, {X.shape[1]} features")
            return X, y
            
        except Exception as e:
            print(f"Could not load from DBFS datasets: {e}")
            
            # Option 2: Generate synthetic test data for demonstration
            print("Generating synthetic test data...")
            np.random.seed(42)
            n_samples = 1000
            
            X = pd.DataFrame({
                'longitude': np.random.uniform(-124, -114, n_samples),
                'latitude': np.random.uniform(32, 42, n_samples),
                'housing_median_age': np.random.uniform(1, 52, n_samples),
                'total_rooms': np.random.uniform(1, 10000, n_samples),
                'total_bedrooms': np.random.uniform(1, 2000, n_samples),
                'population': np.random.uniform(1, 8000, n_samples),
                'households': np.random.uniform(1, 2000, n_samples),
                'median_income': np.random.uniform(0.5, 15, n_samples)
            })
            
            # Generate target with some relationship to features
            y = (X['median_income'] * 50000 + 
                 X['total_rooms'] * 10 + 
                 np.random.normal(0, 50000, n_samples))
            y = np.abs(y)  # Ensure positive prices
            
            print(f"Generated synthetic test data: {X.shape[0]} samples, {X.shape[1]} features")
            return X, y
            
    except Exception as e:
        logger.error(f"Error loading test data: {e}")
        raise

# Load test data
X_test, y_test = load_test_data()

# Display data info
print("\nTest Data Overview:")
print(f"Features shape: {X_test.shape}")
print(f"Target shape: {y_test.shape}")
print(f"Features: {list(X_test.columns)}")
print(f"Target range: {y_test.min():.2f} - {y_test.max():.2f}")

# COMMAND ----------

# MAGIC %md
# MAGIC ## 4. Model Loading and Basic Evaluation

# COMMAND ----------

def load_model_for_evaluation(model_name: str, version: str) -> Any:
    """Load model from MLflow Model Registry."""
    try:
        client = MlflowClient()
        
        # Get model version
        if version.lower() == "latest":
            # Get latest version
            latest_versions = client.get_latest_versions(model_name, stages=["None", "Staging", "Production"])
            if not latest_versions:
                raise ValueError(f"No versions found for model {model_name}")
            
            # Sort by version number and get the latest
            latest_version = max(latest_versions, key=lambda x: int(x.version))
            version = latest_version.version
            print(f"Using latest version: {version}")
        
        # Load model
        model_uri = f"models:/{model_name}/{version}"
        model = mlflow.sklearn.load_model(model_uri)
        
        # Get model version info
        model_version = client.get_model_version(model_name, version)
        
        print(f"Loaded model: {model_name} version {version}")
        print(f"Model stage: {model_version.current_stage}")
        print(f"Model description: {model_version.description}")
        
        return model, model_version
        
    except Exception as e:
        logger.error(f"Error loading model: {e}")
        raise

# Load model
model, model_version_info = load_model_for_evaluation(MODEL_NAME, MODEL_VERSION)

# COMMAND ----------

# MAGIC %md
# MAGIC ## 5. Custom Evaluation Metrics

# COMMAND ----------

def custom_metrics():
    """Define custom metrics for mlflow.evaluate."""
    
    def mean_absolute_percentage_error(y_true, y_pred):
        """Calculate Mean Absolute Percentage Error."""
        return np.mean(np.abs((y_true - y_pred) / y_true)) * 100
    
    def root_mean_squared_error(y_true, y_pred):
        """Calculate Root Mean Squared Error."""
        return np.sqrt(mean_squared_error(y_true, y_pred))
    
    def explained_variance_score(y_true, y_pred):
        """Calculate Explained Variance Score."""
        from sklearn.metrics import explained_variance_score as evs
        return evs(y_true, y_pred)
    
    def prediction_bounds_coverage(y_true, y_pred):
        """Calculate percentage of predictions within reasonable bounds."""
        # Define reasonable bounds for California housing prices
        min_price, max_price = 50000, 2000000
        within_bounds = np.sum((y_pred >= min_price) & (y_pred <= max_price))
        return (within_bounds / len(y_pred)) * 100
    
    return {
        "mape": mean_absolute_percentage_error,
        "rmse": root_mean_squared_error,
        "explained_variance": explained_variance_score,
        "bounds_coverage": prediction_bounds_coverage
    }

# COMMAND ----------

# MAGIC %md
# MAGIC ## 6. MLflow Evaluate Execution

# COMMAND ----------

def run_mlflow_evaluation():
    """Run comprehensive model evaluation using mlflow.evaluate."""
    
    # Set up evaluation experiment
    mlflow.set_experiment(EVALUATION_EXPERIMENT)
    
    with mlflow.start_run(run_name=f"evaluation_{MODEL_NAME}_v{model_version_info.version}") as run:
        
        # Log evaluation parameters
        mlflow.log_param("model_name", MODEL_NAME)
        mlflow.log_param("model_version", model_version_info.version)
        mlflow.log_param("model_stage", model_version_info.current_stage)
        mlflow.log_param("evaluation_type", EVALUATION_TYPE)
        mlflow.log_param("test_samples", len(X_test))
        
        # Create evaluation dataset
        eval_data = X_test.copy()
        eval_data['target'] = y_test
        
        print("Starting MLflow evaluation...")
        
        # Configure evaluation based on type
        if EVALUATION_TYPE == "comprehensive":
            # Comprehensive evaluation with all features
            evaluator_config = {
                "log_model_explainability": True,
                "explainability_algorithm": "shap",
                "explainability_nsamples": 100,
                "log_model_signatures": True,
                "log_input_examples": True,
                "log_model_inferences": True
            }
        else:
            # Quick evaluation
            evaluator_config = {
                "log_model_explainability": False,
                "log_model_signatures": True,
                "log_input_examples": True
            }
        
        # Run MLflow evaluation
        evaluation_result = mlflow.evaluate(
            model=model,
            data=eval_data,
            targets="target",
            model_type="regressor",
            custom_metrics=custom_metrics(),
            evaluator_config=evaluator_config,
            extra_metrics=[
                "mean_squared_error",
                "mean_absolute_error", 
                "r2_score",
                "max_error"
            ]
        )
        
        # Log additional insights
        predictions = model.predict(X_test)
        
        # Prediction statistics
        mlflow.log_metric("pred_mean", float(np.mean(predictions)))
        mlflow.log_metric("pred_std", float(np.std(predictions)))
        mlflow.log_metric("pred_min", float(np.min(predictions)))
        mlflow.log_metric("pred_max", float(np.max(predictions)))
        
        # Residual analysis
        residuals = y_test - predictions
        mlflow.log_metric("residual_mean", float(np.mean(residuals)))
        mlflow.log_metric("residual_std", float(np.std(residuals)))
        
        # Feature importance (if available)
        if hasattr(model, 'feature_importances_'):
            feature_importance = dict(zip(X_test.columns, model.feature_importances_))
            for feature, importance in feature_importance.items():
                mlflow.log_metric(f"feature_importance_{feature}", float(importance))
        
        # Create and log visualizations
        create_evaluation_plots(y_test, predictions, residuals)
        
        print(f"Evaluation completed. Run ID: {run.info.run_id}")
        
        # Return evaluation results
        return evaluation_result, run.info.run_id

# COMMAND ----------

# MAGIC %md
# MAGIC ## 7. Evaluation Visualizations

# COMMAND ----------

def create_evaluation_plots(y_true, y_pred, residuals):
    """Create evaluation plots and log them to MLflow."""
    
    plt.style.use('default')
    
    # 1. Predictions vs Actual
    plt.figure(figsize=(12, 8))
    
    plt.subplot(2, 2, 1)
    plt.scatter(y_true, y_pred, alpha=0.6, s=20)
    plt.plot([y_true.min(), y_true.max()], [y_true.min(), y_true.max()], 'r--', lw=2)
    plt.xlabel('Actual Values')
    plt.ylabel('Predicted Values')
    plt.title('Predictions vs Actual Values')
    plt.grid(True, alpha=0.3)
    
    # Add R² score
    r2 = r2_score(y_true, y_pred)
    plt.text(0.05, 0.95, f'R² = {r2:.3f}', transform=plt.gca().transAxes, 
             bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
    
    # 2. Residuals plot
    plt.subplot(2, 2, 2)
    plt.scatter(y_pred, residuals, alpha=0.6, s=20)
    plt.axhline(y=0, color='r', linestyle='--')
    plt.xlabel('Predicted Values')
    plt.ylabel('Residuals')
    plt.title('Residuals vs Predicted Values')
    plt.grid(True, alpha=0.3)
    
    # 3. Residuals histogram
    plt.subplot(2, 2, 3)
    plt.hist(residuals, bins=30, alpha=0.7, edgecolor='black')
    plt.xlabel('Residuals')
    plt.ylabel('Frequency')
    plt.title('Distribution of Residuals')
    plt.grid(True, alpha=0.3)
    
    # 4. Q-Q plot for residuals
    plt.subplot(2, 2, 4)
    from scipy import stats
    stats.probplot(residuals, dist="norm", plot=plt)
    plt.title('Q-Q Plot of Residuals')
    plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    mlflow.log_figure(plt.gcf(), "evaluation_plots.png")
    plt.show()
    
    # Error distribution by prediction ranges
    plt.figure(figsize=(10, 6))
    
    # Create prediction bins
    n_bins = 10
    pred_bins = pd.cut(y_pred, bins=n_bins)
    
    # Calculate metrics by bin
    bin_metrics = []
    for bin_label in pred_bins.cat.categories:
        mask = pred_bins == bin_label
        if mask.sum() > 0:
            bin_mae = mean_absolute_error(y_true[mask], y_pred[mask])
            bin_rmse = np.sqrt(mean_squared_error(y_true[mask], y_pred[mask]))
            bin_metrics.append({
                'bin': str(bin_label),
                'count': mask.sum(),
                'mae': bin_mae,
                'rmse': bin_rmse
            })
    
    if bin_metrics:
        metrics_df = pd.DataFrame(bin_metrics)
        
        plt.subplot(1, 2, 1)
        plt.bar(range(len(metrics_df)), metrics_df['mae'])
        plt.xlabel('Prediction Bins')
        plt.ylabel('Mean Absolute Error')
        plt.title('MAE by Prediction Range')
        plt.xticks(range(len(metrics_df)), [f"Bin {i+1}" for i in range(len(metrics_df))], rotation=45)
        
        plt.subplot(1, 2, 2)
        plt.bar(range(len(metrics_df)), metrics_df['rmse'])
        plt.xlabel('Prediction Bins')
        plt.ylabel('Root Mean Squared Error')
        plt.title('RMSE by Prediction Range')
        plt.xticks(range(len(metrics_df)), [f"Bin {i+1}" for i in range(len(metrics_df))], rotation=45)
        
        plt.tight_layout()
        mlflow.log_figure(plt.gcf(), "error_by_prediction_range.png")
        plt.show()

# COMMAND ----------

# MAGIC %md
# MAGIC ## 8. Run Evaluation

# COMMAND ----------

# Execute the evaluation
evaluation_result, evaluation_run_id = run_mlflow_evaluation()

print(f"\n{'='*60}")
print("EVALUATION COMPLETED SUCCESSFULLY")
print(f"{'='*60}")
print(f"Model: {MODEL_NAME} v{model_version_info.version}")
print(f"Evaluation Run ID: {evaluation_run_id}")
print(f"Evaluation Experiment: {EVALUATION_EXPERIMENT}")

# COMMAND ----------

# MAGIC %md
# MAGIC ## 9. Evaluation Summary and Results

# COMMAND ----------

def display_evaluation_summary(eval_result):
    """Display a comprehensive evaluation summary."""
    
    print("EVALUATION SUMMARY")
    print("=" * 50)
    
    # Extract metrics
    metrics = eval_result.metrics
    
    print("\n📊 PERFORMANCE METRICS:")
    print("-" * 30)
    
    # Core regression metrics
    core_metrics = ['mean_squared_error', 'mean_absolute_error', 'r2_score', 'max_error']
    for metric in core_metrics:
        if metric in metrics:
            print(f"{metric.upper().replace('_', ' ')}: {metrics[metric]:.4f}")
    
    # Custom metrics
    custom_metric_names = ['mape', 'rmse', 'explained_variance', 'bounds_coverage']
    print(f"\n🎯 CUSTOM METRICS:")
    print("-" * 30)
    for metric in custom_metric_names:
        if metric in metrics:
            if metric == 'mape':
                print(f"MEAN ABSOLUTE PERCENTAGE ERROR: {metrics[metric]:.2f}%")
            elif metric == 'bounds_coverage':
                print(f"PREDICTIONS WITHIN BOUNDS: {metrics[metric]:.1f}%")
            else:
                print(f"{metric.upper().replace('_', ' ')}: {metrics[metric]:.4f}")
    
    # Model quality assessment
    print(f"\n🏆 MODEL QUALITY ASSESSMENT:")
    print("-" * 30)
    
    r2 = metrics.get('r2_score', 0)
    mape = metrics.get('mape', float('inf'))
    
    if r2 > 0.9:
        quality = "EXCELLENT"
    elif r2 > 0.8:
        quality = "GOOD"
    elif r2 > 0.7:
        quality = "ACCEPTABLE"
    else:
        quality = "NEEDS IMPROVEMENT"
    
    print(f"Overall Quality: {quality}")
    print(f"R² Score: {r2:.3f}")
    
    if mape != float('inf'):
        if mape < 10:
            mape_assessment = "EXCELLENT"
        elif mape < 20:
            mape_assessment = "GOOD"
        elif mape < 30:
            mape_assessment = "ACCEPTABLE"
        else:
            mape_assessment = "POOR"
        print(f"MAPE Assessment: {mape_assessment} ({mape:.1f}%)")
    
    return quality, metrics

# Display the summary
model_quality, eval_metrics = display_evaluation_summary(evaluation_result)

# COMMAND ----------

# MAGIC %md
# MAGIC ## 10. Evaluation Decision and Next Steps

# COMMAND ----------

def make_evaluation_decision(quality: str, metrics: Dict[str, float]) -> Dict[str, Any]:
    """Make evaluation decision based on metrics."""
    
    # Define thresholds
    thresholds = {
        'min_r2_score': 0.7,
        'max_mape': 25.0,
        'min_bounds_coverage': 95.0
    }
    
    # Check each criterion
    r2_score = metrics.get('r2_score', 0)
    mape = metrics.get('mape', float('inf'))
    bounds_coverage = metrics.get('bounds_coverage', 0)
    
    passed_checks = []
    failed_checks = []
    
    # R² Score check
    if r2_score >= thresholds['min_r2_score']:
        passed_checks.append(f"R² Score: {r2_score:.3f} >= {thresholds['min_r2_score']}")
    else:
        failed_checks.append(f"R² Score: {r2_score:.3f} < {thresholds['min_r2_score']}")
    
    # MAPE check
    if mape <= thresholds['max_mape']:
        passed_checks.append(f"MAPE: {mape:.1f}% <= {thresholds['max_mape']}%")
    else:
        failed_checks.append(f"MAPE: {mape:.1f}% > {thresholds['max_mape']}%")
    
    # Bounds coverage check
    if bounds_coverage >= thresholds['min_bounds_coverage']:
        passed_checks.append(f"Bounds Coverage: {bounds_coverage:.1f}% >= {thresholds['min_bounds_coverage']}%")
    else:
        failed_checks.append(f"Bounds Coverage: {bounds_coverage:.1f}% < {thresholds['min_bounds_coverage']}%")
    
    # Make decision
    passes_evaluation = len(failed_checks) == 0
    
    decision = {
        'passes_evaluation': passes_evaluation,
        'recommendation': 'APPROVE' if passes_evaluation else 'REJECT',
        'quality_score': quality,
        'passed_checks': passed_checks,
        'failed_checks': failed_checks,
        'thresholds_used': thresholds,
        'evaluation_timestamp': pd.Timestamp.now().isoformat(),
        'model_name': MODEL_NAME,
        'model_version': model_version_info.version
    }
    
    return decision

# Make evaluation decision
evaluation_decision = make_evaluation_decision(model_quality, eval_metrics)

# Log decision to MLflow
with mlflow.start_run(run_id=evaluation_run_id):
    mlflow.log_param("evaluation_decision", evaluation_decision['recommendation'])
    mlflow.log_param("evaluation_quality", evaluation_decision['quality_score'])
    mlflow.log_metric("passes_evaluation", 1.0 if evaluation_decision['passes_evaluation'] else 0.0)
    
    # Log decision details as JSON
    import json
    mlflow.log_text(json.dumps(evaluation_decision, indent=2), "evaluation_decision.json")

# Display decision
print(f"\n{'='*60}")
print("EVALUATION DECISION")
print(f"{'='*60}")
print(f"Model: {MODEL_NAME} v{model_version_info.version}")
print(f"Recommendation: {evaluation_decision['recommendation']}")
print(f"Quality Score: {evaluation_decision['quality_score']}")

if evaluation_decision['passed_checks']:
    print(f"\n✅ PASSED CHECKS:")
    for check in evaluation_decision['passed_checks']:
        print(f"  • {check}")

if evaluation_decision['failed_checks']:
    print(f"\n❌ FAILED CHECKS:")
    for check in evaluation_decision['failed_checks']:
        print(f"  • {check}")

print(f"\n📋 NEXT STEPS:")
if evaluation_decision['passes_evaluation']:
    print("  1. Model is ready for approval review")
    print("  2. Run the approval notebook for human review")
    print("  3. If approved, proceed to deployment")
else:
    print("  1. Model requires improvement before approval")
    print("  2. Review failed checks and retrain model")
    print("  3. Re-run evaluation after improvements")

print(f"\n📊 Evaluation Run ID: {evaluation_run_id}")
print(f"🔗 View results in MLflow UI: {EVALUATION_EXPERIMENT}")

# COMMAND ----------

# MAGIC %md
# MAGIC ## 11. Export Results

# COMMAND ----------

# Create evaluation summary for next stage
evaluation_summary = {
    'model_name': MODEL_NAME,
    'model_version': model_version_info.version,
    'evaluation_run_id': evaluation_run_id,
    'evaluation_decision': evaluation_decision,
    'metrics': eval_metrics,
    'experiment_name': EVALUATION_EXPERIMENT,
    'evaluation_type': EVALUATION_TYPE,
    'test_data_samples': len(X_test)
}

# Save to DBFS for next notebook
import json
dbutils.fs.put("/tmp/model_evaluation_results.json", 
               json.dumps(evaluation_summary, indent=2, default=str))

print("✅ Evaluation results exported to /tmp/model_evaluation_results.json")
print("🚀 Ready for approval workflow!")
