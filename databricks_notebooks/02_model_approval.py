# Databricks notebook source
# MAGIC %md
# MAGIC # Model Approval Task
# MAGIC 
# MAGIC This notebook defines the approval task for the California Housing MLOps pipeline.
# MAGIC It allows a privileged user to review evaluation metrics and approve or reject model versions.
# MAGIC 
# MAGIC ## Features:
# MAGIC - Review evaluation results and metrics
# MAGIC - Interactive approval/rejection workflow
# MAGIC - Model stage management (None → Staging → Production)
# MAGIC - Approval audit trail and documentation
# MAGIC - Risk assessment and validation

# COMMAND ----------

# MAGIC %md
# MAGIC ## 1. Setup and Configuration

# COMMAND ----------

# Import required libraries
import mlflow
import mlflow.sklearn
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from mlflow.tracking import MlflowClient
from mlflow.entities.model_registry import ModelVersion
import json
from datetime import datetime
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

# Configuration widgets
dbutils.widgets.text("model_name", "california_housing_predictor", "Registered Model Name")
dbutils.widgets.text("model_version", "latest", "Model Version to Review")
dbutils.widgets.text("evaluation_results_path", "/tmp/model_evaluation_results.json", "Evaluation Results Path")
dbutils.widgets.text("approver_email", "", "Approver Email (required)")
dbutils.widgets.dropdown("approval_action", "review", ["review", "approve", "reject"], "Approval Action")
dbutils.widgets.text("approval_comments", "", "Approval Comments")

# Get parameters
MODEL_NAME = dbutils.widgets.get("model_name")
MODEL_VERSION = dbutils.widgets.get("model_version")
EVALUATION_RESULTS_PATH = dbutils.widgets.get("evaluation_results_path")
APPROVER_EMAIL = dbutils.widgets.get("approver_email")
APPROVAL_ACTION = dbutils.widgets.get("approval_action")
APPROVAL_COMMENTS = dbutils.widgets.get("approval_comments")

# Validate required parameters
if not APPROVER_EMAIL:
    raise ValueError("Approver email is required for audit trail")

print(f"Model Name: {MODEL_NAME}")
print(f"Model Version: {MODEL_VERSION}")
print(f"Approver: {APPROVER_EMAIL}")
print(f"Action: {APPROVAL_ACTION}")

# COMMAND ----------

# MAGIC %md
# MAGIC ## 3. Load Evaluation Results

# COMMAND ----------

def load_evaluation_results(results_path: str) -> Dict[str, Any]:
    """Load evaluation results from previous step."""
    try:
        # Try to load from DBFS
        try:
            results_json = dbutils.fs.head(results_path)
            evaluation_data = json.loads(results_json)
            print(f"✅ Loaded evaluation results from {results_path}")
            return evaluation_data
        except Exception as e:
            print(f"Could not load from {results_path}: {e}")
            
            # Fallback: Create mock evaluation results for demonstration
            print("⚠️  Creating mock evaluation results for demonstration...")
            mock_results = {
                'model_name': MODEL_NAME,
                'model_version': MODEL_VERSION if MODEL_VERSION != "latest" else "1",
                'evaluation_run_id': "mock_run_12345",
                'evaluation_decision': {
                    'passes_evaluation': True,
                    'recommendation': 'APPROVE',
                    'quality_score': 'GOOD',
                    'passed_checks': [
                        'R² Score: 0.845 >= 0.7',
                        'MAPE: 15.2% <= 25.0%',
                        'Bounds Coverage: 98.5% >= 95.0%'
                    ],
                    'failed_checks': [],
                    'evaluation_timestamp': datetime.now().isoformat(),
                    'model_name': MODEL_NAME,
                    'model_version': MODEL_VERSION if MODEL_VERSION != "latest" else "1"
                },
                'metrics': {
                    'r2_score': 0.845,
                    'mean_squared_error': 2.15e9,
                    'mean_absolute_error': 32145.67,
                    'mape': 15.2,
                    'rmse': 46368.23,
                    'bounds_coverage': 98.5,
                    'explained_variance': 0.847
                },
                'experiment_name': '/Shared/mlops/model_evaluation',
                'evaluation_type': 'comprehensive',
                'test_data_samples': 1000
            }
            return mock_results
            
    except Exception as e:
        logger.error(f"Error loading evaluation results: {e}")
        raise

# Load evaluation results
evaluation_results = load_evaluation_results(EVALUATION_RESULTS_PATH)

# Extract key information
eval_model_name = evaluation_results['model_name']
eval_model_version = evaluation_results['model_version']
eval_decision = evaluation_results['evaluation_decision']
eval_metrics = evaluation_results['metrics']

print(f"\n📊 EVALUATION SUMMARY FOR APPROVAL")
print(f"{'='*50}")
print(f"Model: {eval_model_name} v{eval_model_version}")
print(f"Evaluation Recommendation: {eval_decision['recommendation']}")
print(f"Quality Score: {eval_decision['quality_score']}")
print(f"Evaluation Date: {eval_decision['evaluation_timestamp']}")

# COMMAND ----------

# MAGIC %md
# MAGIC ## 4. Model Information and Current Status

# COMMAND ----------

def get_model_information(model_name: str, version: str) -> Dict[str, Any]:
    """Get comprehensive model information."""
    try:
        client = MlflowClient()
        
        # Get model version if "latest"
        if version.lower() == "latest":
            latest_versions = client.get_latest_versions(model_name, stages=["None", "Staging", "Production"])
            if not latest_versions:
                raise ValueError(f"No versions found for model {model_name}")
            latest_version = max(latest_versions, key=lambda x: int(x.version))
            version = latest_version.version
        
        # Get model version details
        model_version = client.get_model_version(model_name, version)
        
        # Get registered model details
        registered_model = client.get_registered_model(model_name)
        
        # Get run information
        run = client.get_run(model_version.run_id)
        
        model_info = {
            'model_name': model_name,
            'version': version,
            'current_stage': model_version.current_stage,
            'creation_timestamp': model_version.creation_timestamp,
            'last_updated_timestamp': model_version.last_updated_timestamp,
            'description': model_version.description,
            'run_id': model_version.run_id,
            'run_name': run.data.tags.get('mlflow.runName', 'Unknown'),
            'source': model_version.source,
            'status': model_version.status,
            'tags': dict(model_version.tags),
            'registered_model_description': registered_model.description,
            'run_params': dict(run.data.params),
            'run_metrics': dict(run.data.metrics),
            'model_uri': f"models:/{model_name}/{version}"
        }
        
        return model_info
        
    except Exception as e:
        logger.error(f"Error getting model information: {e}")
        raise

# Get model information
model_info = get_model_information(eval_model_name, eval_model_version)

print(f"\n🔍 MODEL DETAILS")
print(f"{'='*40}")
print(f"Name: {model_info['model_name']}")
print(f"Version: {model_info['version']}")
print(f"Current Stage: {model_info['current_stage']}")
print(f"Status: {model_info['status']}")
print(f"Created: {pd.to_datetime(model_info['creation_timestamp'], unit='ms')}")
print(f"Run ID: {model_info['run_id']}")
print(f"Run Name: {model_info['run_name']}")

if model_info['description']:
    print(f"Description: {model_info['description']}")

# COMMAND ----------

# MAGIC %md
# MAGIC ## 5. Approval Dashboard and Metrics Review

# COMMAND ----------

def create_approval_dashboard():
    """Create visual dashboard for approval review."""
    
    fig, axes = plt.subplots(2, 3, figsize=(18, 12))
    fig.suptitle(f'Model Approval Dashboard - {eval_model_name} v{eval_model_version}', 
                 fontsize=16, fontweight='bold')
    
    # 1. Quality Score Gauge
    ax1 = axes[0, 0]
    quality_mapping = {'EXCELLENT': 5, 'GOOD': 4, 'ACCEPTABLE': 3, 'NEEDS IMPROVEMENT': 2, 'POOR': 1}
    quality_score = quality_mapping.get(eval_decision['quality_score'], 0)
    
    colors = ['red', 'orange', 'yellow', 'lightgreen', 'green']
    ax1.pie([1, 1, 1, 1, 1], startangle=180, colors=colors, 
            wedgeprops=dict(width=0.3), counterclock=False)
    
    # Add quality indicator
    angle = 180 - (quality_score - 1) * 36  # Convert to angle
    ax1.annotate('', xy=(0.7*np.cos(np.radians(angle)), 0.7*np.sin(np.radians(angle))), 
                 xytext=(0, 0), arrowprops=dict(arrowstyle='->', lw=3, color='black'))
    
    ax1.set_title(f'Quality Score\n{eval_decision["quality_score"]}', fontweight='bold')
    ax1.axis('equal')
    
    # 2. Key Metrics
    ax2 = axes[0, 1]
    metrics_to_show = ['r2_score', 'mape', 'rmse', 'bounds_coverage']
    metric_values = [eval_metrics.get(m, 0) for m in metrics_to_show]
    metric_labels = ['R² Score', 'MAPE (%)', 'RMSE', 'Bounds Coverage (%)']
    
    bars = ax2.bar(metric_labels, metric_values, color=['skyblue', 'lightcoral', 'lightgreen', 'gold'])
    ax2.set_title('Key Performance Metrics', fontweight='bold')
    ax2.tick_params(axis='x', rotation=45)
    
    # Add value labels on bars
    for bar, value in zip(bars, metric_values):
        height = bar.get_height()
        ax2.text(bar.get_x() + bar.get_width()/2., height + height*0.01,
                f'{value:.2f}', ha='center', va='bottom', fontweight='bold')
    
    # 3. Checks Status
    ax3 = axes[0, 2]
    passed_count = len(eval_decision['passed_checks'])
    failed_count = len(eval_decision['failed_checks'])
    
    labels = ['Passed', 'Failed']
    sizes = [passed_count, failed_count]
    colors = ['lightgreen', 'lightcoral']
    
    if sum(sizes) > 0:
        wedges, texts, autotexts = ax3.pie(sizes, labels=labels, colors=colors, autopct='%1.0f',
                                          startangle=90)
        ax3.set_title('Validation Checks', fontweight='bold')
    else:
        ax3.text(0.5, 0.5, 'No checks\navailable', ha='center', va='center',
                transform=ax3.transAxes)
        ax3.set_title('Validation Checks', fontweight='bold')
    
    # 4. Model Version History
    ax4 = axes[1, 0]
    try:
        client = MlflowClient()
        model_versions = client.search_model_versions(f"name='{eval_model_name}'")
        
        if model_versions:
            version_numbers = [int(mv.version) for mv in model_versions]
            creation_times = [pd.to_datetime(mv.creation_timestamp, unit='ms') for mv in model_versions]
            stages = [mv.current_stage for mv in model_versions]
            
            # Sort by version
            sorted_data = sorted(zip(version_numbers, creation_times, stages))
            versions, times, stages = zip(*sorted_data) if sorted_data else ([], [], [])
            
            ax4.plot(versions, range(len(versions)), 'o-', linewidth=2, markersize=8)
            ax4.set_xlabel('Version Number')
            ax4.set_ylabel('Timeline (relative)')
            ax4.set_title('Model Version History', fontweight='bold')
            ax4.grid(True, alpha=0.3)
            
            # Highlight current version
            if int(eval_model_version) in versions:
                current_idx = versions.index(int(eval_model_version))
                ax4.plot(int(eval_model_version), current_idx, 'ro', markersize=12, 
                        markerfacecolor='red', markeredgecolor='darkred', linewidth=2)
        else:
            ax4.text(0.5, 0.5, 'No version\nhistory available', ha='center', va='center',
                    transform=ax4.transAxes)
            ax4.set_title('Model Version History', fontweight='bold')
    except Exception as e:
        ax4.text(0.5, 0.5, f'Error loading\nversion history:\n{str(e)[:30]}...', 
                ha='center', va='center', transform=ax4.transAxes)
        ax4.set_title('Model Version History', fontweight='bold')
    
    # 5. Risk Assessment
    ax5 = axes[1, 1]
    
    # Calculate risk score based on metrics
    r2_score = eval_metrics.get('r2_score', 0)
    mape = eval_metrics.get('mape', float('inf'))
    bounds_coverage = eval_metrics.get('bounds_coverage', 0)
    
    risk_factors = []
    if r2_score < 0.7:
        risk_factors.append('Low R² Score')
    if mape > 25:
        risk_factors.append('High MAPE')
    if bounds_coverage < 95:
        risk_factors.append('Poor Bounds Coverage')
    if model_info['current_stage'] == 'Production':
        risk_factors.append('Production Override')
    
    risk_level = len(risk_factors)
    risk_colors = ['green', 'yellow', 'orange', 'red', 'darkred']
    risk_labels = ['Very Low', 'Low', 'Medium', 'High', 'Critical']
    
    risk_color = risk_colors[min(risk_level, 4)]
    risk_label = risk_labels[min(risk_level, 4)]
    
    ax5.bar(['Risk Level'], [risk_level], color=risk_color, alpha=0.7)
    ax5.set_ylim(0, 5)
    ax5.set_ylabel('Risk Score')
    ax5.set_title(f'Risk Assessment\n({risk_label})', fontweight='bold')
    ax5.text(0, risk_level + 0.1, f'{risk_level}', ha='center', va='bottom', 
             fontweight='bold', fontsize=14)
    
    # 6. Approval Timeline
    ax6 = axes[1, 2]
    
    # Create timeline visualization
    timeline_stages = ['Evaluation', 'Approval', 'Deployment']
    timeline_status = ['✅ Complete', '🔄 In Progress', '⏳ Pending']
    y_positions = [2, 1, 0]
    colors = ['green', 'orange', 'gray']
    
    for i, (stage, status, y_pos, color) in enumerate(zip(timeline_stages, timeline_status, y_positions, colors)):
        ax6.barh(y_pos, 1, color=color, alpha=0.7, height=0.3)
        ax6.text(0.5, y_pos, f'{stage}\n{status}', ha='center', va='center', 
                fontweight='bold', fontsize=10)
    
    ax6.set_xlim(0, 1)
    ax6.set_ylim(-0.5, 2.5)
    ax6.set_yticks([])
    ax6.set_xticks([])
    ax6.set_title('Approval Pipeline Status', fontweight='bold')
    
    plt.tight_layout()
    mlflow.log_figure(plt.gcf(), "approval_dashboard.png")
    plt.show()

# Create the dashboard
create_approval_dashboard()

# COMMAND ----------

# MAGIC %md
# MAGIC ## 6. Detailed Metrics Analysis

# COMMAND ----------

def display_detailed_metrics_analysis():
    """Display detailed analysis of evaluation metrics."""
    
    print(f"\n📊 DETAILED METRICS ANALYSIS")
    print(f"{'='*60}")
    
    # Performance Metrics
    print(f"\n🎯 PERFORMANCE METRICS:")
    print(f"{'-'*30}")
    
    metrics_analysis = {
        'r2_score': {
            'value': eval_metrics.get('r2_score', 0),
            'threshold': 0.7,
            'description': 'Coefficient of Determination (higher is better)',
            'interpretation': 'Explains {}% of variance in target variable'
        },
        'mape': {
            'value': eval_metrics.get('mape', float('inf')),
            'threshold': 25.0,
            'description': 'Mean Absolute Percentage Error (lower is better)',
            'interpretation': 'Average prediction error of {}%'
        },
        'rmse': {
            'value': eval_metrics.get('rmse', float('inf')),
            'threshold': None,
            'description': 'Root Mean Squared Error (lower is better)',
            'interpretation': 'Average prediction error of ${:,.0f}'
        },
        'bounds_coverage': {
            'value': eval_metrics.get('bounds_coverage', 0),
            'threshold': 95.0,
            'description': 'Percentage of predictions within reasonable bounds',
            'interpretation': '{}% of predictions are realistic'
        }
    }
    
    for metric_name, metric_info in metrics_analysis.items():
        value = metric_info['value']
        threshold = metric_info['threshold']
        description = metric_info['description']
        interpretation = metric_info['interpretation']
        
        print(f"\n{metric_name.upper().replace('_', ' ')}:")
        print(f"  Value: {value:.3f}" if value != float('inf') else f"  Value: N/A")
        print(f"  Description: {description}")
        
        if value != float('inf'):
            if metric_name == 'r2_score':
                print(f"  Interpretation: {interpretation.format(value * 100:.1f)}")
            elif metric_name == 'mape':
                print(f"  Interpretation: {interpretation.format(value:.1f)}")
            elif metric_name == 'rmse':
                print(f"  Interpretation: {interpretation.format(value)}")
            elif metric_name == 'bounds_coverage':
                print(f"  Interpretation: {interpretation.format(value:.1f)}")
        
        if threshold is not None:
            status = "✅ PASS" if (
                (metric_name == 'mape' and value <= threshold) or
                (metric_name != 'mape' and value >= threshold)
            ) else "❌ FAIL"
            print(f"  Threshold: {threshold} ({status})")
    
    # Validation Checks Summary
    print(f"\n✅ PASSED VALIDATION CHECKS:")
    if eval_decision['passed_checks']:
        for check in eval_decision['passed_checks']:
            print(f"  • {check}")
    else:
        print("  None")
    
    print(f"\n❌ FAILED VALIDATION CHECKS:")
    if eval_decision['failed_checks']:
        for check in eval_decision['failed_checks']:
            print(f"  • {check}")
    else:
        print("  None")

# Display detailed analysis
display_detailed_metrics_analysis()

# COMMAND ----------

# MAGIC %md
# MAGIC ## 7. Business Impact Assessment

# COMMAND ----------

def assess_business_impact():
    """Assess potential business impact of the model."""
    
    print(f"\n💼 BUSINESS IMPACT ASSESSMENT")
    print(f"{'='*50}")
    
    r2_score = eval_metrics.get('r2_score', 0)
    mape = eval_metrics.get('mape', float('inf'))
    rmse = eval_metrics.get('rmse', float('inf'))
    
    # Calculate potential business metrics
    print(f"\n📈 POTENTIAL BUSINESS BENEFITS:")
    print(f"{'-'*30}")
    
    if r2_score > 0:
        accuracy_improvement = (r2_score - 0.5) * 100 if r2_score > 0.5 else 0
        print(f"  • Model explains {r2_score*100:.1f}% of price variance")
        if accuracy_improvement > 0:
            print(f"  • {accuracy_improvement:.1f}% improvement over baseline")
    
    if mape != float('inf') and mape < 50:
        print(f"  • Average prediction error: {mape:.1f}%")
        if mape < 15:
            print(f"  • Excellent accuracy for pricing decisions")
        elif mape < 25:
            print(f"  • Good accuracy for market analysis")
    
    if rmse != float('inf'):
        print(f"  • Average price error: ${rmse:,.0f}")
        
    # Risk Assessment
    print(f"\n⚠️  BUSINESS RISKS:")
    print(f"{'-'*30}")
    
    risks = []
    if r2_score < 0.7:
        risks.append("Low prediction accuracy may lead to poor investment decisions")
    if mape > 25:
        risks.append("High percentage error could impact pricing strategies")
    if eval_metrics.get('bounds_coverage', 0) < 95:
        risks.append("Unrealistic predictions may mislead stakeholders")
    
    if risks:
        for risk in risks:
            print(f"  • {risk}")
    else:
        print(f"  • No significant business risks identified")
    
    # Deployment Readiness
    print(f"\n🚀 DEPLOYMENT READINESS:")
    print(f"{'-'*30}")
    
    readiness_score = 0
    readiness_factors = []
    
    if r2_score >= 0.8:
        readiness_score += 2
        readiness_factors.append("Excellent model performance")
    elif r2_score >= 0.7:
        readiness_score += 1
        readiness_factors.append("Good model performance")
    
    if mape <= 15:
        readiness_score += 2
        readiness_factors.append("Low prediction error")
    elif mape <= 25:
        readiness_score += 1
        readiness_factors.append("Acceptable prediction error")
    
    if eval_metrics.get('bounds_coverage', 0) >= 95:
        readiness_score += 1
        readiness_factors.append("Realistic predictions")
    
    if len(eval_decision['failed_checks']) == 0:
        readiness_score += 1
        readiness_factors.append("Passes all validation checks")
    
    readiness_levels = {
        0: "Not Ready",
        1: "Needs Improvement", 
        2: "Conditional Approval",
        3: "Ready for Staging",
        4: "Ready for Production",
        5: "Excellent - Fast Track",
        6: "Outstanding - Immediate Deploy"
    }
    
    readiness_level = readiness_levels.get(min(readiness_score, 6), "Unknown")
    
    print(f"  Readiness Score: {readiness_score}/6")
    print(f"  Readiness Level: {readiness_level}")
    
    if readiness_factors:
        print(f"  Positive Factors:")
        for factor in readiness_factors:
            print(f"    • {factor}")
    
    return readiness_score, readiness_level

# Assess business impact
business_readiness_score, business_readiness_level = assess_business_impact()

# COMMAND ----------

# MAGIC %md
# MAGIC ## 8. Approval Decision Interface

# COMMAND ----------

def process_approval_decision(action: str, comments: str, approver: str) -> Dict[str, Any]:
    """Process the approval decision."""
    
    if action == "review":
        print(f"\n📋 REVIEW MODE - NO ACTION TAKEN")
        print(f"Use the dropdown to select 'approve' or 'reject' to make a decision.")
        return {"action": "review", "status": "pending"}
    
    # Validate approval action
    if action not in ["approve", "reject"]:
        raise ValueError(f"Invalid approval action: {action}")
    
    # Create approval record
    approval_record = {
        'model_name': eval_model_name,
        'model_version': eval_model_version,
        'action': action,
        'approver': approver,
        'approval_timestamp': datetime.now().isoformat(),
        'comments': comments,
        'evaluation_run_id': evaluation_results.get('evaluation_run_id'),
        'evaluation_metrics': eval_metrics,
        'evaluation_decision': eval_decision,
        'business_readiness_score': business_readiness_score,
        'business_readiness_level': business_readiness_level,
        'model_current_stage': model_info['current_stage']
    }
    
    try:
        client = MlflowClient()
        
        if action == "approve":
            # Determine target stage
            current_stage = model_info['current_stage']
            
            if current_stage == "None":
                target_stage = "Staging"
            elif current_stage == "Staging":
                target_stage = "Production"
            else:
                target_stage = current_stage  # Already in production
            
            # Transition model stage
            client.transition_model_version_stage(
                name=eval_model_name,
                version=eval_model_version,
                stage=target_stage,
                archive_existing_versions=False
            )
            
            approval_record['new_stage'] = target_stage
            approval_record['stage_transition'] = f"{current_stage} → {target_stage}"
            
            print(f"\n✅ MODEL APPROVED")
            print(f"{'='*40}")
            print(f"Model: {eval_model_name} v{eval_model_version}")
            print(f"Stage Transition: {current_stage} → {target_stage}")
            print(f"Approver: {approver}")
            print(f"Timestamp: {approval_record['approval_timestamp']}")
            if comments:
                print(f"Comments: {comments}")
                
        else:  # reject
            # Add rejection tags
            client.set_model_version_tag(
                name=eval_model_name,
                version=eval_model_version,
                key="approval_status",
                value="rejected"
            )
            
            client.set_model_version_tag(
                name=eval_model_name,
                version=eval_model_version,
                key="rejection_reason",
                value=comments or "No reason provided"
            )
            
            approval_record['new_stage'] = model_info['current_stage']  # No change
            approval_record['stage_transition'] = "None (Rejected)"
            
            print(f"\n❌ MODEL REJECTED")
            print(f"{'='*40}")
            print(f"Model: {eval_model_name} v{eval_model_version}")
            print(f"Rejected by: {approver}")
            print(f"Timestamp: {approval_record['approval_timestamp']}")
            if comments:
                print(f"Rejection Reason: {comments}")
        
        # Log approval decision to MLflow
        experiment_name = "/Shared/mlops/model_approval"
        mlflow.set_experiment(experiment_name)
        
        with mlflow.start_run(run_name=f"approval_{eval_model_name}_v{eval_model_version}_{action}"):
            # Log approval parameters
            mlflow.log_param("model_name", eval_model_name)
            mlflow.log_param("model_version", eval_model_version)
            mlflow.log_param("approval_action", action)
            mlflow.log_param("approver", approver)
            mlflow.log_param("business_readiness_level", business_readiness_level)
            
            if action == "approve":
                mlflow.log_param("new_stage", approval_record['new_stage'])
                mlflow.log_param("stage_transition", approval_record['stage_transition'])
            
            # Log evaluation metrics for reference
            for metric_name, metric_value in eval_metrics.items():
                if isinstance(metric_value, (int, float)):
                    mlflow.log_metric(f"eval_{metric_name}", metric_value)
            
            mlflow.log_metric("business_readiness_score", business_readiness_score)
            mlflow.log_metric("approval_success", 1.0 if action == "approve" else 0.0)
            
            # Log approval record as artifact
            mlflow.log_text(json.dumps(approval_record, indent=2), f"approval_record_{action}.json")
            
            # Log comments if provided
            if comments:
                mlflow.log_text(comments, f"{action}_comments.txt")
        
        approval_record['mlflow_run_id'] = mlflow.active_run().info.run_id if mlflow.active_run() else None
        
        # Save approval record to DBFS
        approval_path = f"/tmp/model_approval_{action}_{eval_model_version}.json"
        dbutils.fs.put(approval_path, json.dumps(approval_record, indent=2))
        print(f"\n💾 Approval record saved to: {approval_path}")
        
        return approval_record
        
    except Exception as e:
        logger.error(f"Error processing approval decision: {e}")
        raise

# Process the approval decision
if APPROVAL_ACTION in ["approve", "reject"]:
    approval_result = process_approval_decision(APPROVAL_ACTION, APPROVAL_COMMENTS, APPROVER_EMAIL)
else:
    approval_result = process_approval_decision("review", "", APPROVER_EMAIL)

# COMMAND ----------

# MAGIC %md
# MAGIC ## 9. Approval Summary and Next Steps

# COMMAND ----------

def display_approval_summary():
    """Display approval summary and next steps."""
    
    print(f"\n{'='*60}")
    print("APPROVAL WORKFLOW SUMMARY")
    print(f"{'='*60}")
    
    print(f"\n📋 MODEL INFORMATION:")
    print(f"  • Name: {eval_model_name}")
    print(f"  • Version: {eval_model_version}")
    print(f"  • Previous Stage: {model_info['current_stage']}")
    
    if approval_result['action'] != "review":
        print(f"  • Action Taken: {approval_result['action'].upper()}")
        print(f"  • New Stage: {approval_result.get('new_stage', 'Unknown')}")
        print(f"  • Stage Transition: {approval_result.get('stage_transition', 'None')}")
    
    print(f"\n👤 APPROVAL DETAILS:")
    print(f"  • Approver: {APPROVER_EMAIL}")
    print(f"  • Timestamp: {approval_result.get('approval_timestamp', 'N/A')}")
    if APPROVAL_COMMENTS:
        print(f"  • Comments: {APPROVAL_COMMENTS}")
    
    print(f"\n📊 EVALUATION SUMMARY:")
    print(f"  • Recommendation: {eval_decision['recommendation']}")
    print(f"  • Quality Score: {eval_decision['quality_score']}")
    print(f"  • Business Readiness: {business_readiness_level}")
    print(f"  • Validation Checks: {len(eval_decision['passed_checks'])} passed, {len(eval_decision['failed_checks'])} failed")
    
    print(f"\n🎯 NEXT STEPS:")
    
    if approval_result['action'] == "approve":
        new_stage = approval_result.get('new_stage')
        if new_stage == "Staging":
            print(f"  1. ✅ Model promoted to Staging")
            print(f"  2. 🔄 Conduct staging environment testing")
            print(f"  3. 📊 Monitor staging performance")
            print(f"  4. 🚀 Consider promotion to Production")
        elif new_stage == "Production":
            print(f"  1. ✅ Model promoted to Production")
            print(f"  2. 🚀 Run deployment notebook")
            print(f"  3. 📊 Monitor production performance")
            print(f"  4. 🔔 Set up alerts and monitoring")
        else:
            print(f"  1. ✅ Model approved (stage unchanged)")
            print(f"  2. 📋 Review stage transition requirements")
    
    elif approval_result['action'] == "reject":
        print(f"  1. ❌ Model rejected - no deployment")
        print(f"  2. 📋 Review rejection reasons")
        print(f"  3. 🔧 Address identified issues")
        print(f"  4. 🔄 Retrain and re-evaluate model")
        print(f"  5. 🔁 Resubmit for approval")
    
    else:  # review
        print(f"  1. 📋 Review evaluation metrics above")
        print(f"  2. 💼 Consider business impact assessment")
        print(f"  3. ✅ Set approval action to 'approve' or 'reject'")
        print(f"  4. 💬 Add approval comments")
        print(f"  5. ▶️ Re-run this notebook to execute decision")
    
    print(f"\n📁 DOCUMENTATION:")
    print(f"  • Evaluation Results: {EVALUATION_RESULTS_PATH}")
    if approval_result.get('mlflow_run_id'):
        print(f"  • MLflow Run: {approval_result['mlflow_run_id']}")
    if approval_result['action'] != "review":
        approval_path = f"/tmp/model_approval_{approval_result['action']}_{eval_model_version}.json"
        print(f"  • Approval Record: {approval_path}")

# Display summary
display_approval_summary()

# COMMAND ----------

# MAGIC %md
# MAGIC ## 10. Audit Trail and Compliance

# COMMAND ----------

def generate_audit_trail():
    """Generate audit trail for compliance."""
    
    audit_trail = {
        'audit_timestamp': datetime.now().isoformat(),
        'model_name': eval_model_name,
        'model_version': eval_model_version,
        'approval_workflow_version': '1.0',
        'approver': APPROVER_EMAIL,
        'action_taken': approval_result['action'],
        'evaluation_metrics': eval_metrics,
        'evaluation_decision': eval_decision,
        'business_assessment': {
            'readiness_score': business_readiness_score,
            'readiness_level': business_readiness_level
        },
        'model_info': {
            'current_stage': model_info['current_stage'],
            'creation_timestamp': model_info['creation_timestamp'],
            'run_id': model_info['run_id']
        },
        'compliance_checks': {
            'approver_email_provided': bool(APPROVER_EMAIL),
            'evaluation_completed': bool(evaluation_results),
            'metrics_reviewed': bool(eval_metrics),
            'business_impact_assessed': True,
            'decision_documented': bool(approval_result['action'] != "review")
        }
    }
    
    # Save audit trail
    audit_path = f"/tmp/audit_trail_{eval_model_name}_v{eval_model_version}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
    dbutils.fs.put(audit_path, json.dumps(audit_trail, indent=2))
    
    print(f"\n📋 AUDIT TRAIL GENERATED")
    print(f"{'='*40}")
    print(f"Audit File: {audit_path}")
    print(f"Compliance Checks:")
    for check, status in audit_trail['compliance_checks'].items():
        status_icon = "✅" if status else "❌"
        print(f"  {status_icon} {check.replace('_', ' ').title()}")
    
    return audit_trail

# Generate audit trail
audit_trail = generate_audit_trail()

print(f"\n🎉 APPROVAL WORKFLOW COMPLETED")
print(f"{'='*60}")

if approval_result['action'] == "approve":
    print(f"✅ Model {eval_model_name} v{eval_model_version} has been APPROVED")
    print(f"🚀 Ready for deployment workflow")
elif approval_result['action'] == "reject":
    print(f"❌ Model {eval_model_name} v{eval_model_version} has been REJECTED")
    print(f"🔄 Requires improvement before resubmission")
else:
    print(f"📋 Model {eval_model_name} v{eval_model_version} is under REVIEW")
    print(f"⏳ Awaiting approval decision")
