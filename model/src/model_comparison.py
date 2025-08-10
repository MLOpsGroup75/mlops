"""
Model comparison and selection utilities.
"""

import pandas as pd
import numpy as np
from typing import Dict, Any, List, Tuple, Optional
import logging
import mlflow
import joblib
import os
from pathlib import Path
import matplotlib.pyplot as plt
import seaborn as sns
import mlflow.sklearn
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

from .model_registry import ModelRegistry

logger = logging.getLogger(__name__)


class ModelComparator:
    """Compare and select best models from training results."""

    def __init__(self, tracking_uri: str = "databricks"):
        """
        Initialize model comparator.

        Args:
            tracking_uri: MLflow tracking URI
        """
        self.tracking_uri = tracking_uri
        self.registry = ModelRegistry(tracking_uri)
        self.comparison_results = {}
        self.best_model_info = None

    def compare_training_results(
        self,
        training_results: List[Dict[str, Any]],
        primary_metric: str = "test_r2_score",
        secondary_metrics: List[str] = None,
    ) -> pd.DataFrame:
        """
        Compare results from different model training runs.

        Args:
            training_results: List of training result dictionaries
            primary_metric: Primary metric for ranking models
            secondary_metrics: Additional metrics to include in comparison

        Returns:
            DataFrame with comparison results
        """
        if secondary_metrics is None:
            secondary_metrics = ["test_rmse", "test_mae", "cv_r2_mean", "cv_r2_std"]

        comparison_data = []

        for i, result in enumerate(training_results):
            row = {
                "model_id": f"model_{i}",
                "model_type": result.get("model_type", "Unknown"),
                "hyperparameter_optimization": result.get("best_params", {}) != {},
            }

            # Add test metrics
            test_metrics = result.get("test_metrics", {})
            for metric in [primary_metric] + secondary_metrics:
                if metric in test_metrics:
                    row[metric] = test_metrics[metric]
                elif metric.replace("test_", "") in test_metrics:
                    row[metric] = test_metrics[metric.replace("test_", "")]

            # Add CV metrics
            cv_metrics = result.get("cv_metrics", {})
            for metric in cv_metrics:
                if metric not in row:
                    row[metric] = cv_metrics[metric]

            # Add training time if available
            if "training_time" in result:
                row["training_time"] = result["training_time"]

            # Add model complexity metrics
            if "n_features" in result:
                row["n_features"] = result["n_features"]

            # Add best parameters summary
            best_params = result.get("best_params", {})
            if best_params:
                row["best_params_count"] = len(best_params)
                # Add key parameters
                for key, value in best_params.items():
                    if isinstance(value, (int, float, str, bool)):
                        row[f"param_{key}"] = value

            comparison_data.append(row)

        comparison_df = pd.DataFrame(comparison_data)

        # Sort by primary metric (descending for R², ascending for RMSE/MAE)
        ascending = "rmse" in primary_metric.lower() or "mae" in primary_metric.lower()
        comparison_df = comparison_df.sort_values(primary_metric, ascending=ascending)

        # Add ranking
        comparison_df["rank"] = range(1, len(comparison_df) + 1)

        self.comparison_results = comparison_df

        # Identify best model
        if not comparison_df.empty:
            best_idx = comparison_df.index[0]
            self.best_model_info = {
                "index": best_idx,
                "model_type": comparison_df.loc[best_idx, "model_type"],
                "primary_metric_value": comparison_df.loc[best_idx, primary_metric],
                "rank": 1,
            }

        logger.info(f"Model comparison completed. Best model: {self.best_model_info}")

        return comparison_df

    def create_comparison_visualizations(
        self, save_dir: str = "../../plots/model_comparison"
    ) -> Dict[str, str]:
        """
        Create visualizations for model comparison.

        Args:
            save_dir: Directory to save plots

        Returns:
            Dictionary of plot file paths
        """
        if self.comparison_results.empty:
            logger.error(
                "No comparison results available. Run compare_training_results first."
            )
            return {}

        # Create save directory
        save_path = Path(save_dir)
        save_path.mkdir(parents=True, exist_ok=True)

        plot_paths = {}
        df = self.comparison_results

        # 1. Model performance comparison bar plot
        plt.figure(figsize=(12, 8))

        # Select numeric metrics for plotting
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        metric_cols = [col for col in numeric_cols if "test_" in col or "cv_" in col]
        metric_cols = metric_cols[:4]  # Limit to top 4 metrics

        if metric_cols:
            x = np.arange(len(df))
            width = 0.8 / len(metric_cols)

            for i, metric in enumerate(metric_cols):
                plt.bar(x + i * width, df[metric], width, label=metric, alpha=0.8)

            plt.xlabel("Models")
            plt.ylabel("Metric Values")
            plt.title("Model Performance Comparison")
            plt.xticks(
                x + width * (len(metric_cols) - 1) / 2,
                [f"{row['model_type']}" for _, row in df.iterrows()],
                rotation=45,
            )
            plt.legend()
            plt.grid(True, alpha=0.3)
            plt.tight_layout()

            performance_plot_path = save_path / "model_performance_comparison.png"
            plt.savefig(performance_plot_path, dpi=300, bbox_inches="tight")
            plt.close()
            plot_paths["performance_comparison"] = str(performance_plot_path)

        # 2. Metric correlation heatmap
        if len(metric_cols) > 1:
            plt.figure(figsize=(10, 8))
            correlation_matrix = df[metric_cols].corr()

            sns.heatmap(
                correlation_matrix,
                annot=True,
                cmap="coolwarm",
                center=0,
                square=True,
                linewidths=0.5,
            )
            plt.title("Metric Correlation Heatmap")
            plt.tight_layout()

            correlation_plot_path = save_path / "metric_correlation_heatmap.png"
            plt.savefig(correlation_plot_path, dpi=300, bbox_inches="tight")
            plt.close()
            plot_paths["metric_correlation"] = str(correlation_plot_path)

        # 3. Model ranking visualization
        if "rank" in df.columns:
            plt.figure(figsize=(10, 6))

            # Create a horizontal bar plot of rankings
            y_pos = np.arange(len(df))
            plt.barh(y_pos, df["rank"], alpha=0.7)
            plt.yticks(y_pos, [f"{row['model_type']}" for _, row in df.iterrows()])
            plt.xlabel("Rank (1 = Best)")
            plt.title("Model Ranking")
            plt.gca().invert_yaxis()  # Reverse order to show best at top
            plt.grid(True, alpha=0.3, axis="x")

            # Add performance values as text
            primary_metric = None
            for col in df.columns:
                if "test_r2" in col or "test_rmse" in col:
                    primary_metric = col
                    break

            if primary_metric:
                for i, (_, row) in enumerate(df.iterrows()):
                    plt.text(
                        row["rank"] + 0.1,
                        i,
                        f"{row[primary_metric]:.4f}",
                        va="center",
                        fontsize=10,
                    )

            plt.tight_layout()

            ranking_plot_path = save_path / "model_ranking.png"
            plt.savefig(ranking_plot_path, dpi=300, bbox_inches="tight")
            plt.close()
            plot_paths["ranking"] = str(ranking_plot_path)

        # 4. Training complexity vs performance
        if "training_time" in df.columns or "best_params_count" in df.columns:
            plt.figure(figsize=(10, 6))

            # Use training time or parameter count as complexity measure
            complexity_col = None
            if "training_time" in df.columns:
                complexity_col = "training_time"
                complexity_label = "Training Time (seconds)"
            elif "best_params_count" in df.columns:
                complexity_col = "best_params_count"
                complexity_label = "Number of Tuned Parameters"

            if complexity_col and metric_cols:
                primary_metric = metric_cols[0]

                plt.scatter(
                    df[complexity_col],
                    df[primary_metric],
                    alpha=0.7,
                    s=100,
                    c=df["rank"],
                    cmap="viridis_r",
                )

                # Add model type labels
                for _, row in df.iterrows():
                    plt.annotate(
                        row["model_type"],
                        (row[complexity_col], row[primary_metric]),
                        xytext=(5, 5),
                        textcoords="offset points",
                        fontsize=10,
                        alpha=0.8,
                    )

                plt.xlabel(complexity_label)
                plt.ylabel(primary_metric)
                plt.title("Model Complexity vs Performance")
                plt.colorbar(label="Rank (Lower is Better)")
                plt.grid(True, alpha=0.3)
                plt.tight_layout()

                complexity_plot_path = save_path / "complexity_vs_performance.png"
                plt.savefig(complexity_plot_path, dpi=300, bbox_inches="tight")
                plt.close()
                plot_paths["complexity_vs_performance"] = str(complexity_plot_path)

        logger.info(f"Created {len(plot_paths)} comparison visualizations")
        return plot_paths

    def select_best_model(
        self,
        training_results: List[Dict[str, Any]],
        selection_criteria: Dict[str, Any] = None,
        weights: Dict[str, float] = None,
    ) -> Dict[str, Any]:
        """
        Select the best model based on multiple criteria.

        Args:
            training_results: List of training result dictionaries
            selection_criteria: Criteria for model selection
            weights: Weights for different metrics

        Returns:
            Best model information
        """
        if selection_criteria is None:
            selection_criteria = {
                "primary_metric": "test_r2_score",
                "min_r2_threshold": 0.6,
                "max_complexity_penalty": True,
                "prefer_simpler_models": True,
            }

        if weights is None:
            weights = {"performance": 0.7, "simplicity": 0.2, "robustness": 0.1}

        # Compare models
        comparison_df = self.compare_training_results(
            training_results, primary_metric=selection_criteria["primary_metric"]
        )

        # Apply selection criteria
        filtered_df = comparison_df.copy()

        # Filter by minimum performance threshold
        if "min_r2_threshold" in selection_criteria:
            min_threshold = selection_criteria["min_r2_threshold"]
            r2_col = selection_criteria["primary_metric"]
            if r2_col in filtered_df.columns:
                filtered_df = filtered_df[filtered_df[r2_col] >= min_threshold]

        if filtered_df.empty:
            logger.warning("No models meet the minimum performance criteria")
            filtered_df = comparison_df

        # Calculate composite score
        if len(filtered_df) > 1:
            scores = []

            for _, row in filtered_df.iterrows():
                score = 0

                # Performance score (normalized)
                performance_score = row[selection_criteria["primary_metric"]]
                max_performance = filtered_df[
                    selection_criteria["primary_metric"]
                ].max()
                normalized_performance = performance_score / max_performance
                score += weights["performance"] * normalized_performance

                # Simplicity score (inverse of complexity)
                if "best_params_count" in row and not pd.isna(row["best_params_count"]):
                    complexity = row["best_params_count"]
                    max_complexity = filtered_df["best_params_count"].max()
                    simplicity_score = (
                        1 - (complexity / max_complexity) if max_complexity > 0 else 1
                    )
                    score += weights["simplicity"] * simplicity_score

                # Robustness score (CV stability)
                if "cv_r2_std" in row and not pd.isna(row["cv_r2_std"]):
                    cv_std = row["cv_r2_std"]
                    max_std = filtered_df["cv_r2_std"].max()
                    robustness_score = 1 - (cv_std / max_std) if max_std > 0 else 1
                    score += weights["robustness"] * robustness_score

                scores.append(score)

            # Select model with highest composite score
            best_idx = np.argmax(scores)
            best_model_idx = filtered_df.index[best_idx]
            best_composite_score = scores[best_idx]
        else:
            # Only one model or using simple ranking
            best_model_idx = filtered_df.index[0]
            best_composite_score = 1.0

        # Get best model information
        best_model_info = {
            "index": best_model_idx,
            "model_type": comparison_df.loc[best_model_idx, "model_type"],
            "performance": comparison_df.loc[
                best_model_idx, selection_criteria["primary_metric"]
            ],
            "rank": comparison_df.loc[best_model_idx, "rank"],
            "composite_score": best_composite_score,
            "selection_criteria": selection_criteria,
            "weights_used": weights,
            "comparison_results": comparison_df,
        }

        # Add all metrics for the best model
        for col in comparison_df.columns:
            if col not in best_model_info:
                best_model_info[col] = comparison_df.loc[best_model_idx, col]

        self.best_model_info = best_model_info

        logger.info(
            f"Best model selected: {best_model_info['model_type']} "
            f"with {selection_criteria['primary_metric']}: {best_model_info['performance']:.4f}"
        )

        return best_model_info

    def generate_model_report(
        self, save_path: str = "model_comparison_report.html"
    ) -> str:
        """
        Generate a comprehensive HTML report of model comparison.

        Args:
            save_path: Path to save the report

        Returns:
            Path to the generated report
        """
        if self.comparison_results.empty:
            logger.error("No comparison results available")
            return ""

        html_content = f"""
        <!DOCTYPE html>
        <html>
        <head>
            <title>Model Comparison Report</title>
            <style>
                body {{ font-family: Arial, sans-serif; margin: 20px; }}
                table {{ border-collapse: collapse; width: 100%; }}
                th, td {{ border: 1px solid #ddd; padding: 8px; text-align: left; }}
                th {{ background-color: #f2f2f2; }}
                .best-model {{ background-color: #e8f5e8; }}
                .metric {{ font-weight: bold; }}
            </style>
        </head>
        <body>
            <h1>Model Comparison Report</h1>
            <p>Generated on: {pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S')}</p>
            
            <h2>Summary</h2>
            <p>Total models compared: {len(self.comparison_results)}</p>
            
            {f"<p><strong>Best model:</strong> {self.best_model_info['model_type']} " +
             f"(Rank: {self.best_model_info['rank']})</p>" if self.best_model_info else ""}
            
            <h2>Detailed Comparison</h2>
            {self.comparison_results.to_html(classes='comparison-table', escape=False)}
            
            <h2>Model Performance Metrics</h2>
            <ul>
        """

        # Add metric descriptions
        metric_descriptions = {
            "test_r2_score": "R² Score on test set (higher is better, max = 1.0)",
            "test_rmse": "Root Mean Square Error on test set (lower is better)",
            "test_mae": "Mean Absolute Error on test set (lower is better)",
            "cv_r2_mean": "Mean R² Score from cross-validation",
            "cv_r2_std": "Standard deviation of CV R² scores (lower indicates more stable model)",
        }

        for metric, description in metric_descriptions.items():
            if metric in self.comparison_results.columns:
                html_content += (
                    f"<li><span class='metric'>{metric}:</span> {description}</li>"
                )

        html_content += """
            </ul>
            
            <h2>Model Selection Criteria</h2>
            <p>Models were ranked based on their test set performance and additional criteria such as:</p>
            <ul>
                <li>Primary performance metric (typically R² score)</li>
                <li>Cross-validation stability</li>
                <li>Model complexity and interpretability</li>
                <li>Training efficiency</li>
            </ul>
            
        </body>
        </html>
        """

        # Save report
        with open(save_path, "w") as f:
            f.write(html_content)

        logger.info(f"Model comparison report saved to {save_path}")
        return save_path

    def save_comparison_results(
        self, save_path: str = "model_comparison_results.csv"
    ) -> str:
        """
        Save comparison results to CSV file.

        Args:
            save_path: Path to save the results

        Returns:
            Path to the saved file
        """
        if self.comparison_results.empty:
            logger.error("No comparison results to save")
            return ""

        self.comparison_results.to_csv(save_path, index=False)
        logger.info(f"Comparison results saved to {save_path}")
        return save_path
