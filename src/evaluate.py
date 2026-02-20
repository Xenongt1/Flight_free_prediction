"""
Module for evaluating the model.
"""


import pandas as pd
import json
import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import mlflow
from src import config
from src.utils import get_logger

logger = get_logger(__name__)

def evaluate_model(model, X_test: pd.DataFrame, y_test: pd.Series) -> dict:
    """
    Evaluate the trained model, save metrics, and generate plots.
    
    Args:
        model: Trained model.
        X_test (pd.DataFrame): Test features.
        y_test (pd.Series): True values.
        
    Returns:
        dict: Dictionary of evaluation metrics.
    """
    try:
        logger.info("Starting model evaluation...")
        
        y_pred = model.predict(X_test)
        
        # Calculate metrics
        metrics = {
            "MAE": float(mean_absolute_error(y_test, y_pred)),
            "MSE": float(mean_squared_error(y_test, y_pred)),
            "RMSE": float(np.sqrt(mean_squared_error(y_test, y_pred))),
            "R2": float(r2_score(y_test, y_pred))
        }
        
        # Log metrics
        logger.info(f"Model Evaluation Metrics:")
        for metric, value in metrics.items():
            logger.info(f"{metric}: {value:.4f}")
            
        # Save metrics to JSON
        with open(config.METRICS_FILE, 'w') as f:
            json.dump(metrics, f, indent=4)
        logger.info(f"Metrics saved to {config.METRICS_FILE}")

        # Generate and save plots
        _plot_predictions(y_test, y_pred)
        _plot_residuals(y_test, y_pred)
        
        # Log to MLflow if run is active
        if mlflow.active_run():
            logger.info("Logging metrics and plots to MLflow...")
            mlflow.log_metrics(metrics)
            mlflow.log_artifact(os.path.join(config.PLOTS_DIR, "prediction_scatter.png"), artifact_path="plots")
            mlflow.log_artifact(os.path.join(config.PLOTS_DIR, "residuals_dist.png"), artifact_path="plots")
            
        return metrics
    except Exception as e:
        logger.error(f"Error during evaluation: {e}")
        raise

def _plot_predictions(y_true, y_pred):
    """Plot Actual vs Predicted values."""
    plt.figure(figsize=(10, 6))
    sns.scatterplot(x=y_true, y=y_pred, alpha=0.5)
    plt.plot([y_true.min(), y_true.max()], [y_true.min(), y_true.max()], 'r--', lw=2)
    plt.xlabel('Actual Fare')
    plt.ylabel('Predicted Fare')
    plt.title('Actual vs Predicted Fares')
    plt.tight_layout()
    plt.savefig(os.path.join(config.PLOTS_DIR, "prediction_scatter.png"))
    plt.close()
    logger.info("Prediction scatter plot saved.")

def _plot_residuals(y_true, y_pred):
    """Plot residuals."""
    residuals = y_true - y_pred
    plt.figure(figsize=(10, 6))
    sns.histplot(residuals, kde=True, bins=30)
    plt.xlabel('Residuals')
    plt.title('Residual Distribution')
    plt.tight_layout()
    plt.savefig(os.path.join(config.PLOTS_DIR, "residuals_dist.png"))
    plt.close()
    logger.info("Residuals plot saved.")

