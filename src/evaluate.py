"""
Module for evaluating the model.
"""


import pandas as pd
import numpy as np
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from src.utils import get_logger

logger = get_logger(__name__)

def evaluate_model(model, X_test: pd.DataFrame, y_test: pd.Series) -> dict:
    """
    Evaluate the trained model.
    
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
        
        metrics = {
            "MAE": mean_absolute_error(y_test, y_pred),
            "MSE": mean_squared_error(y_test, y_pred),
            "RMSE": np.sqrt(mean_squared_error(y_test, y_pred)),
            "R2": r2_score(y_test, y_pred)
        }
        
        logger.info(f"Model Evaluation Metrics:")
        for metric, value in metrics.items():
            logger.info(f"{metric}: {value:.4f}")
            
        return metrics
    except Exception as e:
        logger.error(f"Error during evaluation: {e}")
        raise

