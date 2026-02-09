"""
Module for training the model.
"""

"""
Module for training the model.
"""
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score
from src import config
from src.utils import get_logger, save_model

logger = get_logger(__name__)

def train_model(X: pd.DataFrame, y: pd.Series) -> RandomForestRegressor:
    """
    Train a machine learning model.
    
    Args:
        X (pd.DataFrame): Features.
        y (pd.Series): Target variable.
        
    Returns:
        RandomForestRegressor: Trained model.
    """
    try:
        logger.info("Starting model training...")
        
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=config.TEST_SIZE, random_state=config.RANDOM_SEED
        )
        logger.info(f"Data split into train ({X_train.shape}) and test ({X_test.shape}) sets.")
        
        # Initialize model
        model = RandomForestRegressor(
            n_estimators=100,
            random_state=config.RANDOM_SEED,
            n_jobs=-1
        )
        
        # Train model
        logger.info("Training Random Forest Regressor...")
        model.fit(X_train, y_train)
        logger.info("Model training completed.")
        
        # Evaluate on test set (internal check)
        y_pred = model.predict(X_test)
        mse = mean_squared_error(y_test, y_pred)
        r2 = r2_score(y_test, y_pred)
        logger.info(f"Test Set Metrics - MSE: {mse:.4f}, R2: {r2:.4f}")
        
        # Save model
        save_model(model, config.MODEL_PATH)
        
        return model, X_test, y_test
    except Exception as e:
        logger.error(f"Error during model training: {e}")
        raise

