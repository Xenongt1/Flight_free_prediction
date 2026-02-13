"""
Module for training the model.
"""

"""
Module for training the model.
"""
import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.tree import DecisionTreeRegressor
from sklearn.compose import TransformedTargetRegressor
from sklearn.metrics import mean_squared_error, r2_score
from src import config
from src.utils import get_logger, save_model

logger = get_logger(__name__)

def train_model(X: pd.DataFrame, y: pd.Series, model_type: str = "random_forest", 
                log_target: bool = False, remove_outliers: bool = False) -> tuple:
    """
    Train a machine learning model.
    Supports: 'random_forest', 'linear_regression', 'ridge', 'lasso', 'decision_tree', 'gradient_boosting'.
    Includes feature importance visualization.
    
    Args:
        X (pd.DataFrame): Features.
        y (pd.Series): Target variable.
        model_type (str): Type of model to train.
        log_target (bool): Whether to apply log1p transformation to target.
        remove_outliers (bool): Whether to remove outliers from training set (>99th percentile).
        
    Returns:
        RandomForestRegressor: Trained model.
    """
    try:
        logger.info("Starting model training...")
        
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=config.TEST_SIZE, random_state=config.RANDOM_SEED
        )
        if remove_outliers:
            # Remove outliers from training set only
            upper_limit = y_train.quantile(0.99)
            mask = y_train <= upper_limit
            logger.info(f"Removing outliers > {upper_limit:.2f} from training set. Dropped {(~mask).sum()} rows.")
            X_train = X_train[mask]
            y_train = y_train[mask]

            logger.info(f"Removing outliers > {upper_limit:.2f} from training set. Dropped {(~mask).sum()} rows.")
            X_train = X_train[mask]
            y_train = y_train[mask]

        # Initialize model base
        if model_type == "linear_regression":
            base_model = LinearRegression()
            logger.info("Initialized Linear Regression model.")
        elif model_type == "ridge":
            base_model = Ridge(alpha=1.0, random_state=config.RANDOM_SEED)
            logger.info("Initialized Ridge Regression model.")
        elif model_type == "lasso":
            base_model = Lasso(alpha=1.0, random_state=config.RANDOM_SEED)
            logger.info("Initialized Lasso Regression model.")
        elif model_type == "decision_tree":
            base_model = DecisionTreeRegressor(random_state=config.RANDOM_SEED)
            logger.info("Initialized Decision Tree Regressor.")
        elif model_type == "gradient_boosting":
            base_model = GradientBoostingRegressor(random_state=config.RANDOM_SEED)
            logger.info("Initialized Gradient Boosting Regressor.")
        elif model_type == "tune_gradient_boosting":
            logger.info("Starting Hyperparameter Tuning for Gradient Boosting...")
            param_grid = {
                'n_estimators': [100, 200],
                'learning_rate': [0.05, 0.1],
                'max_depth': [3, 5],
                'subsample': [0.8, 1.0]
            }
            # Note: For GridSearchCV inside TransformedTargetRegressor, we usually wrap the grid search or search inside.
            # However, simpler: Wrap the Regressor, then GridSearch the Wrapper? Or GridSearch the Regressor, then Wrap?
            # Correct approach: TransformedTargetRegressor(regressor=GridSearchCV(...)) - No, inverse func applies to y.
            # Simpler: GridSearchCV(TransformedTargetRegressor(regressor=GB, ...))
            # But params need to be prefixed with 'regressor__'.
            
            # Let's define the base estimator as the wrapped regressor if log_target is on.
            gb = GradientBoostingRegressor(random_state=config.RANDOM_SEED)
            
            if log_target:
                # If tuning, we want to tune the INNER model.
                # So we wrap the GB first.
                # But GridSearchCV expects a model that implements fit/predict.
                # If we wrap GB in TTR, then pass TTR to GridSearchCV, we tune TTR.
                # Params would be 'regressor__n_estimators'.
                
                # New Param Grid with prefix
                param_grid = {f'regressor__{k}': v for k, v in param_grid.items()}
                
                wrapped_gb = TransformedTargetRegressor(
                    regressor=gb, 
                    func=np.log1p, 
                    inverse_func=np.expm1
                )
                base_model = GridSearchCV(
                    estimator=wrapped_gb,
                    param_grid=param_grid,
                    cv=3,
                    scoring='r2',
                    n_jobs=-1,
                    verbose=2
                )
                # Since we handled wrapping here, we set log_target to False for the final wrapping step below
                # to avoid double wrapping.
                log_target_handled = True
            else:
                base_model = GridSearchCV(
                    estimator=gb,
                    param_grid=param_grid,
                    cv=3,
                    scoring='r2',
                    n_jobs=-1,
                    verbose=2
                )
                log_target_handled = False
                
        else:
            # Default to Random Forest
            base_model = RandomForestRegressor(
                n_estimators=100,
                max_depth=20, # Added depth limit to prevent overfitting
                random_state=config.RANDOM_SEED,
                n_jobs=-1
            )
            logger.info("Initialized Random Forest Regressor.")

        # Final Model Wrapping (if not handled in tuning)
        if log_target and model_type != "tune_gradient_boosting":
            logger.info("Wrapping model with TransformedTargetRegressor (Log1p).")
            model = TransformedTargetRegressor(
                regressor=base_model,
                func=np.log1p,
                inverse_func=np.expm1
            )
        elif log_target and model_type == "tune_gradient_boosting":
             # Already handled inside the tuning block logic above
             model = base_model
        else:
            model = base_model
            
        
        # Train model
        logger.info(f"Training {model_type}...")
        model.fit(X_train, y_train)
        logger.info("Model training completed.")
        
        if hasattr(model, 'feature_importances_'):
            try:
                _plot_feature_importance(model, X_train.columns)
            except Exception:
                pass # safely ignore if plotting fails
        # Handle wrapped models (TransformedTargetRegressor or GridSearchCV)
        elif isinstance(model, TransformedTargetRegressor):
             if hasattr(model.regressor, 'feature_importances_'):
                 try:
                    _plot_feature_importance(model.regressor, X_train.columns)
                 except Exception: pass
             elif hasattr(model.regressor_, 'feature_importances_'): # after fit
                 try:
                    _plot_feature_importance(model.regressor_, X_train.columns)
                 except Exception: pass
        elif isinstance(model, GridSearchCV):
             # Evaluate best estimator
             best_est = model.best_estimator_
             if isinstance(best_est, TransformedTargetRegressor):
                 if hasattr(best_est.regressor_, 'feature_importances_'):
                     try:
                        _plot_feature_importance(best_est.regressor_, X_train.columns)
                     except Exception: pass
             elif hasattr(best_est, 'feature_importances_'):
                 try:
                    _plot_feature_importance(best_est, X_train.columns)
                 except Exception: pass

        # Evaluate on test set (internal check)
        y_pred = model.predict(X_test)
        
        # Since we use TransformedTargetRegressor, y_pred is already in original scale.
        # No manual inverse transform needed.
            
        mse = mean_squared_error(y_test, y_pred)
        r2 = r2_score(y_test, y_pred)
        logger.info(f"Test Set Metrics - MSE: {mse:.4f}, R2: {r2:.4f}")
        
        # Save model
        # Save model (if GridSearchCV, save best_estimator_)
        if isinstance(model, GridSearchCV):
            best_model = model.best_estimator_
            logger.info(f"Best Parameters: {model.best_params_}")
            save_model(best_model, config.MODEL_PATH)
            return best_model, X_test, y_test
        else:
            save_model(model, config.MODEL_PATH)
            return model, X_test, y_test
    except Exception as e:
        logger.error(f"Error during model training: {e}")
        raise

def _plot_coefficients(model, feature_names):
    """Plot and save linear regression coefficients."""
    try:
        coefs = model.coef_
        indices = np.argsort(np.abs(coefs))[::-1]
        
        # Select top 20 features
        top_n = 20
        indices = indices[:top_n]
        
        plt.figure(figsize=(12, 8))
        plt.title("Top Linear Model Coefficients")
        plt.bar(range(top_n), coefs[indices], align="center")
        plt.xticks(range(top_n), [feature_names[i] for i in indices], rotation=45, ha='right')
        plt.xlim([-1, top_n])
        plt.tight_layout()
        plt.savefig(os.path.join(config.PLOTS_DIR, "linear_coefficients.png"))
        plt.close()
        logger.info("Linear coefficients plot saved.")
    except Exception as e:
        logger.warning(f"Could not plot coefficients: {e}")

def _plot_feature_importance(model, feature_names):
    """Plot and save feature importance."""
    try:
        if hasattr(model, 'feature_importances_'):
            importances = model.feature_importances_
            indices = np.argsort(importances)[::-1]
            
            # Select top 20 features
            top_n = 20
            indices = indices[:top_n]
            
            plt.figure(figsize=(12, 8))
            plt.title("Top Feature Importances")
            plt.bar(range(top_n), importances[indices], align="center")
            plt.xticks(range(top_n), [feature_names[i] for i in indices], rotation=45, ha='right')
            plt.xlim([-1, top_n])
            plt.tight_layout()
            plt.savefig(os.path.join(config.PLOTS_DIR, "feature_importance.png"))
            plt.close()
            logger.info("Feature importance plot saved.")
    except Exception as e:
        logger.warning(f"Could not plot feature importance: {e}")

