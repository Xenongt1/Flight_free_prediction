"""
Module for training the model with sklearn Pipeline support.
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
from sklearn.pipeline import Pipeline
from sklearn.metrics import mean_squared_error, r2_score
from src import config
from src.utils import get_logger, save_model

logger = get_logger(__name__)

def train_model(X: pd.DataFrame, y: pd.Series, preprocessing_pipeline=None,
                model_type: str = "random_forest", 
                log_target: bool = False, remove_outliers: bool = False) -> tuple:
    """
    Train a machine learning model with optional preprocessing pipeline.
    
    Args:
        X (pd.DataFrame): Raw features (before preprocessing) if preprocessing_pipeline is provided,
                          or already-transformed features if preprocessing_pipeline is None.
        y (pd.Series): Target variable.
        preprocessing_pipeline: sklearn Pipeline for feature engineering (optional).
        model_type (str): Type of model to train.
        log_target (bool): Whether to apply log1p transformation to target.
        remove_outliers (bool): Whether to remove outliers from training set (>99th percentile).
        
    Returns:
        tuple: (trained_pipeline_or_model, X_test, y_test)
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
                'n_estimators': [100], # Reduced from [100, 200]
                'learning_rate': [0.1], # Kept the most common sweet spot
                'max_depth': [3, 5],
                'subsample': [0.8] # Standard for GB
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
                    n_jobs=1, # Set to 1 to avoid BrokenPipeError in Airflow/Docker
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
                    n_jobs=1, # Set to 1 to avoid BrokenPipeError in Airflow/Docker
                    verbose=2
                )
                log_target_handled = False
                
        else:
            # Default to Random Forest
            base_model = RandomForestRegressor(
                n_estimators=100,
                max_depth=20, # Added depth limit to prevent overfitting
                random_state=config.RANDOM_SEED,
                n_jobs=1 # Set to 1 to avoid BrokenPipeError in Airflow/Docker
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
            
        # 4. Integrate Preprocessing and Model into a single Pipeline
        if preprocessing_pipeline is not None:
            logger.info("Creating full pipeline (preprocessing + model)...")
            # The model here could be a simple regressor, a TTR, or a GridSearchCV
            full_pipeline = Pipeline([
                ('preprocessing', preprocessing_pipeline),
                ('model', model)
            ])
            
            logger.info(f"Training full pipeline on {X_train.shape[0]} samples...")
            full_pipeline.fit(X_train, y_train)
            
            # Prepare for evaluation
            # For evaluation and prediction, we use the raw X_test
            X_test_to_eval = X_test
            trained_model = full_pipeline
            
            # For feature importance plotting, we need transformed data names
            try:
                # Extract transformed features names
                preprocessor = full_pipeline.named_steps['preprocessing']
                if hasattr(preprocessor, 'get_feature_names_out'):
                    feature_names = preprocessor.get_feature_names_out()
                    # Plot importance if applicable
                    _plot_feature_importance_helper(model, feature_names)
            except Exception as e:
                logger.warning(f"Feature importance plotting skipped: {e}")
        else:
            logger.info(f"Training model on {X_train.shape[0]} samples...")
            model.fit(X_train, y_train)
            X_test_to_eval = X_test
            trained_model = model
            
            # Plot importance if applicable
            _plot_feature_importance_helper(model, X_train.columns)

        # 5. Evaluate on test set
        logger.info("Evaluating model on test set...")
        y_pred = trained_model.predict(X_test_to_eval)
            
        mse = mean_squared_error(y_test, y_pred)
        r2 = r2_score(y_test, y_pred)
        logger.info(f"Test Set Metrics - MSE: {mse:.4f}, R2: {r2:.4f}")
        
        # 6. Save and Return
        save_model(trained_model, config.MODEL_PATH)
        return trained_model, X_test, y_test
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

def _plot_feature_importance_helper(model, feature_names):
    """Helper to extract regressor from various wrappers and plot importance."""
    regressor = None
    
    # Unwrap GridSearchCV
    if hasattr(model, 'best_estimator_'):
        model = model.best_estimator_
    
    # Unwrap TransformedTargetRegressor
    if hasattr(model, 'regressor_'):
        regressor = model.regressor_
    elif hasattr(model, 'regressor'):
        regressor = model.regressor
    else:
        regressor = model
        
    if hasattr(regressor, 'feature_importances_'):
        _plot_feature_importance(regressor, feature_names)

