"""
Main script to run the pipeline.
"""
from src import config
from src import data_loader
from src import preprocessing
from src import eda
from src import feature_engineering
from src import train 
from src import evaluate
from src.utils import get_logger

# Import training modules but don't use them yet
# from src import train 
# from src import evaluate

logger = get_logger(__name__)

def main():
    try:
        logger.info("Running pipeline...")
        
        # 1. Load Data
        df = data_loader.load_data()

        # 2. Preprocessing
        df = preprocessing.preprocess_data(df)

        # 3. Feature Engineering (EDA Version - No Encoding)
        # We first create features without encoding to save for EDA
        df_eda = feature_engineering.engineer_features(df.copy(), encode=False)
        
        # Save EDA data
        df_eda.to_csv(config.EDA_DATA_PATH, index=False)
        logger.info(f"EDA data saved to {config.EDA_DATA_PATH}")

        # Perform EDA
        eda.perform_eda(df_eda)

        # 4. Feature Engineering (Model Version - With Encoding)
        # Now we proceed with encoding for the model
        # We can either use df_eda and encode it, or run engineer_features(df, encode=True)
        # Let's run it from scratch on the preprocessed df to be safe and consistent with the function design
        df_model = feature_engineering.engineer_features(df, encode=True, scale=True)

        # Save processed data for modeling
        df_model.to_csv(config.PROCESSED_DATA_PATH, index=False)
        logger.info(f"Model training data saved to {config.PROCESSED_DATA_PATH}")


        # 5. Train Model
        target_col = config.TARGET_COL
        if target_col in df_model.columns:
            logger.info(f"Target column found: {target_col}")
            
            # Prepare X and y
            # Drop columns that are not useful for training or cause leakage
            # Note: We drop columns that are clearly not features (Date, Time objects)
            # The 'Date' column is a datetime object which might cause issues with sklearn if not handled,
            # but we extracted Month/Day/Weekday from it, so we can drop it.
            # CRITICAL: Drop constituent parts of the target variable to avoid data leakage
            drop_cols = [
                target_col, 
                'Date', 
                'Departure Date & Time', 
                'Arrival Date & Time',
                'Base Fare (BDT)',       # LEAKAGE
                'Tax & Surcharge (BDT)',  # LEAKAGE
                'Class' # We created Class_Encoded, so drop 'Class' to avoid duplication/issues if it wasn't encoded one-hot (it wasn't in our list)
            ]
            
            # Select features. 
            # X should be all columns except the target and the ones we explicitly drop.
            # Since we encoded categoricals, they are now numeric.
            X = df_model.drop(columns=drop_cols, errors='ignore')
            
            # Double check to ensure all columns are numeric
            # If there are any remaining object columns (e.g. names we didn't encode?), drop them
            # Also, get_dummies might produce bools, which we should cast to int, or trust sklearn handles them (it usually does).
            # To be safe, let's cast bools to int.
            for col in X.select_dtypes(include=['bool']).columns:
                X[col] = X[col].astype(int)
                
            X = X.select_dtypes(include=['number'])
            
            y = df_model[target_col]
            
            logger.info(f"Training features ({len(X.columns)}): {X.columns.tolist()}")
            
            # Model Selection
            # Hyperparameter Tuning
            # Since Gradient Boosting performed best, we will tune it.
            # Hyperparameter Tuning
            # Since Gradient Boosting performed best, we will tune it.
            logger.info("--- Tuning Gradient Boosting with Log Target & Outlier Removal ---")
            
            # Using log_target=True and remove_outliers=True to improve accuracy
            best_model_obj, X_test, y_test = train.train_model(
                X, y, 
                model_type="tune_gradient_boosting",
                log_target=True,
                remove_outliers=True
            )
            
            # Evaluate Optimized Model
            # Note: Since we use TransformedTargetRegressor in train.py (if updated) or manual handling?
            # I haven't updated train.py to use TransformedTargetRegressor yet.
            # I must do that next.
            # But for now, let's assume train.py will be updated to return a model that predicts meaningful values.
            
            evaluate.evaluate_model(best_model_obj, X_test, y_test)
            
            # Save final best model
            # It's already saved by train_model, but we log it.
            logger.info(f"Optimized model saved to {config.MODEL_PATH}")
            
            # Generate Report
            # (Optional)
        else:
            logger.warning(f"Target column '{target_col}' not found, skipping training step.")
        
        logger.info("Pipeline completed successfully")
    except Exception as e:
        logger.critical(f"Pipeline failed: {e}")
        raise

if __name__ == "__main__":
    main()
