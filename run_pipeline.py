"""
Main script to run the pipeline.
"""
from src import config
from src import data_loader
from src import preprocessing
from src import feature_engineering
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

        # 3. Feature Engineering
        df = feature_engineering.engineer_features(df)

        # Save processed data
        df.to_csv(config.PROCESSED_DATA_PATH, index=False)
        logger.info(f"Processed data saved to {config.PROCESSED_DATA_PATH}")

        # 4. Train Model (Placeholder)
        # Uncomment the following when ready to train
        """
        target_col = config.TARGET_COL
        if target_col in df.columns:
            logger.info(f"Target column found: {target_col}")
            
            # Prepare X and y
            # Drop columns that are not useful for training or cause leakage
            drop_cols = [target_col, 'Date', 'Departure Date & Time', 'Arrival Date & Time']
            
            # Select numeric columns for now (results of One-Hot Encoding + original numeric)
            X = df.select_dtypes(include=['number']).drop(columns=[target_col], errors='ignore')
            y = df[target_col]
            
            logger.info(f"Training features: {X.columns.tolist()}")
            
            # Train model
            model, X_test, y_test = train.train_model(X, y)
            
            # 5. Evaluate Model
            evaluate.evaluate_model(model, X_test, y_test)
            
        else:
            logger.warning(f"Target column '{target_col}' not found, skipping training step.")
        """
        
        logger.info("Pipeline completed successfully (Data Preparation Stage)")
    except Exception as e:
        logger.critical(f"Pipeline failed: {e}")
        raise

if __name__ == "__main__":
    main()
