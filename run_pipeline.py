"""
Main script to run the pipeline.
"""
from datetime import datetime
from src import config
from src import data_loader
from src import preprocessing
from src import eda
from src import feature_engineering
from src import train 
from src import evaluate
from src.utils import get_logger
from src.log_metadata import log_training_metadata
import mlflow
import mlflow.sklearn

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

        # 4. NEW PIPELINE-BASED APPROACH
        # Create the preprocessing pipeline
        preprocessing_pipeline = feature_engineering.create_preprocessing_pipeline(scale_numeric=True)
        
        # Prepare target variable
        target_col = config.TARGET_COL
        if target_col in df.columns:
            logger.info(f"Target column found: {target_col}")
            
            # Separate features and target
            # We pass RAW data (after basic preprocessing) to the pipeline
            # Only drop the target column (leakage columns already removed from query)
            drop_cols = [target_col]
            
            X_raw = df.drop(columns=drop_cols, errors='ignore')
            y = df[target_col]
            
            logger.info(f"Raw input shape: {X_raw.shape}")
            logger.info(f"Raw input columns: {X_raw.columns.tolist()}")
            
            # 5. Train Model with Pipeline
            # The pipeline will handle all feature engineering internally
            logger.info("--- Training with Full Pipeline (Preprocessing + Model) ---")
            
            mlflow.set_tracking_uri(config.MLFLOW_TRACKING_URI)
            mlflow.set_experiment(config.MLFLOW_EXPERIMENT_NAME)
            
            with mlflow.start_run(run_name=f"Flight_Fare_{datetime.now().strftime('%Y%m%d_%H%M%S')}"):
                best_model_pipeline, X_test_raw, y_test = train.train_model(
                    X_raw, y,
                    preprocessing_pipeline=preprocessing_pipeline,
                    model_type="tune_gradient_boosting",
                    log_target=True,
                    remove_outliers=True
                )
                
                # 6. Evaluate
                # Note: X_test_raw is still raw data, the pipeline will transform it
                metrics = evaluate.evaluate_model(best_model_pipeline, X_test_raw, y_test)
                
                # Register the model in MLflow Model Registry
                logger.info(f"Registering model in MLflow Registry: {config.MLFLOW_MODEL_NAME}")
                mlflow.sklearn.log_model(
                    sk_model=best_model_pipeline,
                    artifact_path="model",
                    registered_model_name=config.MLFLOW_MODEL_NAME
                )
                
                logger.info(f"Full pipeline saved to {config.MODEL_PATH} and logged to MLflow")
                logger.info("The saved model can now accept raw input data for inference!")
            
            # 7. Log Training Metadata (Database)
            # Get the max timestamp from the data
            if 'Date' in df.columns:
                data_max_timestamp = df['Date'].max()
            else:
                data_max_timestamp = datetime.now()
            
            # Extract metrics if available
            r2_score = metrics.get('r2') if isinstance(metrics, dict) else None
            mse = metrics.get('mse') if isinstance(metrics, dict) else None
            
            log_training_metadata(
                data_max_timestamp=data_max_timestamp,
                records_count=len(df),
                r2_score=r2_score,
                mse=mse
            )
            
        else:
            logger.warning(f"Target column '{target_col}' not found, skipping training step.")
        
        logger.info("Pipeline completed successfully")
    except Exception as e:
        logger.critical(f"Pipeline failed: {e}")
        raise

if __name__ == "__main__":
    main()
