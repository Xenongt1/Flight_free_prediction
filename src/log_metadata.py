"""
Utility function to log ML training metadata to the database.
"""
import os
from sqlalchemy import create_engine, text
from datetime import datetime

def log_training_metadata(data_max_timestamp, records_count, r2_score=None, mse=None):
    """
    Log training metadata to ml_metadata.model_training_log table.
    
    Args:
        data_max_timestamp: Latest timestamp in the training data
        records_count: Number of records used for training
        r2_score: R2 score of the model (optional)
        mse: Mean Squared Error (optional)
    """
    # Database connection
    DB_USER = os.getenv('DB_USER', 'postgres')
    DB_PASSWORD = os.getenv('DB_PASSWORD', 'postgres')
    DB_HOST = os.getenv('DB_HOST', 'localhost')
    DB_PORT = os.getenv('DB_PORT', '5432')
    DB_NAME = os.getenv('DB_NAME', 'flight_analytics')
    
    connection_string = f"postgresql+psycopg2://{DB_USER}:{DB_PASSWORD}@{DB_HOST}:{DB_PORT}/{DB_NAME}"
    
    try:
        engine = create_engine(connection_string)
        
        with engine.begin() as conn:
            # Create schema if it doesn't exist
            conn.execute(text("CREATE SCHEMA IF NOT EXISTS ml_metadata"))
            
            # Create table if it doesn't exist
            create_table_query = text("""
                CREATE TABLE IF NOT EXISTS ml_metadata.model_training_log (
                    training_id SERIAL PRIMARY KEY,
                    model_name VARCHAR(100) NOT NULL,
                    training_timestamp TIMESTAMP NOT NULL DEFAULT CURRENT_TIMESTAMP,
                    data_max_timestamp TIMESTAMP NOT NULL,
                    records_trained_on INTEGER,
                    model_version VARCHAR(50),
                    r2_score FLOAT,
                    mse FLOAT,
                    notes TEXT
                )
            """)
            conn.execute(create_table_query)
            
            # Insert training log
            insert_query = text("""
                INSERT INTO ml_metadata.model_training_log 
                (model_name, data_max_timestamp, records_trained_on, r2_score, mse, model_version)
                VALUES (:model_name, :data_max_timestamp, :records_count, :r2_score, :mse, :version)
            """)
            
            conn.execute(insert_query, {
                'model_name': 'flight_fare_predictor',
                'data_max_timestamp': data_max_timestamp,
                'records_count': records_count,
                'r2_score': r2_score,
                'mse': mse,
                'version': datetime.now().strftime('%Y%m%d_%H%M%S')
            })
            
            print(f"Training metadata logged successfully. Data timestamp: {data_max_timestamp}, Records: {records_count}")
            
    except Exception as e:
        print(f"Error logging training metadata: {e}")
        # Don't raise - logging failure shouldn't stop the pipeline
