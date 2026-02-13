import sys
import os
import pandas as pd
from sqlalchemy import text # Import text for SQL queries

# Add project root to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src import config
from src.utils import get_db_engine

def test_connection():
    print(f"Testing connection to: {config.DB_HOST}:{config.DB_PORT}/{config.DB_NAME}")
    try:
        engine = get_db_engine()
        with engine.connect() as connection:
            print("Successfully connected to the database!")
            
            # Test query - using raw execution to bypass pandas dispatch issues
            print("\nAttempting to run test query...")
            query = text("SELECT 1 as connection_test")
            result = connection.execute(query)
            
            # Fetch results and manually construct DataFrame
            columns = result.keys()
            data = result.fetchall()
            df = pd.DataFrame(data, columns=columns)
            
            print("Query successful. Result:")
            print(df)
            
    except Exception as e:
        print(f"\nConnection Failed! Error: {e}")

if __name__ == "__main__":
    test_connection()
