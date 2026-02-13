

import pandas as pd
from sqlalchemy import text
from src import config
from src.utils import get_logger, retry, get_db_engine

logger = get_logger(__name__)

def load_data_from_db() -> pd.DataFrame:
    """
    Load data from PostgreSQL database using the star schema.
    """
    logger.info("Loading data from PostgreSQL...")
    engine = get_db_engine()
    
    query = """
    SELECT 
        da.airline_name as "Airline",
        dd.date as "Date",
        dl_src.city_name as "Source",
        dl_dst.city_name as "Destination",
        ff.departure_time as "Departure Date & Time", 
        ff.arrival_time as "Arrival_Time",
        ff.duration_hrs as "Duration (hrs)",
        dfd.stopovers as "Stopovers",
        dfd.class as "Class",
        dfd.aircraft_type as "Aircraft Type",
        ff.base_fare_bdt as "Base Fare (BDT)",
        ff.tax_surcharge_bdt as "Tax & Surcharge (BDT)",
        ff.total_fare_bdt as "Total Fare (BDT)",
        dd.day as "Day",
        dd.month as "Month",
        dd.year as "Year",
        dd.season as "Season",
        dd.day_of_week as "Day_of_Week",
        dd.is_holiday_window as "Is_Holiday"
    FROM fact_flights ff
    JOIN dim_date dd ON ff.date_id = dd.date_id
    JOIN dim_airline da ON ff.airline_id = da.airline_id
    JOIN dim_location dl_src ON ff.source_location_id = dl_src.location_id
    FROM fact_flights ff
    JOIN dim_date dd ON ff.date_id = dd.date_id
    JOIN dim_airline da ON ff.airline_id = da.airline_id
    JOIN dim_location dl_src ON ff.source_location_id = dl_src.location_id
    JOIN dim_location dl_dst ON ff.destination_location_id = dl_dst.location_id
    JOIN dim_flight_details dfd ON ff.detail_id = dfd.detail_id
    """
    
    try:
        # Using raw execution to avoid pandas/sqlalchemy version mismatch
        with engine.connect() as connection:
            result = connection.execute(text(query))
            columns = result.keys()
            data = result.fetchall()
            df = pd.DataFrame(data, columns=columns)
            
        logger.info(f"Data loaded from DB successfully. Shape: {df.shape}")
        return df
    except Exception as e:
        logger.error(f"Error loading data from DB: {e}")
        raise

@retry(max_retries=3, delay=2, exceptions=(FileNotFoundError, IOError))
def load_data(filepath: str = None) -> pd.DataFrame:
    """
    Load data from a CSV file or Database based on config.
    
    Args:
        filepath (str, optional): Path to the CSV file. Defaults to config.DATA_PATH.
        
    Returns:
        pd.DataFrame: Loaded dataframe.
    """
    if config.DATA_SOURCE == 'postgres':
        return load_data_from_db()

    if filepath is None:
        filepath = config.DATA_PATH
    
    try:
        logger.info(f"Loading data from {filepath}")
        df = pd.read_csv(filepath)
        logger.info(f"Data loaded successfully. Shape: {df.shape}")
        return df
    except FileNotFoundError:
        logger.error(f"File not found at {filepath}")
        raise
    except Exception as e:
        logger.error(f"An error occurred while loading data: {e}")
        raise

    
