

import pandas as pd
from src import config
from src.utils import get_logger, retry

logger = get_logger(__name__)

@retry(max_retries=3, delay=2, exceptions=(FileNotFoundError, IOError))
def load_data(filepath: str = None) -> pd.DataFrame:
    """
    Load data from a CSV file.
    
    Args:
        filepath (str, optional): Path to the CSV file. Defaults to config.DATA_PATH.
        
    Returns:
        pd.DataFrame: Loaded dataframe.
    """
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

    
