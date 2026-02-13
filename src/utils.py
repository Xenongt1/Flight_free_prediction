"""
Utility functions for the project.
"""
import logging
import joblib
from src import config

import time
import functools
from sqlalchemy import create_engine

def get_db_engine():
    """
    Creates and returns a SQLAlchemy engine for the database.
    """
    try:
        engine = create_engine(config.DB_CONNECTION_STRING)
        return engine
    except Exception as e:
        get_logger(__name__).error(f"Error creating DB engine: {e}")
        raise

def get_logger(name):
    """
    Creates and returns a logger instance.
    """
    logger = logging.getLogger(name)
    if not logger.handlers:
        logger.setLevel(logging.INFO)
        
        # File Handler
        file_handler = logging.FileHandler(config.LOG_FILE)
        file_formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
        file_handler.setFormatter(file_formatter)
        logger.addHandler(file_handler)
        
        # Stream Handler (Console)
        stream_handler = logging.StreamHandler()
        stream_formatter = logging.Formatter('%(levelname)s: %(message)s')
        stream_handler.setFormatter(stream_formatter)
        logger.addHandler(stream_handler)
        
    return logger

def retry(max_retries=3, delay=1, backoff=2, exceptions=(Exception,)):
    """
    Retry decorator with exponential backoff.
    
    Args:
        max_retries (int): Maximum number of retries.
        delay (int): Initial delay in seconds.
        backoff (int): Multiplier for delay after each retry.
        exceptions (tuple): Tuple of exceptions to catch and retry on.
    """
    def decorator(func):
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            mtries, mdelay = max_retries, delay
            while mtries > 1:
                try:
                    return func(*args, **kwargs)
                except exceptions as e:
                    logger = get_logger(__name__)
                    msg = f"Retrying {func.__name__} in {mdelay} seconds... Error: {e}"
                    logger.warning(msg)
                    time.sleep(mdelay)
                    mtries -= 1
                    mdelay *= backoff
            return func(*args, **kwargs)
        return wrapper
    return decorator

def save_model(model, filepath):
    """
    Save the trained model to a file.
    """
    try:
        joblib.dump(model, filepath)
        get_logger(__name__).info(f"Model saved to {filepath}")
    except Exception as e:
        get_logger(__name__).error(f"Error saving model: {e}")
        raise

def load_model(filepath):
    """
    Load a trained model from a file.
    """
    try:
        model = joblib.load(filepath)
        get_logger(__name__).info(f"Model loaded from {filepath}")
        return model
    except Exception as e:
        get_logger(__name__).error(f"Error loading model: {e}")
        raise

