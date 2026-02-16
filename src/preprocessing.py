"""
Module for data preprocessing.
"""

import pandas as pd
from src.utils import get_logger

logger = get_logger(__name__)

def preprocess_data(df: pd.DataFrame) -> pd.DataFrame:
    """
    Preprocess the raw dataframe by cleaning missing values and fixing types.
    
    Args:
        df (pd.DataFrame): Raw dataframe.
        
    Returns:
        pd.DataFrame: Cleaned dataframe.
    """
    try:
        logger.info("Starting preprocessing...")
        initial_shape = df.shape
        
        # Drop junk columns
        df = df.loc[:, ~df.columns.str.contains("Unnamed")]

        # Normalize City Names
        # Replace 'Dacca' with 'Dhaka' for consistency
        for col in ['Source', 'Destination']:
            if col in df.columns:
                df[col] = df[col].replace('Dacca', 'Dhaka')
                logger.info(f"Normalized '{col}': Replaced 'Dacca' with 'Dhaka'.")

        # Cast numeric columns to float (Fix for PostgreSQL Decimal types)
        numeric_cols = ["Base Fare (BDT)", "Tax & Surcharge (BDT)", "Total Fare (BDT)", "Duration (hrs)"]
        for col in numeric_cols:
            if col in df.columns:
                df[col] = pd.to_numeric(df[col], errors='coerce').astype(float)
                logger.info(f"Cast '{col}' to float.")

        # Handle missing values
        # Note: In production, median should ideally be calculated from training set and applied to test.
        # For simplicity here, we assume batch processing or we would move this to a pipeline transformer.
        if "Base Fare (BDT)" in df.columns:
            df["Base Fare (BDT)"].fillna(df["Base Fare (BDT)"].median(), inplace=True)
            df = df[df["Base Fare (BDT)"] >= 0] # Remove invalid rows
            
        if "Tax & Surcharge (BDT)" in df.columns:
            df["Tax & Surcharge (BDT)"].fillna(df["Tax & Surcharge (BDT)"].median(), inplace=True)

        # Fix datatypes
        if "Departure Date & Time" in df.columns:
            df["Date"] = pd.to_datetime(df["Departure Date & Time"])
            logger.info("Converted 'Departure Date & Time' to datetime object.")

        logger.info(f"Preprocessing completed. Rows removed: {initial_shape[0] - df.shape[0]}")
        return df
    except Exception as e:
        logger.error(f"Error during preprocessing: {e}")
        raise

