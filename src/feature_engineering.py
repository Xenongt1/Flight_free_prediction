"""
Module for feature engineering.
"""

import pandas as pd
from src.utils import get_logger

logger = get_logger(__name__)

def engineer_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Create new features from existing data.
    
    Args:
        df (pd.DataFrame): Preprocessed dataframe.
        
    Returns:
        pd.DataFrame: Dataframe with new features.
    """
    try:
        logger.info("Starting feature engineering...")
        
        # Create target if not exists (though it seems to exist in the dataset as 'Total Fare (BDT)')
        if "Total Fare (BDT)" not in df.columns:
            if "Base Fare (BDT)" in df.columns and "Tax & Surcharge (BDT)" in df.columns:
                df["Total Fare (BDT)"] = df["Base Fare (BDT)"] + df["Tax & Surcharge (BDT)"]
                logger.info("Created 'Total Fare (BDT)' feature.")

        # Date features
        if "Date" in df.columns:
            df["Month"] = df["Date"].dt.month
            df["Day"] = df["Date"].dt.day
            df["Weekday"] = df["Date"].dt.weekday
            logger.info("Created date-based features (Month, Day, Weekday).")

        # Encode categoricals - THIS IS NOT IDEAL FOR PRODUCTION
        # In a real production pipeline, we would use sklearn.pipeline.Pipeline with OneHotEncoder
        # to ensure consistency between training and inference.
        # However, to keep it simple as requested, we will use get_dummies but wrap it safely.
        # A better approach would be to move encoding to the training/inference pipeline.
        
        # For now, let's keep get_dummies as "feature engineering" but be aware of the drift risk.
        categorical_cols = ["Airline", "Source", "Destination"]
        # Only encode columns that actually exist
        cols_to_encode = [col for col in categorical_cols if col in df.columns]
        
        if cols_to_encode:
            df = pd.get_dummies(
                df,
                columns=cols_to_encode,
                drop_first=True
            )
            logger.info(f"One-hot encoded columns: {cols_to_encode}")

        logger.info(f"Feature engineering completed. Final shape: {df.shape}")
        return df
    except Exception as e:
        logger.error(f"Error during feature engineering: {e}")
        raise

