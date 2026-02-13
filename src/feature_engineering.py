"""
Module for feature engineering.
"""

import pandas as pd
from src.utils import get_logger
from sklearn.preprocessing import StandardScaler
import numpy as np

logger = get_logger(__name__)

def engineer_features(df: pd.DataFrame, encode: bool = True, scale: bool = False) -> pd.DataFrame:
    """
    Create new features from existing data.
    
    Args:
        df (pd.DataFrame): Preprocessed dataframe.
        encode (bool): Whether to perform one-hot encoding.
        scale (bool): Whether to scale numerical features.
        
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
            
            # Time of Day Features
            # Extract hour of departure
            df["Dep_Hour"] = df["Date"].dt.hour
            # Bin into Morning, Afternoon, Evening, Night
            # Morning: 5-11, Afternoon: 12-16, Evening: 17-21, Night: 22-4
            bins = [0, 5, 12, 17, 22, 24]
            labels = ['Night', 'Morning', 'Afternoon', 'Evening', 'Night']
            # Using ordered=False to allow duplicates in labels ('Night' appears twice) if using pandas cut directly with duplicate labels raises error, 
            # so we handle it slightly carefully.
            # actually pandas cut handles duplicate labels fine if ordered=False is not strictly enforced on the labels themselves.
            # Alternatively, map hour to label.
            def get_time_of_day(hour):
                if 5 <= hour < 12: return 'Morning'
                elif 12 <= hour < 17: return 'Afternoon'
                elif 17 <= hour < 22: return 'Evening'
                else: return 'Night'
            
            df['Time_of_Day'] = df['Dep_Hour'].apply(get_time_of_day)
            
            # Cyclical Features (Month, Day, Weekday, Hour)
            # Encode cyclic nature: 23:00 close to 00:00, Dec close to Jan
            df['Month_Sin'] = np.sin(2 * np.pi * df['Month']/12)
            df['Month_Cos'] = np.cos(2 * np.pi * df['Month']/12)
            df['Day_Sin'] = np.sin(2 * np.pi * df['Day']/31)
            df['Day_Cos'] = np.cos(2 * np.pi * df['Day']/31)
            df['Weekday_Sin'] = np.sin(2 * np.pi * df['Weekday']/7)
            df['Weekday_Cos'] = np.cos(2 * np.pi * df['Weekday']/7)
            df['Hour_Sin'] = np.sin(2 * np.pi * df['Dep_Hour']/24)
            df['Hour_Cos'] = np.cos(2 * np.pi * df['Dep_Hour']/24)
            
            logger.info("Created cyclical time features (Sin/Cos).")

            # Route Feature
            if "Source" in df.columns and "Destination" in df.columns:
                df["Route"] = df["Source"] + "_" + df["Destination"]
                logger.info("Created 'Route' feature.")

            # Duration Binning
            if "Duration (hrs)" in df.columns:
                # Bin duration: Short (<5), Medium (5-10), Long (>10)
                # Need to handle potential errors if Duration is not numeric, but preproc should have fixed it.
                # Assuming numerical.
                df['Duration_Category'] = pd.cut(df['Duration (hrs)'], 
                                               bins=[-1, 5, 10, 100], 
                                               labels=['Short-Haul', 'Medium-Haul', 'Long-Haul'])
                logger.info("Created 'Duration_Category' feature.")

            # Season feature
            def get_season(month):
                if month in [12, 1, 2]:
                    return 'Winter'
                elif month in [3, 4, 5]:
                    return 'Spring'
                elif month in [6, 7, 8]:
                    return 'Summer'
                else:
                    return 'Autumn'
            
            df['Season'] = df['Month'].apply(get_season)
            logger.info("Created 'Season' feature.")

        # Encode categorical features: Stopovers (Ordinal Encoding)
        if "Stopovers" in df.columns:
            stopover_map = {"Direct": 0, "1 Stop": 1, "2 Stops": 2, "3 Stops": 3}
            # Fill NaN with mode (most frequent) or 0
            df["Stopovers"] = df["Stopovers"].map(stopover_map).fillna(0).astype(int)
            logger.info("Encoded Stopovers as ordinal.")

        # Encode categorical features: Class (Ordinal Encoding)
        # Assuming typical hierarchy: Economy < Business < First Class
        if "Class" in df.columns:
             class_map = {"Economy": 1, "Business": 2, "First Class": 3}
             df["Class_Encoded"] = df["Class"].map(class_map).fillna(1).astype(int)
             # We can keep 'Class' for one-hot if we prefer, but ordinal is good for tree models too.
             # Let's use ordinal for simplicity.
             logger.info("Encoded Class as ordinal.")


        if encode:
            # Encode categoricals - THIS IS NOT IDEAL FOR PRODUCTION
            # In a real production pipeline, we would use sklearn.pipeline.Pipeline with OneHotEncoder
            # to ensure consistency between training and inference.
            # However, to keep it simple as requested, we will use get_dummies but wrap it safely.
            # A better approach would be to move encoding to the training/inference pipeline.
            
            # For now, let's keep get_dummies as "feature engineering" but be aware of the drift risk.
            categorical_cols = [
                "Airline", "Source", "Destination", "Time_of_Day", 
                "Seasonality", "Aircraft Type", "Booking Source", "Season",
                "Route", "Duration_Category"
            ]
            # Only encode columns that actually exist
            cols_to_encode = [col for col in categorical_cols if col in df.columns]
            
            if cols_to_encode:
                df = pd.get_dummies(
                    df,
                    columns=cols_to_encode,
                    drop_first=True
                )
                logger.info(f"One-hot encoded columns: {cols_to_encode}")

        if scale:
            # Scale numerical features
            # Identify numerical columns to scale (exclude target and binary derived from OneHot)
            # Typically valid for: Day, Month, Weekday, Dep_Hour, Stopovers, Class_Encoded
            numeric_cols_to_scale = [
                "Day", "Month", "Weekday", "Dep_Hour", "Stopovers", "Class_Encoded",
                "Month_Sin", "Month_Cos", "Day_Sin", "Day_Cos", "Weekday_Sin", "Weekday_Cos", "Hour_Sin", "Hour_Cos"
            ]
            # Only scale columns that exist in the dataframe
            cols_to_scale = [col for col in numeric_cols_to_scale if col in df.columns]
            
            if cols_to_scale:
                scaler = StandardScaler()
                df[cols_to_scale] = scaler.fit_transform(df[cols_to_scale])
                logger.info(f"Scaled numerical features: {cols_to_scale}")

        logger.info(f"Feature engineering completed. Final shape: {df.shape}")
        return df
    except Exception as e:
        logger.error(f"Error during feature engineering: {e}")
        raise

