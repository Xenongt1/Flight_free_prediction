"""
Module for feature engineering using Scikit-Learn Pipelines.
"""

import pandas as pd
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from src.utils import get_logger
from src.transformers import (
    DateFeatureEngineer, 
    TimeOfDayEngineer, 
    CyclicalFeatureEngineer, 
    RouteEngineer, 
    DurationCategoryEngineer, 
    SeasonEngineer, 
    CustomOrdinalEncoder,
    Columndropper
)

logger = get_logger(__name__)

def create_preprocessing_pipeline(scale_numeric: bool = True):
    """
    Creates a Scikit-Learn Pipeline for all feature engineering and preprocessing.
    
    Args:
        scale_numeric (bool): Whether to scale numerical features.
        
    Returns:
        sklearn.pipeline.Pipeline: The complete preprocessing pipeline.
    """
    logger.info("Creating preprocessing pipeline...")
    
    # Define categorical and numerical columns for the preprocessor
    categorical_cols = [
        "Airline", "Source", "Destination", "Time_of_Day", 
        "Aircraft Type", "Season", "Duration_Category"
    ]
    
    numeric_cols = [
        "Day", "Month", "Weekday", "Dep_Hour", "Stopovers", "Class_Encoded",
        "Month_Sin", "Month_Cos", "Day_Sin", "Day_Cos", 
        "Weekday_Sin", "Weekday_Cos", "Dep_Hour_Sin", "Dep_Hour_Cos"
    ]
    
    preprocessor = ColumnTransformer(
        transformers=[
            ('cat', OneHotEncoder(sparse_output=False, handle_unknown='ignore'), categorical_cols),
            ('num', StandardScaler() if scale_numeric else 'passthrough', numeric_cols)
        ],
        remainder='drop', 
        verbose_feature_names_out=False
    )
    
    # Return a FLAT pipeline to avoid NotFittedError issues with nesting
    return Pipeline([
        ('date_engineer', DateFeatureEngineer(date_col='Date')),
        ('time_of_day', TimeOfDayEngineer(hour_col='Dep_Hour')),
        ('cyclical', CyclicalFeatureEngineer()),
        ('route', RouteEngineer()),
        ('duration_cat', DurationCategoryEngineer(duration_col='Duration (hrs)')),
        ('season', SeasonEngineer()),
        ('ordinal', CustomOrdinalEncoder()),
        ('preprocessor', preprocessor)
    ])

def engineer_features(df: pd.DataFrame, encode: bool = True, scale: bool = False) -> pd.DataFrame:
    """
    Apply feature engineering to a DataFrame.
    
    Args:
        df (pd.DataFrame): Input dataframe.
        encode (bool): Whether to apply categorical encoding (OneHot). 
                      If False, returns a DataFrame with human-readable features (good for EDA).
        scale (bool): Whether to scale numeric features.
        
    Returns:
        pd.DataFrame: Transformed dataframe.
    """
    # 1. Always create the feature generation part
    feature_generation = Pipeline([
        ('date_engineer', DateFeatureEngineer(date_col='Date')),
        ('time_of_day', TimeOfDayEngineer(hour_col='Dep_Hour')),
        ('cyclical', CyclicalFeatureEngineer()),
        ('route', RouteEngineer()),
        ('duration_cat', DurationCategoryEngineer(duration_col='Duration (hrs)')),
        ('season', SeasonEngineer()),
        ('ordinal', CustomOrdinalEncoder()),
    ])
    
    # Apply feature generation
    df_transformed = feature_generation.fit_transform(df)
    
    if not encode:
        # Return the feature-engineered but NOT encoded DataFrame
        # This is strictly for EDA as it preserves all columns including target
        return df_transformed
        
    # 2. If encode is True, apply the full pipeline including ColumnTransformer
    pipeline = create_preprocessing_pipeline(scale_numeric=scale)
    
    # We use the full pipeline on the original df
    array_output = pipeline.fit_transform(df)
    
    # Get feature names from the preprocessor
    preprocessor = pipeline.named_steps['preprocessor']
    feature_names = preprocessor.get_feature_names_out()
    
    return pd.DataFrame(array_output, columns=feature_names)


