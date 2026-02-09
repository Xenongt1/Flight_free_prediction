"""
Module for Exploratory Data Analysis (EDA).
"""

def perform_eda(df):
    """
    Perform EDA and print basic stats.
    """
    print(df.info())
    print(df.describe())
