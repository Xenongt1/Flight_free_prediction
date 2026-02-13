
import pandas as pd
from src import config

try:
    df = pd.read_csv(config.DATA_PATH)
    print("Unique values for potential features:")
    for col in ['Stopovers', 'Class', 'Seasonality', 'Aircraft Type', 'Booking Source']:
        if col in df.columns:
            print(f"\n--- {col} ---")
            print(df[col].value_counts().head(10))
            print(f"Unique count: {df[col].nunique()}")
        else:
            print(f"\nColumn '{col}' not found.")
except Exception as e:
    print(f"Error: {e}")
