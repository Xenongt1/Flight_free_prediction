"""
Module for Exploratory Data Analysis (EDA).
"""
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import os
from src import config
from src.utils import get_logger

logger = get_logger(__name__)

def perform_eda(df: pd.DataFrame):
    """
    Perform comprehensive EDA: Descriptive stats, Visuals, and KPIs.
    
    Args:
        df (pd.DataFrame): Dataframe with engineered features (no encoding needed for visuals).
    """
    try:
        logger.info("Starting Exploratory Data Analysis...")
        
        # Ensure plots directory exists
        os.makedirs(config.PLOTS_DIR, exist_ok=True)
        
        # 1. Descriptive Statistics
        _generate_descriptive_stats(df)
        
        # 2. Visual Analysis
        _generate_visual_analysis(df)
        
        # 3. KPI Exploration
        _explore_kpis(df)
        
        logger.info("EDA completed successfully.")
    except Exception as e:
        logger.error(f"Error during EDA: {e}")
        # Don't raise, just log error so pipeline can continue if this is optional
        # But if critical, raise. Let's raise to be safe.
        raise

def _generate_descriptive_stats(df):
    logger.info("Generating descriptive statistics...")
    
    # Summarize fares by groups
    groups = ['Airline', 'Source', 'Destination', 'Season']
    target = 'Total Fare (BDT)'
    
    for group in groups:
        if group in df.columns:
            logger.info(f"Fare summary by {group}:")
            summary = df.groupby(group)[target].describe()
            # print(f"\n--- Fare Summary by {group} ---\n{summary}")
            summary.to_csv(os.path.join(config.PLOTS_DIR, f"stats_fare_by_{group}.csv"))

    # Correlations
    # Select only numeric columns
    numeric_df = df.select_dtypes(include=['number'])
    if not numeric_df.empty:
        corr = numeric_df.corr()
        # print(f"\n--- Correlation Matrix ---\n{corr}")
        corr.to_csv(os.path.join(config.PLOTS_DIR, "correlation_matrix.csv"))

def _generate_visual_analysis(df):
    logger.info("Generating visual analysis...")
    
    target = 'Total Fare (BDT)'
    
    # Set style
    sns.set_theme(style="whitegrid")
    
    # 1. Distributions
    # Check which columns exist
    cols_to_plot = [target, 'Base Fare (BDT)', 'Tax & Surcharge (BDT)']
    cols_present = [c for c in cols_to_plot if c in df.columns]
    
    if cols_present:
        fig, axes = plt.subplots(1, len(cols_present), figsize=(6 * len(cols_present), 5))
        if len(cols_present) == 1:
            axes = [axes]
            
        for i, col in enumerate(cols_present):
            sns.histplot(df[col], kde=True, ax=axes[i])
            axes[i].set_title(f'Distribution of {col}')
        
        plt.tight_layout()
        plt.savefig(os.path.join(config.PLOTS_DIR, "distributions.png"))
        plt.close()
    
    # 2. Boxplot: Fare Variation across Airlines
    if 'Airline' in df.columns:
        plt.figure(figsize=(12, 6))
        sns.boxplot(x='Airline', y=target, data=df)
        plt.xticks(rotation=45)
        plt.title('Fare Variation across Airlines')
        plt.tight_layout()
        plt.savefig(os.path.join(config.PLOTS_DIR, "boxplot_fare_by_airline.png"))
        plt.close()
        
    # 3. Average Fare by Month and Season
    if 'Month' in df.columns:
        plt.figure(figsize=(10, 5))
        sns.barplot(x='Month', y=target, data=df, estimator='mean', errorbar=None)
        plt.title('Average Fare by Month')
        plt.savefig(os.path.join(config.PLOTS_DIR, "avg_fare_by_month.png"))
        plt.close()
        
    if 'Season' in df.columns:
        plt.figure(figsize=(8, 5))
        sns.barplot(x='Season', y=target, data=df, estimator='mean', errorbar=None)
        plt.title('Average Fare by Season')
        plt.savefig(os.path.join(config.PLOTS_DIR, "avg_fare_by_season.png"))
        plt.close()

    # 4. Correlation Heatmap
    numeric_df = df.select_dtypes(include=['number'])
    if not numeric_df.empty:
        plt.figure(figsize=(12, 10))
        sns.heatmap(numeric_df.corr(), annot=True, cmap='coolwarm', fmt=".2f")
        plt.title('Correlation Heatmap')
        plt.tight_layout()
        plt.savefig(os.path.join(config.PLOTS_DIR, "correlation_heatmap.png"))
        plt.close()

def _explore_kpis(df):
    logger.info("Exploring KPIs...")
    target = 'Total Fare (BDT)'
    
    kpi_file = os.path.join(config.PLOTS_DIR, "kpis.txt")
    
    with open(kpi_file, 'w') as f:
        # 1. Avg fare per airline
        if 'Airline' in df.columns:
            avg_fare = df.groupby('Airline')[target].mean().sort_values(ascending=False)
            f.write(f"\n--- Average Fare per Airline ---\n{avg_fare}\n")
            logger.info("Calculated Average Fare per Airline")
        
        # 2. Most popular route
        if 'Source' in df.columns and 'Destination' in df.columns:
            popular_routes = df.groupby(['Source', 'Destination']).size().sort_values(ascending=False)
            f.write(f"\n--- Most Popular Routes ---\n{popular_routes.head(5)}\n")
            logger.info("Calculated Most Popular Routes")
            
        # 3. Seasonal fare variation
        if 'Season' in df.columns:
            seasonal_fare = df.groupby('Season')[target].mean().sort_values(ascending=False)
            f.write(f"\n--- Seasonal Fare Variation ---\n{seasonal_fare}\n")
            logger.info("Calculated Seasonal Fare Variation")
            
        # 4. Top 5 most expensive routes
        if 'Source' in df.columns and 'Destination' in df.columns:
            expensive_routes = df.groupby(['Source', 'Destination'])[target].mean().sort_values(ascending=False)
            f.write(f"\n--- Top 5 Most Expensive Routes ---\n{expensive_routes.head(5)}\n")
            logger.info("Calculated Top 5 Most Expensive Routes")
