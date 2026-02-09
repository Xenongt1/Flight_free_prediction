"""
Configuration settings for the project.
"""
import os

# Paths
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_PATH = os.path.join(BASE_DIR, "data", "raw", "Flight_Price_Dataset_of_Bangladesh copy.csv")
PROCESSED_DATA_PATH = os.path.join(BASE_DIR, "data", "processed", "clean_flight_data.csv")
MODEL_PATH = os.path.join(BASE_DIR, "models", "best_model.pkl")
LOG_FILE = os.path.join(BASE_DIR, "logs", "pipeline.log")

# Ensure logs directory exists
os.makedirs(os.path.dirname(LOG_FILE), exist_ok=True)

# Random Seed
RANDOM_SEED = 42

# Model Parameters
TEST_SIZE = 0.2
TARGET_COL = "Total Fare (BDT)"
