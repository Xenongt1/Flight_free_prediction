"""
Configuration settings for the project.
"""
import os
from dotenv import load_dotenv

load_dotenv()

# Paths
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_PATH = os.path.join(BASE_DIR, "data", "raw", "Flight_Price_Dataset_of_Bangladesh copy.csv")
EDA_DATA_PATH = os.path.join(BASE_DIR, "data", "processed", "eda_flight_data.csv")
PROCESSED_DATA_PATH = os.path.join(BASE_DIR, "data", "processed", "model_flight_data.csv")
MODEL_PATH = os.path.join(BASE_DIR, "models", "best_model.pkl")
METRICS_FILE = os.path.join(BASE_DIR, "reports", "metrics.json")
PLOTS_DIR = os.path.join(BASE_DIR, "reports", "figures")
LOG_FILE = os.path.join(BASE_DIR, "logs", "pipeline.log")

# Ensure directories exist
os.makedirs(os.path.dirname(LOG_FILE), exist_ok=True)
os.makedirs(os.path.dirname(METRICS_FILE), exist_ok=True)
os.makedirs(PLOTS_DIR, exist_ok=True)

# Random Seed
RANDOM_SEED = 42

# Database Configuration
DB_USER = os.getenv("DB_USER", "analytics_user")
DB_PASSWORD = os.getenv("DB_PASSWORD", "analytics_password")
DB_HOST = os.getenv("DB_HOST", "postgres_analytics")  # ✅ FIX: Container hostname
DB_PORT = os.getenv("DB_PORT", "5432")  # ✅ FIX: Container port (not host-mapped port)
DB_NAME = os.getenv("DB_NAME", "flight_analytics")

DB_CONNECTION_STRING = f"postgresql+psycopg2://{DB_USER}:{DB_PASSWORD}@{DB_HOST}:{DB_PORT}/{DB_NAME}"
DATA_SOURCE = os.getenv("DATA_SOURCE", "csv") # Options: 'csv', 'postgres'

# Model Parameters
TEST_SIZE = 0.2
TARGET_COL = "Total Fare (BDT)"

