
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
import pandas as pd
import joblib
import os
import sys
from datetime import datetime
from src import config
from src.utils import get_logger

# Add project root to sys.path to ensure custom transformers can be unpickled
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

logger = get_logger(__name__)

app = FastAPI(
    title="Flight Fare Prediction API",
    description="API to predict flight fares in Bangladesh using a trained ML pipeline.",
    version="1.0.0"
)

# Load the model pipeline at startup
# Note: Since we saved the 'Pipeline' object, it contains everything:
# feature generation, encoding, scaling, and the regressor.
MODEL_PATH = config.MODEL_PATH

try:
    if os.path.exists(MODEL_PATH):
        model = joblib.load(MODEL_PATH)
        logger.info(f"Model loaded successfully from {MODEL_PATH}")
    else:
        logger.warning(f"Model file not found at {MODEL_PATH}. Prediction endpoint will return 503.")
        model = None
except Exception as e:
    logger.error(f"Error loading model: {e}")
    model = None

class FlightRequest(BaseModel):
    Airline: str = Field(..., example="Jet Airways")
    Date: str = Field(..., description="Format: YYYY-MM-DD", example="2024-03-15")
    Source: str = Field(..., example="Dhaka")
    Destination: str = Field(..., example="Chittagong")
    Stopovers: str = Field(..., example="Direct")
    Class: str = Field(..., example="Economy")
    Duration_hrs: float = Field(..., alias="Duration (hrs)", example=1.5)
    Aircraft_Type: str = Field(..., alias="Aircraft Type", example="Boeing 737")
    Booking_Source: str = Field(..., alias="Booking Source", example="Travel Agent")

    class Config:
        populate_by_name = True

class PredictionResponse(BaseModel):
    prediction: float
    currency: str = "BDT"
    timestamp: str

@app.get("/")
def read_root():
    return {"message": "Welcome to the Flight Fare Prediction API", "status": "online"}

@app.get("/health")
def health_check():
    if model is None:
        raise HTTPException(status_code=503, detail="Model not loaded")
    return {"status": "healthy", "model_loaded": True}

@app.post("/predict", response_model=PredictionResponse)
def predict(request: FlightRequest):
    if model is None:
        raise HTTPException(status_code=503, detail="Model is not available for prediction")
    
    try:
        # 1. Convert request to DataFrame
        # The pipeline expects raw data columns matching the original training format
        input_data = {
            "Airline": [request.Airline],
            "Date": [request.Date],
            "Source": [request.Source],
            "Destination": [request.Destination],
            "Stopovers": [request.Stopovers],
            "Class": [request.Class],
            "Duration (hrs)": [request.Duration_hrs],
            "Aircraft Type": [request.Aircraft_Type],
            "Booking Source": [request.Booking_Source]
        }
        
        df = pd.DataFrame(input_data)
        
        # 2. Convert Date string to datetime as the Custom Transformer expects it
        df["Date"] = pd.to_datetime(df["Date"])
        
        # 3. Predict using the FULL pipeline
        # The pipeline handles all feature engineering, encoding, and scaling internally
        prediction = model.predict(df)[0]
        
        # Handle negative predictions (rare but possible with some regressor types)
        prediction = max(0, float(prediction))
        
        return PredictionResponse(
            prediction=round(prediction, 2),
            timestamp=datetime.now().isoformat()
        )
    except Exception as e:
        logger.error(f"Prediction error: {e}")
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
