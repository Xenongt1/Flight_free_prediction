
import joblib
import pandas as pd
import os
from src import config

MODEL_PATH = config.MODEL_PATH
if not os.path.exists(MODEL_PATH):
    print(f"Model not found at {MODEL_PATH}")
    exit(1)

model = joblib.load(MODEL_PATH)

# Test inputs
test_inputs = [
    {
        "Airline": "Biman Bangladesh Airlines",
        "Date": "2024-03-15",
        "Source": "Dhaka",
        "Destination": "Chittagong",
        "Stopovers": "Direct",
        "Class": "Economy",
        "Duration (hrs)": 1.0,
        "Aircraft Type": "Boeing 737",
        "Booking Source": "Online"
    },
    {
        "Airline": "US-Bangla Airlines",
        "Date": "2024-03-15",
        "Source": "Dhaka",
        "Destination": "Sylhet",
        "Stopovers": "Direct",
        "Class": "Business",
        "Duration (hrs)": 1.2,
        "Aircraft Type": "ATR 72",
        "Booking Source": "Travel Agent"
    },
    {
        "Airline": "Emirates",
        "Date": "2024-06-15",
        "Source": "Dhaka",
        "Destination": "Dubai",
        "Stopovers": "1 Stop",
        "Class": "First Class",
        "Duration (hrs)": 5.5,
        "Aircraft Type": "Boeing 777",
        "Booking Source": "Online"
    }
]

for i, input_data in enumerate(test_inputs):
    df = pd.DataFrame([input_data])
    df["Date"] = pd.to_datetime(df["Date"])
    prediction = model.predict(df)[0]
    print(f"Prediction {i+1}: {prediction:,.2f} BDT")
