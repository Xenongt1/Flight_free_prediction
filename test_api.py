"""
Test script for the Flight Fare Prediction API
"""
import requests
import json
from datetime import datetime, timedelta

API_URL = "http://localhost:8000"

def test_health():
    """Test the health endpoint"""
    print("=" * 60)
    print("Testing Health Endpoint")
    print("=" * 60)
    response = requests.get(f"{API_URL}/health")
    print(f"Status Code: {response.status_code}")
    print(f"Response: {json.dumps(response.json(), indent=2)}")
    print()

def test_prediction(flight_data, description):
    """Test a prediction with given flight data"""
    print("=" * 60)
    print(f"Test: {description}")
    print("=" * 60)
    print(f"Request Data: {json.dumps(flight_data, indent=2)}")
    
    response = requests.post(f"{API_URL}/predict", json=flight_data)
    print(f"Status Code: {response.status_code}")
    
    if response.status_code == 200:
        result = response.json()
        print(f"[OK] Prediction: {result['prediction']} {result['currency']}")
        print(f"Timestamp: {result['timestamp']}")
    else:
        print(f"[ERROR] {response.text}")
    print()

def main():
    print("\n>>> Flight Fare Prediction API Test Suite\n")
    
    # Test 1: Health check
    test_health()
    
    # Test 2: Economy class, direct flight
    test_prediction({
        "Airline": "Biman Bangladesh Airlines",
        "Date": "2024-06-15",
        "Source": "Dhaka",
        "Destination": "Chittagong",
        "Stopovers": "Direct",
        "Class": "Economy",
        "Duration_hrs": 1.0,
        "Aircraft_Type": "Boeing 737",
        "Booking_Source": "Online"
    }, "Economy Direct Flight - Dhaka to Chittagong")
    
    # Test 3: Business class, 1 stop
    test_prediction({
        "Airline": "US-Bangla Airlines",
        "Date": "2024-07-20",
        "Source": "Dhaka",
        "Destination": "Sylhet",
        "Stopovers": "1 Stop",
        "Class": "Business",
        "Duration_hrs": 2.5,
        "Aircraft_Type": "Boeing 777",
        "Booking_Source": "Travel Agent"
    }, "Business Class with 1 Stop - Dhaka to Sylhet")
    
    # Test 4: Long haul flight
    test_prediction({
        "Airline": "Jet Airways",
        "Date": "2024-08-10",
        "Source": "Dhaka",
        "Destination": "Cox's Bazar",
        "Stopovers": "2 Stops",
        "Class": "Economy",
        "Duration_hrs": 4.5,
        "Aircraft_Type": "Airbus A350",
        "Booking_Source": "Online"
    }, "Long Haul with 2 Stops - Dhaka to Cox's Bazar")
    
    # Test 5: First class
    test_prediction({
        "Airline": "Biman Bangladesh Airlines",
        "Date": "2024-09-05",
        "Source": "Chittagong",
        "Destination": "Dhaka",
        "Stopovers": "Direct",
        "Class": "First Class",
        "Duration_hrs": 1.0,
        "Aircraft_Type": "Boeing 787",
        "Booking_Source": "Travel Agent"
    }, "First Class Direct - Chittagong to Dhaka")
    
    # Test 6: Peak season (December)
    test_prediction({
        "Airline": "NovoAir",
        "Date": "2024-12-25",
        "Source": "Dhaka",
        "Destination": "Sylhet",
        "Stopovers": "Direct",
        "Class": "Economy",
        "Duration_hrs": 1.5,
        "Aircraft_Type": "Boeing 737",
        "Booking_Source": "Online"
    }, "Peak Season (December) - Dhaka to Sylhet")
    
    print("=" * 60)
    print("[SUCCESS] All tests completed!")
    print("=" * 60)

if __name__ == "__main__":
    main()
