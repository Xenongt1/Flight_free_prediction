"""
Streamlit Web Application for Flight Fare Prediction
Interactive UI for predicting flight fares in Bangladesh
"""
import streamlit as st
import requests
import pandas as pd
from datetime import datetime, timedelta
import json

# Page configuration
st.set_page_config(
    page_title="Flight Fare Predictor",
    page_icon="✈️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
<style>
    .main-header {
        font-size: 3rem;
        font-weight: bold;
        color: #1E88E5;
        text-align: center;
        margin-bottom: 1rem;
    }
    .sub-header {
        font-size: 1.2rem;
        color: #666;
        text-align: center;
        margin-bottom: 2rem;
    }
    .prediction-box {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 2rem;
        border-radius: 10px;
        color: white;
        text-align: center;
        margin: 2rem 0;
    }
    .prediction-value {
        font-size: 3rem;
        font-weight: bold;
        margin: 1rem 0;
    }
    .info-box {
        background-color: #f0f2f6;
        padding: 1rem;
        border-radius: 5px;
        margin: 1rem 0;
    }
    .stButton>button {
        width: 100%;
        background-color: #1E88E5;
        color: white;
        font-size: 1.2rem;
        padding: 0.75rem;
        border-radius: 5px;
    }
</style>
""", unsafe_allow_html=True)

# API Configuration
API_URL = "http://localhost:8000"

# Airlines list (from your data)
AIRLINES = [
    "Biman Bangladesh Airlines", "US-Bangla Airlines", "NovoAir", "Air Astra",
    "Jet Airways", "Air India", "IndiGo", "Vistara", "AirAsia",
    "Singapore Airlines", "Thai Airways", "Malaysian Airlines",
    "Emirates", "Qatar Airways", "Etihad Airways", "Gulf Air",
    "Kuwait Airways", "Saudia", "Turkish Airlines", "Lufthansa",
    "British Airways", "Air Arabia", "FlyDubai", "SriLankan Airlines",
    "Cathay Pacific"
]

# Cities in Bangladesh and popular destinations
CITIES = [
    "Dhaka", "Chittagong", "Sylhet", "Cox's Bazar", "Rajshahi",
    "Khulna", "Barisal", "Rangpur", "Jessore", "Saidpur",
    # International
    "Dubai", "Singapore", "Bangkok", "Kuala Lumpur", "Delhi",
    "Mumbai", "Kolkata", "London", "New York", "Toronto"
]

# Aircraft types
AIRCRAFT_TYPES = [
    "Boeing 737", "Boeing 777", "Boeing 787", "Airbus A320",
    "Airbus A330", "Airbus A350", "ATR 72"
]

# Header
st.markdown('<div class="main-header">✈️ Flight Fare Predictor</div>', unsafe_allow_html=True)
st.markdown('<div class="sub-header">Predict flight fares in Bangladesh using Machine Learning</div>', unsafe_allow_html=True)

# Sidebar - API Status
with st.sidebar:
    st.header("🔧 System Status")
    
    try:
        response = requests.get(f"{API_URL}/health", timeout=2)
        if response.status_code == 200:
            st.success("✅ API Connected")
            st.json(response.json())
        else:
            st.error("❌ API Error")
    except Exception as e:
        st.error("❌ API Offline")
        st.info("Make sure the API is running:\n```bash\ndocker ps\n```")
    
    st.markdown("---")
    st.header("📊 Model Info")
    st.info("""
    **Algorithm:** Gradient Boosting
    
    **Performance:**
    - R² Score: 0.643
    - MAE: 28,707 BDT
    - RMSE: 48,795 BDT
    
    **Features:** 50+ engineered features including routes, seasons, and cyclical time patterns
    """)

# Main content
col1, col2 = st.columns(2)

with col1:
    st.subheader("🛫 Flight Details")
    
    # Airline
    airline = st.selectbox(
        "Airline",
        AIRLINES,
        index=AIRLINES.index("Biman Bangladesh Airlines")
    )
    
    # Route
    source = st.selectbox(
        "Departure City",
        CITIES,
        index=CITIES.index("Dhaka")
    )
    
    destination = st.selectbox(
        "Arrival City",
        [city for city in CITIES if city != source],
        index=0
    )
    
    # Date
    travel_date = st.date_input(
        "Travel Date",
        value=datetime.now() + timedelta(days=30),
        min_value=datetime.now().date(),
        max_value=datetime.now().date() + timedelta(days=365)
    )
    
    # Duration
    duration = st.slider(
        "Flight Duration (hours)",
        min_value=0.5,
        max_value=15.0,
        value=1.5,
        step=0.5
    )

with col2:
    st.subheader("🎫 Booking Details")
    
    # Class
    travel_class = st.selectbox(
        "Travel Class",
        ["Economy", "Business", "First Class"],
        index=0
    )
    
    # Stopovers
    stopovers = st.selectbox(
        "Stopovers",
        ["Direct", "1 Stop", "2 Stops", "3 Stops"],
        index=0
    )
    
    # Aircraft Type
    aircraft = st.selectbox(
        "Aircraft Type",
        AIRCRAFT_TYPES,
        index=0
    )
    
    # Booking Source
    booking_source = st.selectbox(
        "Booking Source",
        ["Online", "Travel Agent", "Airline Counter"],
        index=0
    )

# Predict button
st.markdown("---")
if st.button("🔮 Predict Fare", use_container_width=True):
    # Prepare request data
    request_data = {
        "Airline": airline,
        "Date": travel_date.strftime("%Y-%m-%d"),
        "Source": source,
        "Destination": destination,
        "Stopovers": stopovers,
        "Class": travel_class,
        "Duration_hrs": duration,
        "Aircraft_Type": aircraft,
        "Booking_Source": booking_source
    }
    
    # Show loading spinner
    with st.spinner("🤖 Analyzing flight data..."):
        try:
            # Make API request
            response = requests.post(
                f"{API_URL}/predict",
                json=request_data,
                timeout=5
            )
            
            if response.status_code == 200:
                result = response.json()
                prediction = result['prediction']
                
                # Display prediction
                st.markdown(f"""
                <div class="prediction-box">
                    <h2>Predicted Fare</h2>
                    <div class="prediction-value">৳ {prediction:,.2f}</div>
                    <p>Bangladeshi Taka (BDT)</p>
                    <p style="font-size: 0.9rem; opacity: 0.8;">Predicted at {result['timestamp']}</p>
                </div>
                """, unsafe_allow_html=True)
                
                # Additional insights
                st.subheader("📈 Fare Breakdown & Insights")
                
                col1, col2, col3 = st.columns(3)
                
                with col1:
                    st.metric(
                        "Base Estimate",
                        f"৳ {prediction * 0.85:,.2f}",
                        help="Estimated base fare (85% of total)"
                    )
                
                with col2:
                    st.metric(
                        "Taxes & Fees",
                        f"৳ {prediction * 0.15:,.2f}",
                        help="Estimated taxes and surcharges (15% of total)"
                    )
                
                with col3:
                    # Price category
                    if prediction < 40000:
                        category = "Budget"
                        color = "🟢"
                    elif prediction < 80000:
                        category = "Standard"
                        color = "🟡"
                    else:
                        category = "Premium"
                        color = "🔴"
                    
                    st.metric(
                        "Price Category",
                        f"{color} {category}"
                    )
                
                # Tips
                st.markdown('<div class="info-box">', unsafe_allow_html=True)
                st.markdown("### 💡 Money-Saving Tips")
                
                tips = []
                
                # Season tip
                month = travel_date.month
                if month in [12, 1, 2]:
                    tips.append("🌨️ **Winter Travel:** Fares are typically 16% higher in winter. Consider traveling in autumn for better prices.")
                elif month in [9, 10, 11]:
                    tips.append("🍂 **Autumn Savings:** You're traveling in the cheapest season! Good choice.")
                
                # Class tip
                if travel_class == "First Class":
                    tips.append("👑 **First Class:** You could save ~60% by choosing Business Class with similar comfort.")
                elif travel_class == "Economy":
                    tips.append("💰 **Economy Choice:** Great value! You're getting the best price-to-service ratio.")
                
                # Stopovers tip
                if stopovers == "Direct":
                    tips.append("⚡ **Direct Flight:** Fastest option, though flights with 1 stop might be cheaper.")
                else:
                    tips.append("🔄 **Connecting Flight:** You might save money, but consider the extra travel time.")
                
                # Booking source tip
                if booking_source == "Online":
                    tips.append("💻 **Online Booking:** Often the cheapest option. Check multiple platforms for best deals.")
                
                for tip in tips:
                    st.markdown(tip)
                
                st.markdown('</div>', unsafe_allow_html=True)
                
                # Show request data
                with st.expander("🔍 View Request Details"):
                    st.json(request_data)
                
            else:
                st.error(f"❌ Prediction failed: {response.text}")
                
        except requests.exceptions.ConnectionError:
            st.error("❌ Cannot connect to API. Make sure the Docker container is running.")
            st.code("docker ps --filter 'name=flight-api'")
        except Exception as e:
            st.error(f"❌ Error: {str(e)}")

# Footer
st.markdown("---")
st.markdown("""
<div style="text-align: center; color: #666; padding: 2rem;">
    <p>Built with ❤️ using Streamlit & FastAPI</p>
    <p>Model trained on Bangladesh flight data | Accuracy: R² = 0.643</p>
</div>
""", unsafe_allow_html=True)

# Sidebar - Quick Examples
with st.sidebar:
    st.markdown("---")
    st.header("🎯 Quick Examples")
    
    if st.button("Budget Domestic"):
        st.info("Dhaka → Chittagong\nEconomy, Direct\n~৳38,000")
    
    if st.button("Business Trip"):
        st.info("Dhaka → Sylhet\nBusiness, 1 Stop\n~৳78,000")
    
    if st.button("Premium International"):
        st.info("Chittagong → Dubai\nFirst Class\n~৳145,000")
