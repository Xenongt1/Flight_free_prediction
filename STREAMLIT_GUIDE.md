# Streamlit Web App - Quick Start Guide

## 🚀 Your Streamlit App is Running!

**Access the app at:** http://localhost:8501

---

## ✨ Features

### 1. **Interactive Flight Fare Prediction**
- Beautiful, user-friendly interface
- Real-time predictions from your ML API
- Instant fare estimates with detailed breakdowns

### 2. **Smart Insights**
- Fare breakdown (base fare + taxes)
- Price category classification (Budget/Standard/Premium)
- Money-saving tips based on your selections
- Seasonal pricing insights

### 3. **System Status Dashboard**
- Live API health monitoring
- Model performance metrics
- Connection status indicators

### 4. **Quick Examples**
- Pre-configured flight scenarios
- Budget domestic flights
- Business trips
- Premium international travel

---

## 🎯 How to Use

1. **Select Flight Details** (Left Column):
   - Choose your airline
   - Select departure and arrival cities
   - Pick your travel date
   - Set flight duration

2. **Choose Booking Options** (Right Column):
   - Travel class (Economy/Business/First)
   - Number of stopovers
   - Aircraft type
   - Booking source

3. **Get Prediction**:
   - Click "🔮 Predict Fare"
   - View your predicted fare
   - Read personalized money-saving tips

---

## 📊 What You'll See

### Prediction Display
- **Large fare amount** in Bangladeshi Taka (BDT)
- **Timestamp** of prediction
- **Beautiful gradient card** design

### Insights Section
- **Base Estimate**: ~85% of total fare
- **Taxes & Fees**: ~15% of total fare
- **Price Category**: Budget 🟢 / Standard 🟡 / Premium 🔴

### Money-Saving Tips
- Seasonal travel advice
- Class comparison suggestions
- Booking source recommendations
- Stopover vs direct flight analysis

---

## 🛠️ Technical Details

### Architecture
```
User Browser (localhost:8501)
    ↓
Streamlit App (streamlit_app.py)
    ↓
FastAPI Backend (localhost:8000)
    ↓
ML Model Pipeline
    ↓
Prediction Result
```

### API Integration
- Connects to your FastAPI service on port 8000
- Real-time health checks
- Error handling with user-friendly messages
- Timeout protection (5 seconds)

---

## 🎨 UI Components

### Custom Styling
- **Gradient prediction cards** with purple theme
- **Responsive layout** with 2-column design
- **Color-coded metrics** for quick insights
- **Professional typography** and spacing

### Interactive Elements
- **Dropdown menus** for all selections
- **Date picker** with validation
- **Slider** for flight duration
- **Expandable sections** for details

---

## 🔧 Commands

### Start the Streamlit App
```bash
streamlit run streamlit_app.py
```

### Stop the App
Press `Ctrl+C` in the terminal

### Restart the App
```bash
# Stop with Ctrl+C, then run again
streamlit run streamlit_app.py
```

### Clear Cache (if needed)
```bash
streamlit cache clear
```

---

## 📱 Access Options

### Local Access
- **URL:** http://localhost:8501
- **Use:** Your computer only

### Network Access
- **URL:** http://192.168.6.22:8501
- **Use:** Other devices on your network
- **Note:** Make sure firewall allows port 8501

---

## 🐛 Troubleshooting

### "API Offline" Error
**Problem:** Streamlit can't connect to FastAPI
**Solution:**
```bash
# Check if API container is running
docker ps --filter "name=flight-api"

# If not running, start it
docker run -d -p 8000:8000 --env-file .env --name flight-api-new flight-api
```

### Port Already in Use
**Problem:** Port 8501 is occupied
**Solution:**
```bash
# Use a different port
streamlit run streamlit_app.py --server.port 8502
```

### Module Not Found
**Problem:** Missing dependencies
**Solution:**
```bash
pip install streamlit requests
```

---

## 🎯 Example Predictions

### Budget Domestic Flight
- **Route:** Dhaka → Chittagong
- **Class:** Economy
- **Stopovers:** Direct
- **Expected:** ~৳38,000 BDT

### Business Trip
- **Route:** Dhaka → Sylhet
- **Class:** Business
- **Stopovers:** 1 Stop
- **Expected:** ~৳78,000 BDT

### Premium International
- **Route:** Chittagong → Dubai
- **Class:** First Class
- **Stopovers:** Direct
- **Expected:** ~৳145,000 BDT

---

## 📈 Performance

- **Response Time:** < 1 second
- **Prediction Accuracy:** R² = 0.643
- **MAE:** 28,707 BDT
- **RMSE:** 48,795 BDT

---

## 🎨 Customization

### Change Theme
Edit `.streamlit/config.toml`:
```toml
[theme]
primaryColor = "#1E88E5"
backgroundColor = "#FFFFFF"
secondaryBackgroundColor = "#F0F2F6"
textColor = "#262730"
font = "sans serif"
```

### Modify API URL
Edit `streamlit_app.py` line 52:
```python
API_URL = "http://your-api-url:8000"
```

---

## 🚀 Next Steps

### Deploy to Cloud
1. **Streamlit Cloud** (Free):
   - Push to GitHub
   - Connect at share.streamlit.io
   - Deploy with one click

2. **Heroku**:
   - Add `Procfile`
   - Deploy with Heroku CLI

3. **AWS/GCP/Azure**:
   - Containerize with Docker
   - Deploy to cloud service

### Add Features
- Historical price charts
- Multi-city comparisons
- Price alerts
- Booking integration
- User accounts
- Saved searches

---

## 📝 Files

- `streamlit_app.py` - Main application
- `requirements.txt` - Updated with streamlit
- `app.py` - FastAPI backend (must be running)

---

## ✅ Checklist

- [x] Streamlit installed
- [x] App created with beautiful UI
- [x] API integration working
- [x] Real-time predictions
- [x] Money-saving tips
- [x] System status monitoring
- [x] Error handling
- [x] Responsive design

---

**Enjoy your beautiful flight fare prediction app!** ✈️

Access it now at: **http://localhost:8501**
