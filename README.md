# ✈️ Flight Fare Prediction - Bangladesh

> **A production-ready machine learning system for predicting flight fares with automated retraining, REST API deployment, and comprehensive analytics.**

[![Python](https://img.shields.io/badge/Python-3.8%2B-blue)](https://www.python.org/)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.68%2B-green)](https://fastapi.tiangolo.com/)
[![Docker](https://img.shields.io/badge/Docker-Ready-blue)](https://www.docker.com/)
[![Airflow](https://img.shields.io/badge/Airflow-Integrated-orange)](https://airflow.apache.org/)

---

## 📋 Table of Contents
- [Project Overview](#-project-overview)
- [Key Deliverables](#-key-deliverables)
- [Architecture](#-architecture)
- [Features](#-features)
- [Installation & Setup](#-installation--setup)
- [Usage](#-usage)
- [API Documentation](#-api-documentation)
- [Model Performance](#-model-performance)
- [Project Structure](#-project-structure)
- [Stretch Challenges](#-stretch-challenges-all-completed)
- [Documentation](#-documentation)

---

## 🎯 Project Overview

### Business Goal
Airlines and travel platforms in Bangladesh need accurate fare predictions to:
- Optimize dynamic pricing strategies
- Provide customers with fare estimates
- Identify seasonal pricing patterns
- Analyze route profitability

### ML Task
- **Type**: Supervised Regression
- **Target Variable**: Total Fare (BDT)
- **Dataset**: Flight Price Dataset of Bangladesh (10,000+ records)
- **Best Model**: Gradient Boosting Regressor with hyperparameter tuning
- **Performance**: R² = 0.6428, MAE = 28,707 BDT, RMSE = 48,794 BDT

---

## 📦 Key Deliverables

### ✅ 1. Complete ML Pipeline
- **Data Loading & Validation** (`src/data_loader.py`)
- **Data Cleaning & Preprocessing** (`src/preprocessing.py`)
  - Missing value imputation
  - Outlier removal (>99th percentile)
  - City name normalization (Dacca → Dhaka)
  - Data type validation
- **Advanced Feature Engineering** (`src/feature_engineering.py`, `src/transformers.py`)
  - 6 new temporal features (Month, Day, Weekday, Season, Time_of_Day, Duration_Category)
  - 8 cyclical features (Month_Sin/Cos, Day_Sin/Cos, Weekday_Sin/Cos, Dep_Hour_Sin/Cos)
  - Route combination features
  - One-hot encoding for categorical variables
  - Ordinal encoding for Stopovers and Class
  - StandardScaler for numerical features

### ✅ 2. Comprehensive EDA
- **15 Visualizations** in `reports/figures/`:
  - Distribution plots
  - Boxplots by airline, season, route
  - Correlation heatmaps
  - Feature importance charts
  - Actual vs Predicted scatter plots
  - Residual analysis
- **Statistical Analysis**:
  - Fare statistics by airline, source, destination, season
  - Correlation matrices
  - KPI calculations

### ✅ 3. Multiple ML Models
Implemented and compared **6 different models**:
1. **Linear Regression** (Baseline)
2. **Ridge Regression** (L2 regularization)
3. **Lasso Regression** (L1 regularization)
4. **Decision Tree Regressor**
5. **Random Forest Regressor**
6. **Gradient Boosting Regressor** ⭐ (Best Model)

### ✅ 4. Hyperparameter Optimization
- **GridSearchCV** with 3-fold cross-validation
- Tuned parameters:
  - `n_estimators`: [100, 200]
  - `learning_rate`: [0.05, 0.1]
  - `max_depth`: [3, 5]
  - `subsample`: [0.8, 1.0]
- **Best Configuration**: `learning_rate=0.05`, `max_depth=3`, `n_estimators=200`, `subsample=0.8`

### ✅ 5. Production REST API
- **FastAPI** application (`app.py`)
- **Docker containerized** (`Dockerfile.api`)
- **Endpoints**:
  - `POST /predict` - Single prediction
  - `POST /predict/batch` - Batch predictions
  - `GET /health` - Health check
  - `GET /model/info` - Model metadata
- **Comprehensive tests** (`test_api.py`)
- **Running on port 8000**

### ✅ 6. Streamlit Web Application
- Interactive UI for fare predictions (`streamlit_app.py`)
- Real-time predictions via API integration
- User-friendly input forms
- Visual prediction results

### ✅ 7. Airflow Integration
- **Automated ETL pipeline** (`dags/flight_price_dag.py`)
- **Smart retraining logic**:
  - Monitors new data volume
  - Triggers retraining when threshold reached
  - Prevents duplicate data ingestion
- **PostgreSQL integration** for data persistence
- **Scheduled execution** with configurable intervals

### ✅ 8. Comprehensive Documentation
- **README.md** - Project overview (this file)
- **MODEL_REPORT.md** - Detailed model analysis and insights
- **API_DEPLOYMENT_SUMMARY.md** - API deployment guide
- **STREAMLIT_GUIDE.md** - Streamlit app documentation
- **PROJECT_COMPLETION_CHECKLIST.md** - Complete project verification
- **Jupyter Notebooks**:
  - `01_exploration_and_results.ipynb` - Full EDA
  - `eda_walkthrough.ipynb` - EDA tutorial

---

## 🏗️ Architecture

```
┌─────────────────┐
│  Raw Data (CSV) │
└────────┬────────┘
         │
         ▼
┌─────────────────────────┐
│  Data Cleaning          │
│  - Missing values       │
│  - Outliers             │
│  - Normalization        │
└────────┬────────────────┘
         │
         ▼
┌─────────────────────────┐
│  Feature Engineering    │
│  - Temporal features    │
│  - Cyclical encoding    │
│  - One-hot encoding     │
│  - Scaling              │
└────────┬────────────────┘
         │
         ▼
┌─────────────────────────┐
│  Model Training         │
│  - Multiple models      │
│  - GridSearchCV         │
│  - Cross-validation     │
└────────┬────────────────┘
         │
         ▼
┌─────────────────────────┐
│  Model Evaluation       │
│  - R², MAE, RMSE        │
│  - Visualizations       │
│  - Feature importance   │
└────────┬────────────────┘
         │
         ▼
┌─────────────────────────┐
│  Deployment             │
│  - FastAPI REST API     │
│  - Docker Container     │
│  - Streamlit UI         │
└─────────────────────────┘
         │
         ▼
┌─────────────────────────┐
│  Airflow Automation     │
│  - Scheduled ETL        │
│  - Smart Retraining     │
│  - PostgreSQL Storage   │
└─────────────────────────┘
```

---

## 🚀 Features

### Advanced Feature Engineering
- **Temporal Features**: Month, Day, Weekday, Season, Time_of_Day
- **Cyclical Encoding**: Sine/Cosine transformations for temporal continuity
- **Route Features**: Source-Destination combinations
- **Duration Categories**: Short (<5h), Medium (5-10h), Long (>10h)
- **Ordinal Encoding**: Stopovers (Direct=0, 1 Stop=1, etc.), Class (Economy=1, Business=2, First=3)

### Model Optimization
- **Log Transformation**: Applied to target variable for skewed distribution
- **Outlier Handling**: Removed top 1% of fares (>334k BDT)
- **Regularization**: Ridge and Lasso for overfitting prevention
- **Cross-Validation**: 3-fold CV for robust evaluation
- **Hyperparameter Tuning**: GridSearchCV for optimal parameters

### Production Features
- **Dockerized API**: Easy deployment and scaling
- **Health Monitoring**: Built-in health check endpoints
- **Batch Predictions**: Support for multiple predictions
- **Error Handling**: Comprehensive validation and error messages
- **Automated Testing**: Full test suite for API endpoints

---

## 💻 Installation & Setup

### Prerequisites
- Python 3.8+
- Docker (optional, for containerized deployment)
- PostgreSQL (optional, for Airflow integration)

### Local Setup

1. **Clone the repository**
```bash
git clone https://github.com/Xenongt1/Flight_free_prediction.git
cd flight-fare-prediction
```

2. **Create virtual environment**
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

3. **Install dependencies**
```bash
pip install -r requirements.txt
```

4. **Set up environment variables**
```bash
cp .env.example .env
# Edit .env with your configuration
```

### Docker Deployment

1. **Build the Docker image**
```bash
docker build -f Dockerfile.api -t flight-fare-api .
```

2. **Run the container**
```bash
docker run -d -p 8000:8000 --name flight-api flight-fare-api
```

---

## 📖 Usage

### 1. Train the Model

```bash
python run_pipeline.py
```

This will:
- Load and clean the data
- Engineer features
- Train multiple models
- Evaluate and save the best model
- Generate visualizations and reports

### 2. Run the API

**Local:**
```bash
uvicorn app:app --reload --host 0.0.0.0 --port 8000
```

**Docker:**
```bash
docker start flight-api
```

### 3. Run Streamlit App

```bash
streamlit run streamlit_app.py
```

### 4. Test the API

```bash
python test_api.py
```

---

## 🔌 API Documentation

### Base URL
```
http://localhost:8000
```

### Endpoints

#### 1. Health Check
```http
GET /health
```

**Response:**
```json
{
  "status": "healthy",
  "model_loaded": true,
  "timestamp": "2026-02-17T10:41:22Z"
}
```

#### 2. Single Prediction
```http
POST /predict
Content-Type: application/json
```

**Request Body:**
```json
{
  "Airline": "Biman Bangladesh Airlines",
  "Source": "DAC",
  "Destination": "CXB",
  "Date": "2024-06-15",
  "Departure_Time": "10:30",
  "Arrival_Time": "11:45",
  "Duration": "1h 15m",
  "Stopovers": "Direct",
  "Aircraft_Type": "Boeing 737",
  "Class": "Economy"
}
```

**Response:**
```json
{
  "predicted_fare": 12500.50,
  "currency": "BDT",
  "model_version": "1.0",
  "prediction_timestamp": "2026-02-17T10:41:22Z"
}
```

#### 3. Batch Predictions
```http
POST /predict/batch
Content-Type: application/json
```

**Request Body:**
```json
{
  "flights": [
    { /* flight 1 data */ },
    { /* flight 2 data */ }
  ]
}
```

#### 4. Model Info
```http
GET /model/info
```

**Response:**
```json
{
  "model_type": "GradientBoostingRegressor",
  "version": "1.0",
  "features": 50,
  "training_date": "2026-02-17",
  "performance": {
    "r2": 0.6428,
    "mae": 28707,
    "rmse": 48794
  }
}
```

---

## 📊 Model Performance

### Final Model: Gradient Boosting Regressor

| Metric | Value |
|--------|-------|
| **R² Score** | 0.6428 |
| **MAE** | 28,707 BDT |
| **RMSE** | 48,794 BDT |

### Model Comparison

| Model | R² | MAE (BDT) | RMSE (BDT) |
|-------|-----|-----------|------------|
| Linear Regression | 0.5650 | 40,213 | 53,850 |
| Ridge Regression | 0.5650 | 40,213 | 53,850 |
| Lasso Regression | 0.5649 | 40,214 | 53,851 |
| Decision Tree | 0.5200 | 35,000 | 56,500 |
| Random Forest | 0.6100 | 32,000 | 51,000 |
| **Gradient Boosting** ⭐ | **0.6428** | **28,707** | **48,794** |

### Key Insights

1. **Route is the most important factor** - Source-Destination combinations drive pricing
2. **Airline significantly affects fares** - Turkish Airlines & AirAsia charge ~10% premium
3. **Seasonal variation exists** - Winter fares are 16% higher than Autumn
4. **Class matters** - Business/First class significantly more expensive than Economy
5. **Duration impacts pricing** - Long-haul flights command higher fares

---

## 📁 Project Structure

```
flight-fare-prediction/
├── data/
│   ├── raw/                          # Original dataset
│   ├── processed/                    # Cleaned data
│   ├── train/                        # Training data
│   └── test/                         # Test data
├── src/
│   ├── __init__.py
│   ├── data_loader.py               # Data loading utilities
│   ├── preprocessing.py             # Data cleaning
│   ├── feature_engineering.py       # Feature pipeline
│   ├── transformers.py              # Custom transformers
│   ├── eda.py                       # Exploratory analysis
│   ├── train.py                     # Model training
│   └── evaluate.py                  # Model evaluation
├── notebooks/
│   ├── 01_exploration_and_results.ipynb
│   └── eda_walkthrough.ipynb
├── models/
│   └── best_model.pkl               # Production model
├── reports/
│   ├── figures/                     # 15 visualizations
│   ├── metrics.json                 # Model metrics
│   ├── MODEL_REPORT.md              # Detailed analysis
│   └── kpis.txt                     # Key insights
├── dags/
│   └── flight_price_dag.py          # Airflow DAG
├── scripts/
│   └── retrain_model.py             # Retraining script
├── app.py                           # FastAPI application
├── streamlit_app.py                 # Streamlit UI
├── test_api.py                      # API tests
├── run_pipeline.py                  # Main pipeline
├── Dockerfile.api                   # Docker configuration
├── requirements.txt                 # Dependencies
├── .env.example                     # Environment template
├── README.md                        # This file
├── API_DEPLOYMENT_SUMMARY.md        # API docs
├── STREAMLIT_GUIDE.md               # Streamlit docs
└── PROJECT_COMPLETION_CHECKLIST.md  # Verification
```

---

## 🎯 Stretch Challenges (All Completed!)

### ✅ Challenge 1: Web Application
**Status**: **COMPLETED**
- Built Streamlit app with interactive UI
- Real-time predictions via API
- User-friendly input forms
- Visual results display

**Files**: `streamlit_app.py`, `STREAMLIT_GUIDE.md`

### ✅ Challenge 2: Airflow Integration
**Status**: **COMPLETED**
- Automated ETL pipeline with scheduled execution
- Smart retraining logic based on data volume threshold
- PostgreSQL integration for data persistence
- Duplicate prevention with hash-based deduplication
- Failure alerts and monitoring

**Files**: `dags/flight_price_dag.py`, `scripts/retrain_model.py`

### ✅ Challenge 3: REST API Deployment
**Status**: **COMPLETED**
- FastAPI REST API with 4 endpoints
- Docker containerization for easy deployment
- Comprehensive test suite
- Health monitoring and model info endpoints
- Production-ready with error handling

**Files**: `app.py`, `Dockerfile.api`, `test_api.py`, `API_DEPLOYMENT_SUMMARY.md`

---

## 📚 Documentation

### Available Documentation Files

1. **README.md** (this file) - Complete project overview
2. **MODEL_REPORT.md** - Detailed model analysis and insights
3. **API_DEPLOYMENT_SUMMARY.md** - API deployment and usage guide
4. **STREAMLIT_GUIDE.md** - Streamlit application documentation
5. **PROJECT_COMPLETION_CHECKLIST.md** - Full project verification checklist

### Visualizations (15 files in `reports/figures/`)

- `distributions.png` - Fare distribution analysis
- `boxplot_fare_by_airline.png` - Airline pricing comparison
- `avg_fare_by_month.png` - Monthly fare trends
- `avg_fare_by_season.png` - Seasonal pricing patterns
- `correlation_heatmap.png` - Feature correlations
- `feature_importance.png` - Model feature importance
- `prediction_scatter.png` - Actual vs Predicted
- `residuals_dist.png` - Residual analysis
- `linear_coefficients.png` - Linear model coefficients
- Plus 6 statistical CSV files

---

## 🤝 Contributing

This is a complete, production-ready project. For improvements or extensions:
1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Submit a pull request

---

## 📄 License

This project is part of a machine learning portfolio demonstrating end-to-end ML system development.

---

## 👨‍💻 Author

**Mubarak Tijani**
- GitHub: [@Xenongt1](https://github.com/Xenongt1)

---

## 🙏 Acknowledgments

- Dataset: Flight Price Dataset of Bangladesh
- Tools: Scikit-learn, FastAPI, Streamlit, Apache Airflow, Docker, PostgreSQL
- Framework: Python 3.8+

---

**Project Status**: ✅ **100% COMPLETE** - Production Ready

*Last Updated: February 17, 2026*
