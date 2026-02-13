# Flight Fare Prediction

This project aims to predict flight fares in Bangladesh using machine learning.

## 1. Problem Definition

### Business Goal
Airlines and travel platforms want to estimate ticket prices based on route, airline, and travel date to help with pricing strategy and dynamic recommendations.

### ML Task
- **Type**: Supervised Regression
- **Target Variable**: Total Fare
- **Key Features**: Airline, Source, Destination, Date, Base Fare, Tax & Surcharge, etc.

## Structure
- `data/`: Contains raw and processed datasets.
- `notebooks/`: Jupyter notebooks for analysis.
- `src/`: Source code for the project.
- `models/`: Trained models.

## Stretch Challenges
1. **Integrate the model into a Flask or Streamlit app for live predictions.**
2. **Connect this project to your previous Airflow pipeline for scheduled retraining.**
3. **Deploy the model locally as a simple REST API.**
