
import pandas as pd
import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin
from src.utils import get_logger

class DataNormalizer(BaseEstimator, TransformerMixin):
    """
    Standardizes names for Airlines and Cities to ensure consistency 
    between Database and UI inputs.
    """
    def __init__(self):
        self.airline_map = {
            'Us-Bangla Airlines': 'US-Bangla Airlines',
            'Novoair': 'NovoAir',
            'Airasia': 'AirAsia',
            'Indigo': 'IndiGo',
            'Srilankan Airlines': 'SriLankan Airlines'
        }
        
    def fit(self, X, y=None):
        return self

    def _clean_city(self, city):
        if not isinstance(city, str): return city
        # Extract city from "Airport Name, City" or "City Airport"
        if ',' in city:
            return city.split(',')[-1].strip()
        # Handle "Cox'S Bazar Airport" -> "Cox's Bazar"
        if "Cox'S Bazar" in city: return "Cox's Bazar"
        # Remove "Airport" suffix
        if ' Airport' in city:
            return city.replace(' Airport', '').strip()
        if ' International' in city:
             return city.replace(' International', '').strip()
        return city.strip()

    def transform(self, X):
        X = X.copy()
        
        # 1. Normalize Airlines
        if 'Airline' in X.columns:
            X['Airline'] = X['Airline'].replace(self.airline_map)
            
        # 2. Normalize Cities
        for col in ['Source', 'Destination']:
            if col in X.columns:
                X[col] = X[col].apply(self._clean_city)
                
        # 3. Handle Aircraft Types (ensure common format)
        if 'Aircraft Type' in X.columns:
            X['Aircraft Type'] = X['Aircraft Type'].str.replace('AirbusA', 'Airbus A', regex=False)
            
        return X

logger = get_logger(__name__)

class DateFeatureEngineer(BaseEstimator, TransformerMixin):
    def __init__(self, date_col='Date'):
        self.date_col = date_col

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        X = X.copy()
        if self.date_col in X.columns:
            # Ensure it's datetime
            if not pd.api.types.is_datetime64_any_dtype(X[self.date_col]):
                X[self.date_col] = pd.to_datetime(X[self.date_col])
                
            X["Month"] = X[self.date_col].dt.month
            X["Day"] = X[self.date_col].dt.day
            X["Weekday"] = X[self.date_col].dt.weekday
            X["Dep_Hour"] = X[self.date_col].dt.hour
        return X

class TimeOfDayEngineer(BaseEstimator, TransformerMixin):
    def __init__(self, hour_col='Dep_Hour'):
        self.hour_col = hour_col

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        X = X.copy()
        if self.hour_col in X.columns:
            def get_time_of_day(hour):
                if 5 <= hour < 12: return 'Morning'
                elif 12 <= hour < 17: return 'Afternoon'
                elif 17 <= hour < 22: return 'Evening'
                else: return 'Night'
            
            X['Time_of_Day'] = X[self.hour_col].apply(get_time_of_day)
        return X

class CyclicalFeatureEngineer(BaseEstimator, TransformerMixin):
    def fit(self, X, y=None):
        return self

    def transform(self, X):
        X = X.copy()
        # Ensure underlying types are float for calculation
        for col, max_val in [('Month', 12), ('Day', 31), ('Weekday', 7), ('Dep_Hour', 24)]:
            if col in X.columns:
                X[f'{col}_Sin'] = np.sin(2 * np.pi * X[col] / max_val)
                X[f'{col}_Cos'] = np.cos(2 * np.pi * X[col] / max_val)
        return X

class RouteEngineer(BaseEstimator, TransformerMixin):
    def fit(self, X, y=None):
        return self

    def transform(self, X):
        X = X.copy()
        if "Source" in X.columns and "Destination" in X.columns:
            X["Route"] = X["Source"] + "_" + X["Destination"]
        return X

class DurationCategoryEngineer(BaseEstimator, TransformerMixin):
    def __init__(self, duration_col='Duration (hrs)'):
        self.duration_col = duration_col

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        X = X.copy()
        if self.duration_col in X.columns:
            # Handle non-numeric gracefully if needed, though preproc should fix
            X['Duration_Category'] = pd.cut(X[self.duration_col], 
                                           bins=[-1, 5, 10, 100], 
                                           labels=['Short-Haul', 'Medium-Haul', 'Long-Haul'])
            # Convert to string to avoid categorical type issues in some sklearn versions/pipelines
            X['Duration_Category'] = X['Duration_Category'].astype(str)
        return X

class SeasonEngineer(BaseEstimator, TransformerMixin):
    def fit(self, X, y=None):
        return self

    def transform(self, X):
        X = X.copy()
        if "Month" in X.columns:
            def get_season(month):
                if month in [12, 1, 2]: return 'Winter'
                elif month in [3, 4, 5]: return 'Spring'
                elif month in [6, 7, 8]: return 'Summer'
                else: return 'Autumn'
            X['Season'] = X['Month'].apply(get_season)
        return X

class CustomOrdinalEncoder(BaseEstimator, TransformerMixin):
    def fit(self, X, y=None):
        return self

    def transform(self, X):
        X = X.copy()
        if "Stopovers" in X.columns:
            stopover_map = {"Direct": 0, "1 Stop": 1, "2 Stops": 2, "3 Stops": 3}
            # Handle unknown (e.g. from API) with 0
            X["Stopovers"] = X["Stopovers"].map(stopover_map).fillna(0).astype(int)
            
        if "Class" in X.columns:
             class_map = {"Economy": 1, "Business": 2, "First Class": 3}
             X["Class_Encoded"] = X["Class"].map(class_map).fillna(1).astype(int)
        return X

class Columndropper(BaseEstimator, TransformerMixin):
    def __init__(self, cols):
        self.cols = cols

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        # Drop columns if they exist
        return X.drop(columns=[c for c in self.cols if c in X.columns], errors='ignore')
