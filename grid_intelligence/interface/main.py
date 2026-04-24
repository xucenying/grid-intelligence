import pandas as pd
import numpy as np
from grid_intelligence.logic.registry import load_model
from grid_intelligence.logic.preprocessor import generate_prediction_features


# Load model once at module import (singleton pattern)
model = load_model()


def predict(input_date: str) -> dict:
    """
    Predict energy prices for the next 24 hours from a given date.
    
    Input:  date string e.g. "2026-04-24"
    Output: dict with predictions
    
    Returns predictions for 24 hours (96 intervals of 15 minutes each).
    """
    try:
        # Generate features for the next 24 hours
        features = generate_prediction_features(input_date, hours_ahead=24)
        
        # Make predictions using the loaded model
        predictions = model.predict(features)
        
        # Convert to list and round to 2 decimal places
        predictions_list = [round(float(pred), 2) for pred in predictions]
        
        # Generate timestamps for each prediction
        start_time = pd.to_datetime(input_date)
        timestamps = pd.date_range(start=start_time, periods=96, freq='15min')
        timestamp_strings = [ts.strftime('%Y-%m-%d %H:%M:%S') for ts in timestamps]
        
        return {
            "date": input_date,
            "predictions_15min": predictions_list,
            "timestamps": timestamp_strings,
            "intervals": 96,
            "hours_covered": 24,
            "unit": "EUR/MWh",
            "model_type": "XGBoost"
        }
        
    except Exception as e:
        return {
            "date": input_date,
            "error": str(e),
            "message": "Prediction failed. Please check the date format (YYYY-MM-DD) and try again."
        }
