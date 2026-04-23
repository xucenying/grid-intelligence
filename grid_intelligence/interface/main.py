import pandas as pd
import numpy as np

def predict(input_date: str) -> dict:
    """
    TODO: Load model and predict price for next 24h
    Input:  date string e.g. "2026-04-24"
    Output: dict with predictions
    """
    # TODO: Replace with real model prediction
    predictions = list(np.random.uniform(50, 150, 24).round(2))

    return {
        "date": input_date,
        "predictions_24h": predictions,
        "unit": "EUR/MWh"
    }
