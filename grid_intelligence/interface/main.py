import pandas as pd
import numpy as np
from grid_intelligence.logic.registry import load_models
from grid_intelligence.logic.preprocessor import generate_features


# Load all models once at module import (singleton pattern)
_models = None

def _get_models():
    global _models
    if _models is None:
        _models = load_models()
    return _models


def predict_multi_regime(features: pd.DataFrame) -> np.ndarray:
    """
    Make predictions using multi-regime XGBoost ensemble.

    Parameters:
    -----------
    features : pd.DataFrame
        Feature dataframe (without price, target_288, regime columns)

    Returns:
    --------
    predictions : np.ndarray
        Predicted electricity prices
    """
    models = _get_models()

    # Extract models and config
    clf = models['regime_classifier']
    model_normal = models['model_normal']
    model_pos = models['model_pos']
    model_neg = models['model_neg']
    config = models['model_config']

    # Get regime probabilities
    probs = clf.predict_proba(features)
    p_normal = probs[:, 0]
    p_pos = probs[:, 1]
    p_neg = probs[:, 2]

    # Get base predictions from each regime model
    pred_normal = model_normal.predict(features)
    pred_pos = model_pos.predict(features)
    pred_neg = model_neg.predict(features)

    # Soft blending: probability-weighted predictions with scaling factors
    scaling = config['scaling_factors']
    predictions = (
        p_neg * pred_neg * scaling['neg'] +
        p_normal * pred_normal * scaling['normal'] +
        p_pos * pred_pos * scaling['pos']
    )

    return predictions


def update_features_for_next_step(features: pd.DataFrame, predicted_price: float,
                                  step: int, start_time: pd.Timestamp,
                                  price_history: list) -> pd.DataFrame:
    """
    Update features for the next prediction step.

    Parameters:
    -----------
    features : pd.DataFrame
        Current features (single row)
    predicted_price : float
        Price predicted for current step
    step : int
        Current step number (0-indexed)
    start_time : pd.Timestamp
        Start time of predictions
    price_history : list
        List of predicted prices so far

    Returns:
    --------
    pd.DataFrame
        Updated features for next step
    """
    features_next = features.copy()

    # 1. Update lag features for price
    lag_windows = [1, 4, 12, 24, 96, 672]
    for lag in lag_windows:
        col_name = f'price_lag_{lag}'
        if col_name in features_next.columns:
            # Get the price from lag steps ago in our prediction history
            if len(price_history) >= lag:
                features_next[col_name] = price_history[-lag]
            # else: keep existing value (from historical data)

    # 2. Update rolling statistics for price
    rolling_windows = [4, 16, 96]
    for window in rolling_windows:
        # Rolling mean
        mean_col = f'price_roll_mean_{window}'
        if mean_col in features_next.columns and len(price_history) >= window:
            features_next[mean_col] = np.mean(price_history[-window:])

        # Rolling std
        std_col = f'price_roll_std_{window}'
        if std_col in features_next.columns and len(price_history) >= window:
            features_next[std_col] = np.std(price_history[-window:])

        # Rolling max
        max_col = f'price_roll_max_{window}'
        if max_col in features_next.columns and len(price_history) >= window:
            features_next[max_col] = np.max(price_history[-window:])

    # 3. Update time-based features
    current_time = start_time + pd.Timedelta(minutes=15 * (step + 1))

    # Hour features
    if 'hour' in features_next.columns:
        features_next['hour'] = current_time.hour
    if 'hour_sin' in features_next.columns:
        features_next['hour_sin'] = np.sin(2 * np.pi * current_time.hour / 24)
    if 'hour_cos' in features_next.columns:
        features_next['hour_cos'] = np.cos(2 * np.pi * current_time.hour / 24)

    # Day features
    if 'day_of_week' in features_next.columns:
        features_next['day_of_week'] = current_time.dayofweek
    if 'day_of_week_sin' in features_next.columns:
        features_next['day_of_week_sin'] = np.sin(2 * np.pi * current_time.dayofweek / 7)
    if 'day_of_week_cos' in features_next.columns:
        features_next['day_of_week_cos'] = np.cos(2 * np.pi * current_time.dayofweek / 7)

    # Month features
    if 'month' in features_next.columns:
        features_next['month'] = current_time.month
    if 'month_sin' in features_next.columns:
        features_next['month_sin'] = np.sin(2 * np.pi * current_time.month / 12)
    if 'month_cos' in features_next.columns:
        features_next['month_cos'] = np.cos(2 * np.pi * current_time.month / 12)

    # Day of month
    if 'day' in features_next.columns:
        features_next['day'] = current_time.day

    # Weekend flag
    if 'is_weekend' in features_next.columns:
        features_next['is_weekend'] = 1 if current_time.dayofweek >= 5 else 0

    # 4. Update interaction features that depend on price or time
    # renewable_hour uses hour_sin
    if 'renewable_hour' in features_next.columns and 'generation_renewable' in features_next.columns:
        features_next['renewable_hour'] = features_next['generation_renewable'] * features_next['hour_sin']

    # renewable_season uses month_sin
    if 'renewable_season' in features_next.columns and 'generation_renewable' in features_next.columns:
        features_next['renewable_season'] = features_next['generation_renewable'] * features_next['month_sin']

    # peak_demand uses hour_sin
    if 'peak_demand' in features_next.columns and 'consumption' in features_next.columns:
        features_next['peak_demand'] = features_next['consumption'] * np.abs(features_next['hour_sin'])

    # Note: Other features (generation, consumption, weather, oil/gas prices)
    # are kept constant (persistence assumption) as we don't have forecasts for them

    return features_next


def predict() -> dict:
    """
    Predict energy prices for the next 288 timesteps (72 hours) using multi-regime XGBoost.

    Uses iterative multi-step forecasting:
    - Predicts next price
    - Updates features based on prediction
    - Repeats for all 288 steps

    Input:  None
    Output: dict with 288 predictions (72 hours at 15-min intervals)

    Returns:
    --------
    dict
        Predictions with timestamps and metadata
    """
    try:
        # Generate features for prediction using default nrows (1632)
        df_features = generate_features(train=False)

        # Store actual historical prices for initial lag/rolling calculations
        if 'price' in df_features.columns:
            historical_prices = df_features['price'].tail(672).tolist()  # Max lag is 672
        else:
            # If no price column, initialize with zeros (features already have lags embedded)
            historical_prices = []

        # Drop columns that shouldn't be used as features
        columns_to_drop = ['price', 'target_288', 'regime']
        features = df_features.drop(columns=[c for c in columns_to_drop if c in df_features.columns])

        # Fixed prediction: 288 timesteps (72 hours at 15-min intervals)
        num_intervals = 288

        # Start with the most recent features
        current_features = features.tail(1).copy()

        # Generate timestamps
        start_time = pd.Timestamp.now().round('15min')
        timestamps = pd.date_range(start=start_time, periods=num_intervals, freq='15min')

        # Iterative multi-step forecasting
        predictions = []
        price_history = historical_prices.copy()  # Combine historical + predicted prices

        for step in range(num_intervals):
            # Make prediction for current step
            pred = predict_multi_regime(current_features)[0]
            predictions.append(pred)
            price_history.append(pred)

            # Update features for next step (if not the last step)
            if step < num_intervals - 1:
                current_features = update_features_for_next_step(
                    current_features,
                    pred,
                    step,
                    start_time,
                    price_history
                )

        # Round predictions
        predictions_list = [round(float(pred), 2) for pred in predictions]
        #predictions_series = pd.Series(predictions).rolling(window=4, center=True, min_periods=1).mean()
        #predictions_list = [round(float(p), 2) for p in predictions_series]


        timestamp_strings = [ts.strftime('%Y-%m-%d %H:%M:%S') for ts in timestamps]

        return {
            "start_time": start_time.strftime('%Y-%m-%d %H:%M:%S'),
            "predictions_15min": predictions_list,
            "timestamps": timestamp_strings,
            "intervals": num_intervals,
            "hours_covered": 72,
            "unit": "EUR/MWh",
            "model_type": "Multi-Regime XGBoost",
            "forecast_method": "Iterative multi-step (rolling forecast)"
        }

    except Exception as e:
        return {
            "error": str(e),
            "message": "Prediction failed. Please check that models are trained and saved.",
            "help": "Run the training notebook (temp.ipynb) and save models first."
        }
