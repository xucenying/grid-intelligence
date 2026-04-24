"""
Preprocessor for generating model-ready features from input data.
"""
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from .data import add_time_features


def generate_prediction_features(date_str: str, hours_ahead: int = 24) -> pd.DataFrame:
    """
    Generate model-ready features for prediction from a given date.
    Creates features for the next N hours with 15-minute intervals.
    
    Parameters:
    -----------
    date_str : str
        Date string in format "YYYY-MM-DD" (e.g., "2026-04-24")
    hours_ahead : int
        Number of hours to predict ahead (default: 24)
        
    Returns:
    --------
    pd.DataFrame
        DataFrame with 15 required model features for each 15-min interval
        
    Example:
    --------
    >>> features = generate_prediction_features("2026-04-24", hours_ahead=24)
    >>> features.shape
    (96, 15)  # 96 intervals (24h * 4) × 15 features
    """
    # Parse input date
    try:
        start_date = pd.to_datetime(date_str)
    except Exception as e:
        raise ValueError(f"Invalid date format: {date_str}. Use YYYY-MM-DD format.") from e
    
    # Generate 15-minute intervals for the specified hours
    num_intervals = hours_ahead * 4  # 4 intervals per hour
    timestamps = pd.date_range(
        start=start_date,
        periods=num_intervals,
        freq='15min'
    )
    
    # Create base dataframe
    df = pd.DataFrame({'datetime_utc': timestamps})
    
    # Add time features using existing function
    df = add_time_features(df, datetime_col='datetime_utc')
    
    # Calculate future holiday/bridge day features (72h ahead = 288 intervals)
    # Create extended dataframe with 288 more intervals to calculate future features
    extended_end = start_date + timedelta(hours=hours_ahead + 72)  # +72h for future features
    extended_timestamps = pd.date_range(
        start=start_date,
        end=extended_end,
        freq='15min'
    )
    
    extended_df = pd.DataFrame({'datetime_utc': extended_timestamps})
    extended_df = add_time_features(extended_df, datetime_col='datetime_utc')
    
    # Shift future features back by 288 steps
    df['is_holiday_288'] = extended_df['is_holiday'].shift(-288).iloc[:num_intervals].values
    df['is_bridge_day_288'] = extended_df['is_bridge_day'].shift(-288).iloc[:num_intervals].values
    
    # Fill any remaining NaN values at the end with 0 (non-holiday/non-bridge)
    df['is_holiday_288'] = df['is_holiday_288'].fillna(0).astype(int)
    df['is_bridge_day_288'] = df['is_bridge_day_288'].fillna(0).astype(int)
    
    # Select only the 15 features required by the model (in correct order)
    model_features = [
        'year',
        'is_holiday',
        'is_bridge_day',
        'quarter_hour_sin',
        'quarter_hour_cos',
        'hour_sin',
        'hour_cos',
        'day_of_week_sin',
        'day_of_week_cos',
        'day_of_year_sin',
        'day_of_year_cos',
        'month_sin',
        'month_cos',
        'is_holiday_288',
        'is_bridge_day_288'
    ]
    
    # Verify all required features exist
    missing_features = set(model_features) - set(df.columns)
    if missing_features:
        raise ValueError(f"Missing features after preprocessing: {missing_features}")
    
    # Return only model features in correct order
    return df[model_features]


def prepare_single_timestamp(timestamp: pd.Timestamp) -> pd.DataFrame:
    """
    Prepare features for a single timestamp.
    Useful for debugging or single-point predictions.
    
    Parameters:
    -----------
    timestamp : pd.Timestamp
        Single timestamp to generate features for
        
    Returns:
    --------
    pd.DataFrame
        Single-row DataFrame with 15 model features
    """
    # Create base dataframe with single timestamp
    df = pd.DataFrame({'datetime_utc': [timestamp]})
    df = add_time_features(df, datetime_col='datetime_utc')
    
    # Calculate future features (72h ahead)
    future_timestamp = timestamp + timedelta(hours=72)
    future_df = pd.DataFrame({'datetime_utc': [future_timestamp]})
    future_df = add_time_features(future_df, datetime_col='datetime_utc')
    
    df['is_holiday_288'] = future_df['is_holiday'].iloc[0]
    df['is_bridge_day_288'] = future_df['is_bridge_day'].iloc[0]
    
    model_features = [
        'year', 'is_holiday', 'is_bridge_day',
        'quarter_hour_sin', 'quarter_hour_cos',
        'hour_sin', 'hour_cos',
        'day_of_week_sin', 'day_of_week_cos',
        'day_of_year_sin', 'day_of_year_cos',
        'month_sin', 'month_cos',
        'is_holiday_288', 'is_bridge_day_288'
    ]
    
    return df[model_features]


def validate_features(df: pd.DataFrame) -> bool:
    """
    Validate that dataframe has all required features.
    
    Parameters:
    -----------
    df : pd.DataFrame
        DataFrame to validate
        
    Returns:
    --------
    bool
        True if all features present, raises ValueError otherwise
    """
    required_features = [
        'year', 'is_holiday', 'is_bridge_day',
        'quarter_hour_sin', 'quarter_hour_cos',
        'hour_sin', 'hour_cos',
        'day_of_week_sin', 'day_of_week_cos',
        'day_of_year_sin', 'day_of_year_cos',
        'month_sin', 'month_cos',
        'is_holiday_288', 'is_bridge_day_288'
    ]
    
    missing = set(required_features) - set(df.columns)
    if missing:
        raise ValueError(f"Missing required features: {missing}")
    
    if df.shape[1] != 15:
        raise ValueError(f"Expected 15 features, got {df.shape[1]}")
    
    return True
