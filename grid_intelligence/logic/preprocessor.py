"""
Preprocessor for generating model-ready features from input data.
"""
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from pathlib import Path
from .data import add_absolute_ramp, add_time, add_lag, add_rolling_mean, add_rolling_std, add_rolling_max


def add_interaction_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Add interaction features to capture non-linear relationships.

    Creates 9 interaction features:
    1. renewable_ratio - Energy mix ratio (key price driver)
    2. renewable_hour - Time-dependent renewable generation (solar peaks midday)
    3. renewable_season - Seasonal renewable generation patterns
    4. holiday_consumption - Holiday consumption patterns
    5. bridge_consumption - Bridge day consumption patterns
    6. oil_nonrenewable - Fossil fuel cost sensitivity for oil
    7. gas_nonrenewable - Fossil fuel cost sensitivity for gas
    8. peak_demand - Peak demand indicator (high consumption at peak hours)
    9. renewable_consumption - Energy mix × consumption interaction
    10. oil_gas_ratio - Oil/gas ratio (fuel switching behavior)

    Parameters:
    -----------
    df : pd.DataFrame
        Dataframe with base features including generation, consumption, prices, etc.

    Returns:
    --------
    pd.DataFrame
        Dataframe with added interaction features
    """
    # 1. Energy mix ratio (key price driver)
    df['renewable_ratio'] = df['generation_renewable'] / (
        df['generation_renewable'] + df['generation_non_renewable'] + 1e-6
    )

    # 2. Time-dependent renewable generation (solar peaks midday, wind varies)
    df['renewable_hour'] = df['generation_renewable'] * df['hour_sin']
    df['renewable_season'] = df['generation_renewable'] * df['month_sin']

    # 3. Holiday consumption patterns (different demand on holidays)
    df['holiday_consumption'] = df['is_holiday'] * df['consumption']
    df['bridge_consumption'] = df['is_bridge_day'] * df['consumption']

    # 4. Fossil fuel cost sensitivity (oil/gas impact when non-renewable is high)
    df['oil_nonrenewable'] = df['wti_oil'] * df['generation_non_renewable']
    df['gas_nonrenewable'] = df['natural_gas'] * df['generation_non_renewable']

    # 5. Peak demand indicator (high consumption at peak business hours)
    df['peak_demand'] = df['consumption'] * np.abs(df['hour_sin'])

    # 6. Energy mix × consumption (price pressure from energy source)
    df['renewable_consumption'] = df['renewable_ratio'] * df['consumption']

    # 7. Oil/gas ratio (fuel switching behavior)
    df['oil_gas_ratio'] = df['wti_oil'] / (df['natural_gas'] + 1e-6)

    return df


def generate_features(nrows: int = 1632, train: bool = True) -> pd.DataFrame:
    """
    Generate model-ready features from consolidated data.

    Loads the last 1632 rows from consolidated_full.csv and engineers features including:
    - 1632 rows (288 for prediction + 1344 for lag calculations)
    - also used for generating df for training model by changing nrows
    - Time-based features (hour, day, holidays, cyclical encodings, future holiday/bridge day flags)
    - Lag features for multiple columns (1, 4, 12, 24, 96, 672 timesteps)
    - Rolling mean and std features (windows: 4, 16, 96)
    - Interaction features (renewable_ratio, renewable_hour, renewable_season, etc.)
    - for testing, drop price column

    Feature columns processed:
    - price
    - generation_renewable
    - generation_non_renewable
    - consumption
    - wti_oil
    - brent_oil
    - natural_gas

    Parameters:
    -----------
    nrows : int
        Number of rows to load from the end of the consolidated_full.csv file.

    Returns:
    --------
    pd.DataFrame
        Dataframe with time features, lag features, and rolling statistics.
        Contains 1632 rows (288 for prediction + 1344 for lag calculations).

    Notes:
    ------
    - Expects consolidated_full.csv in raw_data/ directory at project root
    - 288 rows = 72 hours of 15-minute intervals (for 72h ahead prediction)
    - 1344 rows = additional data needed for max lag window (672 timesteps)
    """
    # Get absolute path to data file
    # Navigate from this file location to project root and then to raw_data
    current_file = Path(__file__)  # grid_intelligence/logic/preprocessor.py
    project_root = current_file.parent.parent.parent  # Up to project root
    data_path = project_root / "raw_data" / "consolidated_full.csv"

    if not data_path.exists():
        raise FileNotFoundError(
            f"Data file not found at {data_path}. "
            f"Please ensure consolidated_full.csv exists in the raw_data/ directory."
        )

    df = pd.read_csv(str(data_path))
    # 288 rows for prediction and 1344 rows for lag features (total 1632)
    df = df.tail(nrows).reset_index(drop=True)

    # Add time-based features
    df = add_time(df, datetime_col="datetime_utc")

    # add price lag, rolling mean, rolling std, absolute ramp features
    df = add_lag(df, target_col="price")
    df = add_rolling_mean(df, target_col="price")
    df = add_rolling_std(df, target_col="price")
    # df = add_absolute_ramp(df, target_col="price")

    # add generation_renewable lag, rolling mean, and rolling std features
    df = add_lag(df, target_col="generation_renewable")
    df = add_rolling_mean(df, target_col="generation_renewable")
    df = add_rolling_std(df, target_col="generation_renewable")

    # add generation_non_renewable lag, rolling mean, and rolling std features
    df = add_lag(df, target_col="generation_non_renewable")
    df = add_rolling_mean(df, target_col="generation_non_renewable")
    df = add_rolling_std(df, target_col="generation_non_renewable")

    # add consumption lag, rolling mean, and rolling std features
    df = add_lag(df, target_col="consumption")
    df = add_rolling_mean(df, target_col="consumption")
    df = add_rolling_std(df, target_col="consumption")

    #add wti_oil lag, rolling mean, and rolling std features
    df = add_lag(df, target_col="wti_oil")
    df = add_rolling_mean(df, target_col="wti_oil")
    df = add_rolling_std(df, target_col="wti_oil")

    #add brent_oil lag, rolling mean, and rolling std features
    df = add_lag(df, target_col="brent_oil")
    df = add_rolling_mean(df, target_col="brent_oil")
    df = add_rolling_std(df, target_col="brent_oil")

    # add natural_gas lag, rolling mean, and rolling std features
    df = add_lag(df, target_col="natural_gas")
    df = add_rolling_mean(df, target_col="natural_gas")
    df = add_rolling_std(df, target_col="natural_gas")

    # add wind_onshore lag, rolling mean, and rolling std features
    df = add_lag(df, target_col="wind_onshore")
    df = add_rolling_mean(df, target_col="wind_onshore")
    df = add_rolling_std(df, target_col="wind_onshore")

    # add temperature_c_observed lag, rolling mean, and rolling std features
    df = add_lag(df, target_col="temperature_c_observed")
    df = add_rolling_mean(df, target_col="temperature_c_observed")
    df = add_rolling_std(df, target_col="temperature_c_observed")

    # add cloud_cover_percent_observed lag, rolling mean, and rolling std features
    df = add_lag(df, target_col="cloud_cover_percent_observed")
    df = add_rolling_mean(df, target_col="cloud_cover_percent_observed")
    df = add_rolling_std(df, target_col="cloud_cover_percent_observed")

    # add shortwave_radiation_wm2_observed lag, rolling mean, and rolling std features
    df = add_lag(df, target_col="shortwave_radiation_wm2_observed")
    df = add_rolling_mean(df, target_col="shortwave_radiation_wm2_observed")
    df = add_rolling_std(df, target_col="shortwave_radiation_wm2_observed")

    # add wind_speed_ms_observed lag, rolling mean, and rolling std features
    df = add_lag(df, target_col="wind_speed_ms_observed")
    df = add_rolling_mean(df, target_col="wind_speed_ms_observed")
    df = add_rolling_std(df, target_col="wind_speed_ms_observed")

    # add ttf_gas lag, rolling mean, and rolling std features
    df = add_lag(df, target_col="ttf_gas")
    df = add_rolling_mean(df, target_col="ttf_gas")
    df = add_rolling_std(df, target_col="ttf_gas")

    # Add interaction features to capture non-linear relationships
    df = add_interaction_features(df)

    # Drop any feature not used for prediction BEFORE dropna
    # (especially forecast columns which have mostly NaN values)
    columns_to_drop = [
        'datetime_utc',
        'temperature_c',
        'humidity_percent',
        'cloud_cover_percent',
        'shortwave_radiation_wm2',
        'wind_speed_ms',
        'temperature_c_forecast',
        'humidity_percent_forecast',
        'cloud_cover_percent_forecast',
        'shortwave_radiation_wm2_forecast',
        'wind_speed_ms_forecast'
    ]

    # Drop low-value features (identified via feature importance analysis)
    # These 31 features contribute only 7.66% of total importance
    features_to_drop_lowval = [
        'generation_renewable_roll_std_4', 'wind_onshore_roll_std_4', 'brent_oil_lag_96',
        'brent_oil_roll_mean_96', 'generation_renewable_lag_12', 'brent_oil_lag_4',
        'humidity_percent_observed', 'is_holiday', 'wind_speed_ms_observed_roll_std_4',
        'brent_oil_lag_24', 'brent_oil_lag_672', 'year', 'brent_oil_lag_12',
        'is_bridge_day_288', 'consumption_roll_std_4', 'brent_oil_roll_std_24',
        'wti_oil_roll_std_4', 'wind_onshore_lag_4', 'ttf_gas_roll_std_4',
        'cloud_cover_percent_observed_roll_std_4', 'brent_oil_roll_std_96',
        'brent_oil_roll_mean_24', 'natural_gas_roll_std_4', 'brent_oil_roll_mean_672',
        'price_roll_std_24', 'brent_oil_lag_1', 'brent_oil_roll_std_4',
        'is_holiday_288', 'is_bridge_day', 'brent_oil', 'wind_speed_ms_observed_roll_std_24'
    ]

    columns_to_drop.extend(features_to_drop_lowval)

    if not train:
        columns_to_drop.append('price')

    df = df.drop(columns=columns_to_drop)

    # Drop rows with missing values (created by lag/rolling features)
    df = df.dropna().reset_index(drop=True)

    return df
