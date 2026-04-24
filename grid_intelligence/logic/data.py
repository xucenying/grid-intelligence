import pandas as pd
import holidays
import numpy as np
from xgboost import XGBRegressor

def clean_data(df: pd.DataFrame) -> pd.DataFrame:
    """Clean raw data."""
    pass

def is_bridge_day(date, years) -> int:
    de_holidays = holidays.Germany(years=years)
    holidays_set = set(de_holidays)
    weekday = date.weekday()

    # Monday before Tuesday holiday
    if weekday == 0 and (date + pd.Timedelta(days=1)) in holidays_set:
        return 1

    # Friday after Thursday holiday
    if weekday == 4 and (date - pd.Timedelta(days=1)) in holidays_set:
        return 1

    return 0


def add_time_features(df: pd.DataFrame, datetime_col: str = "DateTime(UTC)") -> pd.DataFrame:
    """
    Add time-based features to dataframe including basic time features,
    holidays, bridge days, and cyclical encodings.

    Parameters:
    -----------
    df : pd.DataFrame
        Input dataframe with datetime column
    datetime_col : str
        Name of the datetime column (default: "DateTime(UTC)")

    Returns:
    --------
    pd.DataFrame
        Dataframe with added time features
    """
    df = df.copy()
    df[datetime_col] = pd.to_datetime(df[datetime_col])

    # ---- Basic time features ----
    df["day_of_week"] = df[datetime_col].dt.dayofweek
    df["day_of_year"] = df[datetime_col].dt.dayofyear
    df["month"] = df[datetime_col].dt.month
    df["year"] = df[datetime_col].dt.year

    # ---- Hour + quarter-hour ----
    df["hour"] = df[datetime_col].dt.hour

    # quarter of hour: 0,1,2,3
    df["quarter_hour"] = df[datetime_col].dt.minute // 15

    # ---- German public holidays ----
    years = df["year"].unique()
    de_holidays = holidays.Germany(years=years)

    df["is_holiday"] = df[datetime_col].dt.floor("D").isin(de_holidays)

    # ---- Bridge day (Brückentag) ----
    # A bridge day is typically:
    # - Monday before a Tuesday holiday
    # - Friday after a Thursday holiday

    df["date"] = df[datetime_col].dt.date

    df["is_bridge_day"] = df["date"].apply(lambda x: is_bridge_day(x, years))

    # convert holiday boolean to int
    df["is_holiday"] = df["is_holiday"].astype(int)

    # ---- Cyclical encode repeated time-patterns ----
    df = _cyclical_encode(df, "quarter_hour", 4)
    df = _cyclical_encode(df, "hour", 24)
    df = _cyclical_encode(df, "day_of_week", 7)
    df = _cyclical_encode(df, "day_of_year", 365, offset=1)
    df = _cyclical_encode(df, "month", 12, offset=1)

    # Remove temporary date column
    df = df.drop(columns=["date"])

    return df


def _cyclical_encode(df: pd.DataFrame, col: str, max_val: int, offset: int = 0, drop: bool = True) -> pd.DataFrame:
    """
    Cyclical encode an ordinal feature using sin and cos transformations.

    Parameters:
    -----------
    df : pd.DataFrame
        Input dataframe
    col : str
        Column name to encode
    max_val : int
        Maximum value of the ordinal feature
    offset : int
        Offset for encoding (default: 0)
    drop : bool
        Whether to drop original column (default: True)

    Returns:
    --------
    pd.DataFrame
        Dataframe with sin and cos encoded columns
    """
    sin_col = f"{col}_sin"
    cos_col = f"{col}_cos"

    df[sin_col] = np.sin(2 * np.pi * (df[col] - offset) / max_val)
    df[cos_col] = np.cos(2 * np.pi * (df[col] - offset) / max_val)

    if drop:
        df = df.drop(columns=[col])

    return df


def prepare_model_data(df: pd.DataFrame, price_col: str = "Price[Currency/MWh]") -> pd.DataFrame:
    """
    Prepare dataframe for model training by selecting features and creating target.

    This function:
    1. Keeps only the required model features
    2. Creates target_288 (price shifted 288 steps into future)
    3. Creates future feature columns (is_holiday_288, is_bridge_day_288)
    4. Drops rows with missing values

    Parameters:
    -----------
    df : pd.DataFrame
        Input dataframe with time features already added
    price_col : str
        Name of the price column (default: "Price[Currency/MWh]")

    Returns:
    --------
    pd.DataFrame
        Prepared dataframe ready for model training with features and target
    """
    # Define required feature columns
    feature_columns = [
        'year', 'is_holiday', 'is_bridge_day',
        'quarter_hour_sin', 'quarter_hour_cos',
        'hour_sin', 'hour_cos',
        'day_of_week_sin', 'day_of_week_cos',
        'day_of_year_sin', 'day_of_year_cos',
        'month_sin', 'month_cos'
    ]

    # Keep only feature columns and price column
    columns_to_keep = feature_columns + [price_col]
    model_df = df[columns_to_keep].copy()

    # Create target: price 288 steps (72 hours) into the future
    model_df["target_288"] = model_df[price_col].shift(-288)

    # Create future feature columns (known features 288 steps ahead)
    model_df['is_holiday_288'] = model_df['is_holiday'].shift(-288)
    model_df['is_bridge_day_288'] = model_df['is_bridge_day'].shift(-288)

    # Drop the original price column (no longer needed as feature)
    model_df = model_df.drop(columns=[price_col])

    # Drop rows with missing values and reset index
    model_df = model_df.dropna().reset_index(drop=True)

    return model_df
