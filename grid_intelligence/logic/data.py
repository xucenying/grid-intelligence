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

def cyclical_encode(df: pd.DataFrame, col: str, max_val: int, offset: int = 0, drop: bool = True) -> pd.DataFrame:
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


def add_time(df: pd.DataFrame, datetime_col: str = "datetime_utc") -> pd.DataFrame:
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
    df[datetime_col] = pd.to_datetime(df[datetime_col], utc=True)


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
    df = cyclical_encode(df, "quarter_hour", 4)
    df = cyclical_encode(df, "hour", 24)
    df = cyclical_encode(df, "day_of_week", 7)
    df = cyclical_encode(df, "day_of_year", 365, offset=1)
    df = cyclical_encode(df, "month", 12, offset=1)

    # Remove temporary date column
    df = df.drop(columns=["date"])

    # Create future feature columns (known features 288 steps ahead)
    # 288 timesteps = 288 * 15min = 72 hours ahead
    #future_timestamp = df[datetime_col] + pd.Timedelta(minutes=288 * 15)
    df['future_timestamp'] = df[datetime_col] + pd.Timedelta(minutes=288 * 15)

    # Check if future timestamp is a holiday
    df['is_holiday_288'] = df['future_timestamp'].dt.floor("D").isin(de_holidays).astype(int)

    # Check if future timestamp is a bridge day
    future_date = df['future_timestamp'].dt.date
    df['is_bridge_day_288'] = future_date.apply(lambda x: is_bridge_day(x, years))

    return df

def add_lag(df: pd.DataFrame, target_col: str = "price", windows: list = [1, 4, 12, 24, 96, 672]) -> pd.DataFrame:
    """Add rolling lag features for the target variable."""
    df = df.copy()
    for window in windows:
        df[f"{target_col}_lag_{window}"] = df[target_col].shift(window)
    return df

def add_rolling_mean(df: pd.DataFrame, target_col: str = "price", windows: list = [24, 96, 672]) -> pd.DataFrame:
    """Add rolling mean and std features for the target variable."""
    df = df.copy()
    for window in windows:
        df[f"{target_col}_roll_mean_{window}"] = df[target_col].shift(1).rolling(window=window, min_periods=1).mean()
    return df

def add_rolling_std(df: pd.DataFrame, target_col: str = "price", windows: list = [4, 24, 96]) -> pd.DataFrame:
    """Add rolling std features for the target variable."""
    df = df.copy()
    for window in windows:
        df[f"{target_col}_roll_std_{window}"] = df[target_col].shift(1).rolling(window=window, min_periods=1).std()
    return df

def add_rolling_max(df: pd.DataFrame, target_col: str = "price", windows: list = [4, 12]) -> pd.DataFrame:
    """Add rolling max features for the target variable."""
    df = df.copy()
    for window in windows:
        df[f"{target_col}_roll_max_{window}"] = df[target_col].shift(1).rolling(window=window, min_periods=1).max()
    return df

def add_absolute_ramp(df: pd.DataFrame, target_col: str = "price", windows: list = [4, 96, 672]) -> pd.DataFrame:
    """Add absolute ramp features (difference between current value and lagged value) for the target variable."""
    df = df.copy()
    for window in windows:
        df[f"{target_col}_ramp_{window}"] = (df[target_col] - df[target_col].shift(window)).abs()
    return df
