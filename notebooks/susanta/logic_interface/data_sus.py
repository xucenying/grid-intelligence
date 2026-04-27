import os
import pandas as pd


DEFAULT_PATH = "/home/home/code/xucenying/grid-intelligence/notebooks/susanta/data_energy_weather_lag.csv"


def load_data(file_path=None, create_lags=True):
    """
    Loads dataset and optionally creates lag features safely.
    """

    path = file_path or DEFAULT_PATH

    if not isinstance(path, (str, bytes, os.PathLike)):
        raise ValueError(f"Invalid file path: {path}")

    if not os.path.exists(path):
        raise FileNotFoundError(f"File not found: {path}")

    df = pd.read_csv(path)

    # -----------------------------
    # AUTO DETECT TARGET COLUMN
    # -----------------------------
    possible_targets = [
        "Price[Currency/MWh]",
        "price",
        "Price",
        "price_mwh"
    ]

    target_col = next((c for c in possible_targets if c in df.columns), None)

    if target_col is None:
        raise ValueError(f"No valid target column found. Available: {df.columns.tolist()}")

    # -----------------------------
    # CREATE LAG FEATURES
    # -----------------------------
    if create_lags:
        for lag in [1, 4, 8, 24, 96, 192, 672]:
            df[f"price_lag_{lag}"] = df[target_col].shift(lag)

    return df
