from .data_sus import load_data
from sklearn.metrics import mean_absolute_error
from xgboost import XGBRegressor
import numpy as np

_model = None
_mae = None
_features = None
_target = None


def initialize_model(file_path=None, sep=","):
    global _model, _mae, _features, _target

    df = load_data(file_path=file_path).dropna()

    # -----------------------------
    # FEATURES
    # -----------------------------
    weather_features = [
        "temperature_c",
        "humidity_percent",
        "cloud_cover_percent",
        "shortwave_radiation_wm2"
    ]

    lag_features = [
        "price_lag_1", "price_lag_4", "price_lag_8",
        "price_lag_24", "price_lag_96",
        "price_lag_192", "price_lag_672"
    ]

    _features = weather_features + lag_features

    # -----------------------------
    # SAFE TARGET DETECTION
    # -----------------------------
    possible_targets = [
        "Price[Currency/MWh]",
        "price",
        "Price",
        "price_mwh"
    ]

    target = next((c for c in possible_targets if c in df.columns), None)

    if target is None:
        raise ValueError(f"No valid target column found. Available: {df.columns.tolist()}")

    _target = target

    X = df[_features]
    y = df[target]

    # -----------------------------
    # TRAIN / TEST SPLIT
    # -----------------------------
    split = int(len(X) * 0.8)

    X_train = X.iloc[:split]
    X_test = X.iloc[split:]
    y_train = y.iloc[:split]
    y_test = y.iloc[split:]

    # -----------------------------
    # MODEL
    # -----------------------------
    model = XGBRegressor(
        n_estimators=50,
        max_depth=5,
        learning_rate=0.1,
        random_state=42
    )

    model.fit(X_train, y_train)

    preds = model.predict(X_test)
    _mae = float(mean_absolute_error(y_test, preds))

    _model = model


def predict(steps=1):
    """
    Multi-step recursive prediction (simple XGBoost forecasting)
    """

    if _model is None:
        raise RuntimeError("Model not initialized")

    df = load_data().dropna()

    current_input = df[_features].iloc[-1:].copy()

    preds = []

    for _ in range(steps):
        pred = _model.predict(current_input)[0]
        preds.append(pred)

        # -----------------------------
        # SHIFT ONLY LAG FEATURES
        # -----------------------------
        for lag in [672, 192, 96, 24, 8, 4, 1]:
            col = f"price_lag_{lag}"

            if col in current_input.columns:
                if lag == 1:
                    current_input[col] = pred
                else:
                    current_input[col] = current_input[col].values

    return {
    "mae": float(_mae),
    "prediction": [float(x) for x in preds]
}
