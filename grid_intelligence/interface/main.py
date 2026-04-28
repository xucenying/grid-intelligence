import pandas as pd
import numpy as np
from grid_intelligence.logic.registry import load_models
from grid_intelligence.logic.preprocessor import generate_features
import time

_models = None
_feature_cache = None
_cache_timestamp = 0
CACHE_TTL = 900  # 15 minutes

def _get_models():
    global _models
    if _models is None:
        _models = load_models()
    return _models

def _get_features():
    global _feature_cache, _cache_timestamp
    now = time.time()
    print (_cache_timestamp)
    if _feature_cache is None or (now - _cache_timestamp) > CACHE_TTL:
        print (now)
        print("Refreshing feature cache...")
        _feature_cache = generate_features(nrows=5000)
        _cache_timestamp = now
    return _feature_cache

def predict_multi_regime(features: pd.DataFrame) -> np.ndarray:
    models = _get_models()
    clf = models['regime_classifier']
    model_normal = models['model_normal']
    model_pos = models['model_pos']
    model_neg = models['model_neg']
    config = models['model_config']

    probs = clf.predict_proba(features)
    p_normal = probs[:, 0]
    p_pos = probs[:, 1]
    p_neg = probs[:, 2]

    pred_normal = model_normal.predict(features)
    pred_pos = model_pos.predict(features)
    pred_neg = model_neg.predict(features)

    scaling = config['scaling_factors']
    predictions = (
        p_neg * pred_neg * scaling['neg'] +
        p_normal * pred_normal * scaling['normal'] +
        p_pos * pred_pos * scaling['pos']
    )
    return predictions

def predict() -> dict:
    try:
        df = _get_features()

        # Drop columns not used as features
        predict_df = df.drop(columns=[c for c in ['datetime_utc', 'price', 'target_288', 'regime'] if c in df.columns])

        # Direct multi-output — no iterative loop
        predictions = predict_multi_regime(predict_df)
        predictions_list = [round(float(p), 2) for p in predictions[-288:]]

        #start_time = pd.Timestamp.now(tz='UTC').round('15min')
        #TODO
        df = df[df['datetime_utc'] <= pd.Timestamp.now(tz='UTC')]
        start_time = df['datetime_utc'].iloc[-1]
        timestamps = pd.date_range(start=start_time, periods=288, freq='15min', tz='UTC')
        timestamp_strings = [ts.strftime('%Y-%m-%d %H:%M:%S+00:00') for ts in timestamps]

        return {
            "start_time": start_time.strftime('%Y-%m-%d %H:%M:%S'),
            "predictions_15min": predictions_list,
            "timestamps": timestamp_strings,
            "intervals": 288,
            "hours_covered": 72,
            "unit": "EUR/MWh",
            "model_type": "Multi-Regime XGBoost",
            "forecast_method": "Direct multi-output"
        }

    except Exception as e:
        return {
            "error": str(e),
            "message": "Prediction failed. Please check that models are trained and saved.",
            "help": "Run the training notebook (temp.ipynb) and save models first."
        }
