import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from datetime import datetime
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



def predict() -> dict:
    """
    Predict energy prices for the next 288 timesteps (72 hours) using multi-regime XGBoost.

    Input:  None
    Output: dict with 288 predictions (72 hours at 15-min intervals)

    Returns:
    --------
    dict
        Predictions with timestamps and metadata
    """
    try:
        # Generate features for prediction using default nrows (1632)
        df = generate_features()

        # drop datetime column and price column for prediction
        predict_df = df.drop(columns=['datetime_utc', 'price'])

        # use predict_df to perform forecasting
        predictions = predict_multi_regime(predict_df)

        # Round predictions
        predictions_list = [round(float(pred), 2) for pred in predictions]
        #predictions_series = pd.Series(predictions).rolling(window=4, center=True, min_periods=1).mean()
        #predictions_list = [round(float(p), 2) for p in predictions_series]


        timestamp_strings = [ts.strftime('%Y-%m-%d %H:%M:%S') for ts in timestamps]

        return {
            "message": df.shape,
            "datetime": datetime,
            "predictions": predictions,
            "actual_price": actual_prices,
            "unit": "EUR/MWh",
            "model_type": "Multi-Regime XGBoost",
        }



    except Exception as e:
        return {
            "error": str(e),
            "message": "Prediction failed. Please check that models are trained and saved.",
            "help": "Run the training notebook (temp.ipynb) and save models first."
        }


def plot_predictions(predictions=None, actual_prices=None, timestamps=None,
                    save_path=None, show=True, figsize=(15, 6)):
    """
    Plot predicted vs actual electricity prices.

    Parameters:
    -----------
    predictions : list or array-like, optional
        Predicted prices. If None, calls predict() to get predictions.
    actual_prices : list or array-like, optional
        Actual prices. If None, calls predict() to get actual prices.
    timestamps : list or array-like, optional
        Timestamps for x-axis. If None, uses indices.
    save_path : str, optional
        Path to save the plot (e.g., 'predictions_plot.png'). If None, doesn't save.
    show : bool, default=True
        Whether to display the plot.
    figsize : tuple, default=(15, 6)
        Figure size (width, height) in inches.

    Returns:
    --------
    fig, ax : matplotlib figure and axes objects
    """
    # If no data provided, get predictions
    if predictions is None or actual_prices is None:
        result = predict()
        if "error" in result:
            print(f"Error getting predictions: {result['error']}")
            return None, None
        predictions = result.get('predictions', [])
        actual_prices = result.get('actual_price', [])
        timestamps = result.get('datetime')

    # Create figure and axis
    fig, ax = plt.subplots(figsize=figsize)

    # Determine x-axis values
    if timestamps is not None:
        # Parse timestamps if they're strings
        if isinstance(timestamps, str):
            start_time = datetime.strptime(timestamps, '%Y-%m-%d %H:%M:%S')
            x_values = [start_time + pd.Timedelta(minutes=15*i) for i in range(len(predictions))]
        elif isinstance(timestamps, list) and len(timestamps) > 0:
            if isinstance(timestamps[0], str):
                x_values = [datetime.strptime(t, '%Y-%m-%d %H:%M:%S') for t in timestamps]
            else:
                x_values = timestamps
        else:
            x_values = timestamps
        use_dates = True
    else:
        x_values = range(len(predictions))
        use_dates = False

    # Plot actual prices
    ax.plot(x_values[:len(actual_prices)], actual_prices,
            label='Actual Prices', color='blue', linewidth=2, alpha=0.7)

    # Plot predictions
    ax.plot(x_values[:len(predictions)], predictions,
            label='Predicted Prices', color='red', linewidth=2, alpha=0.7, linestyle='--')

    # Formatting
    ax.set_xlabel('Time' if use_dates else 'Timestep (15-min intervals)', fontsize=12)
    ax.set_ylabel('Price (EUR/MWh)', fontsize=12)
    ax.set_title('Electricity Price: Predictions vs Actual', fontsize=14, fontweight='bold')
    ax.legend(loc='best', fontsize=11)
    ax.grid(True, alpha=0.3, linestyle='--')

    # Format x-axis for dates
    if use_dates:
        ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d %H:%M'))
        plt.xticks(rotation=45, ha='right')

    # Add statistics box
    mse = np.mean((np.array(predictions[:len(actual_prices)]) - np.array(actual_prices))**2)
    mae = np.mean(np.abs(np.array(predictions[:len(actual_prices)]) - np.array(actual_prices)))
    stats_text = f'MSE: {mse:.2f}\nMAE: {mae:.2f}'
    ax.text(0.02, 0.98, stats_text, transform=ax.transAxes,
            verticalalignment='top', bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5),
            fontsize=10)

    plt.tight_layout()

    # Save if path provided
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Plot saved to: {save_path}")

    # Show if requested
    if show:
        plt.show()

    return fig, ax
