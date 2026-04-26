"""
Model registry for loading and caching trained models.
Multi-regime XGBoost + GARCH ensemble.
"""
import pickle
from pathlib import Path
from typing import Optional, Dict, Any
import warnings


class ModelRegistry:
    """Singleton model loader with caching for multi-regime ensemble."""

    _instance: Optional['ModelRegistry'] = None
    _models: Optional[Dict[str, Any]] = None

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
        return cls._instance

    def load_models(self, models_dir: Optional[Path] = None) -> Dict[str, Any]:
        """
        Load all trained models from pickle files.
        Uses singleton pattern - models are loaded only once and cached.

        Models loaded:
        - regime_classifier: XGBoost classifier for 3-regime detection
        - model_normal: XGBoost regressor for normal regime
        - model_pos: XGBoost regressor for positive spike regime
        - model_neg: XGBoost regressor for negative spike regime

            Dictionary containing all loaded models and parameters
        """
        if self._models is not None:
            return self._models

        if models_dir is None:
            # Default: grid_intelligence/models/
            package_dir = Path(__file__).parent.parent
            models_dir = package_dir / "models"

        # Check if models directory exists
        if not models_dir.exists():
            raise FileNotFoundError(
                f"Models directory not found at {models_dir}. "
                f"Please run the training notebook to generate models."
            )

        # Define model files to load
        model_files = {
            'regime_classifier': 'regime_classifier.pkl',
            'model_normal': 'model_normal.pkl',
            'model_pos': 'model_pos.pkl',
            'model_neg': 'model_neg.pkl',
            'model_config': 'model_config.pkl'
        }

        self._models = {}

        # Load each model
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore", category=UserWarning)

            for key, filename in model_files.items():
                filepath = models_dir / filename
                if not filepath.exists():
                    raise FileNotFoundError(
                        f"Model file not found: {filepath}. "
                        f"Please run the training notebook to generate all models."
                    )

                with open(filepath, 'rb') as f:
                    self._models[key] = pickle.load(f)

        print(f"✓ All models loaded successfully from {models_dir}")
        print(f"  - Regime classifier: {type(self._models['regime_classifier']).__name__}")
        print(f"  - Normal regime model: {type(self._models['model_normal']).__name__}")
        print(f"  - Positive spike model: {type(self._models['model_pos']).__name__}")
        print(f"  - Negative spike model: {type(self._models['model_neg']).__name__}")
        print(f"  - Model configuration loaded")

        return self._models
        return self._models

    def get_model_info(self) -> dict:
        """Get information about the loaded models."""
        if self._models is None:
            return {"loaded": False}

        return {
            "loaded": True,
            "ensemble_type": "Multi-Regime XGBoost",
            "regime_classifier": type(self._models['regime_classifier']).__name__,
            "regressors": {
                "normal": type(self._models['model_normal']).__name__,
                "positive_spike": type(self._models['model_pos']).__name__,
                "negative_spike": type(self._models['model_neg']).__name__
            },
            "n_features": self._models['model_normal'].n_features_in_,
            "thresholds": {
                "positive_spike": self._models['model_config']['threshold_pos'],
                "negative_spike": self._models['model_config']['threshold_neg']
            }
        }


def load_models(models_dir: Optional[Path] = None) -> Dict[str, Any]:
    """
    Convenience function to load all models using singleton pattern.

    Parameters:
    -----------
    models_dir : Path, optional
        Directory containing model files. If None, uses default location.

    Returns:
    --------
    models : dict
        Dictionary containing all loaded models (cached after first load)
    """
    registry = ModelRegistry()
    return registry.load_models(models_dir)


def get_model_info() -> dict:
    """Get information about the currently loaded models."""
    registry = ModelRegistry()
    return registry.get_model_info()
