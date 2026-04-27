"""
Model registry for loading and caching trained models.
Multi-regime XGBoost + GARCH ensemble.
"""
import pickle
import os
from pathlib import Path
from typing import Optional, Dict, Any
import warnings


BUCKET_NAME = "grid-intelligence-models"
MODEL_FILES = [
    'regime_classifier.pkl',
    'model_normal.pkl',
    'model_pos.pkl',
    'model_neg.pkl',
    'model_config.pkl'
]


def _download_from_gcs(models_dir: Path):
    """Download model files from GCS bucket to local directory."""
    from google.cloud import storage
    print(f"Downloading models from gs://{BUCKET_NAME}/...")
    client = storage.Client()
    bucket = client.bucket(BUCKET_NAME)
    for filename in MODEL_FILES:
        blob = bucket.blob(filename)
        dest = models_dir / filename
        blob.download_to_filename(dest)
        print(f"  ✅ Downloaded: {filename}")


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
        Downloads from GCS if not available locally.
        Uses singleton pattern - models are loaded only once and cached.
        """
        if self._models is not None:
            return self._models

        if models_dir is None:
            package_dir = Path(__file__).parent.parent
            models_dir = package_dir / "models"

        models_dir.mkdir(exist_ok=True)

        # Check if models exist locally — if not, download from GCS
        missing = [f for f in MODEL_FILES if not (models_dir / f).exists()]
        if missing:
            print(f"Models not found locally: {missing}")
            _download_from_gcs(models_dir)

        # Load models from local files
        self._models = {}
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore", category=UserWarning)
            for filename in MODEL_FILES:
                key = filename.replace('.pkl', '')
                filepath = models_dir / filename
                with open(filepath, 'rb') as f:
                    self._models[key] = pickle.load(f)

        print(f"✅ All models loaded from {models_dir}")
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
    """Convenience function to load all models using singleton pattern."""
    registry = ModelRegistry()
    return registry.load_models(models_dir)


def get_model_info() -> dict:
    """Get information about the currently loaded models."""
    registry = ModelRegistry()
    return registry.get_model_info()
