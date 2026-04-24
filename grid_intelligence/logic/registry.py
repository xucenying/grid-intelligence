"""
Model registry for loading and caching trained models.
"""
import pickle
from pathlib import Path
from typing import Optional
import warnings


class ModelRegistry:
    """Singleton model loader with caching."""
    
    _instance: Optional['ModelRegistry'] = None
    _model = None
    
    def __new__(cls):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
        return cls._instance
    
    def load_model(self, model_path: Optional[Path] = None):
        """
        Load the trained model from pickle file.
        Uses singleton pattern - model is loaded only once and cached.
        
        Parameters:
        -----------
        model_path : Path, optional
            Path to model.pkl file. If None, uses default location.
            
        Returns:
        --------
        model : XGBRegressor
            Trained XGBoost model
        """
        if self._model is not None:
            return self._model
        
        if model_path is None:
            # Default: grid_intelligence/models/model.pkl
            package_dir = Path(__file__).parent.parent
            model_path = package_dir / "models" / "model.pkl"
        
        if not model_path.exists():
            raise FileNotFoundError(
                f"Model file not found at {model_path}. "
                f"Please ensure model.pkl is in grid_intelligence/models/"
            )
        
        # Suppress XGBoost version warning
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore", category=UserWarning)
            with open(model_path, 'rb') as f:
                self._model = pickle.load(f)
        
        print(f"✓ Model loaded successfully from {model_path}")
        print(f"  - Type: {type(self._model).__name__}")
        print(f"  - Features: {self._model.n_features_in_}")
        
        return self._model
    
    def get_model_info(self) -> dict:
        """Get information about the loaded model."""
        if self._model is None:
            return {"loaded": False}
        
        return {
            "loaded": True,
            "type": type(self._model).__name__,
            "n_features": self._model.n_features_in_,
            "feature_names": list(self._model.feature_names_in_) if hasattr(self._model, 'feature_names_in_') else None
        }


def load_model(model_path: Optional[Path] = None):
    """
    Convenience function to load model using singleton pattern.
    
    Parameters:
    -----------
    model_path : Path, optional
        Path to model.pkl file. If None, uses default location.
        
    Returns:
    --------
    model : XGBRegressor
        Trained XGBoost model (cached after first load)
    """
    registry = ModelRegistry()
    return registry.load_model(model_path)


def get_model_info() -> dict:
    """Get information about the currently loaded model."""
    registry = ModelRegistry()
    return registry.get_model_info()
