"""
Forecasting Models Package for RetailLens.

Two-model architecture:
  - LSTMForecaster          : AI / Deep Learning  (TensorFlow/Keras)
  - RetailXGBoostForecaster : Traditional ML       (XGBoost tabular)
  - ModelSelector           : Selects best model by holdout RMSE

Imports are lazy so TensorFlow is only loaded when LSTM is actually used.
"""

__all__ = [
    "LSTMForecaster",
    "RetailXGBoostForecaster",
    "ModelSelector",
]


def __getattr__(name: str):  # type: ignore[misc]
    if name == "LSTMForecaster":
        from src.models.lstm_model import LSTMForecaster
        return LSTMForecaster
    if name == "RetailXGBoostForecaster":
        from src.models.xgboost_model import RetailXGBoostForecaster
        return RetailXGBoostForecaster
    if name == "ModelSelector":
        from src.models.model_selector import ModelSelector
        return ModelSelector
    raise AttributeError(f"module 'src.models' has no attribute {name!r}")
