"""
Forecasting Models Package for RetailLens.

Imports are lazy to avoid hard dependency failures when only a subset of
model backends is installed (e.g. running XGBoost tests without Prophet,
or running Prophet tests without TensorFlow).

Primary models (v2 architecture):
  - LSTMForecaster        : AI / Deep Learning (requires tensorflow)
  - RetailXGBoostForecaster : Traditional Machine Learning
  - ModelSelector         : Compares models and selects best by RMSE/MAE

Legacy / companion models (retained, fully functional):
  - RetailProphetForecaster
  - RetailSARIMAForecaster
  - ProphetTuner
"""

__all__ = [
    # AI + ML (new)
    "LSTMForecaster",
    "ModelSelector",
    # ML track (upstream)
    "RetailXGBoostForecaster",
    # Legacy (upstream)
    "RetailProphetForecaster",
    "RetailSARIMAForecaster",
    "ProphetTuner",
]


def __getattr__(name: str):  # type: ignore[misc]
    # ---- New AI/ML layer ----
    if name == "LSTMForecaster":
        from src.models.lstm_model import LSTMForecaster
        return LSTMForecaster
    if name == "ModelSelector":
        from src.models.model_selector import ModelSelector
        return ModelSelector
    # ---- Upstream ML track ----
    if name == "RetailXGBoostForecaster":
        from src.models.xgboost_model import RetailXGBoostForecaster
        return RetailXGBoostForecaster
    # ---- Legacy ----
    if name == "RetailProphetForecaster":
        from src.models.prophet_model import RetailProphetForecaster
        return RetailProphetForecaster
    if name == "RetailSARIMAForecaster":
        from src.models.sarima_model import RetailSARIMAForecaster
        return RetailSARIMAForecaster
    if name == "ProphetTuner":
        from src.models.hyperparameter_tuner import ProphetTuner
        return ProphetTuner
    raise AttributeError(f"module 'src.models' has no attribute {name!r}")
