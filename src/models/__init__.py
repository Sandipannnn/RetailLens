"""
Forecasting Models Package for RetailLens.

Imports are lazy to avoid hard dependency failures when only a subset of
model backends is installed (e.g. running XGBoost tests without Prophet).
"""

__all__ = [
    "RetailProphetForecaster",
    "RetailSARIMAForecaster",
    "ProphetTuner",
    "RetailXGBoostForecaster",
]


def __getattr__(name: str):  # type: ignore[misc]
    if name == "RetailProphetForecaster":
        from src.models.prophet_model import RetailProphetForecaster
        return RetailProphetForecaster
    if name == "RetailSARIMAForecaster":
        from src.models.sarima_model import RetailSARIMAForecaster
        return RetailSARIMAForecaster
    if name == "ProphetTuner":
        from src.models.hyperparameter_tuner import ProphetTuner
        return ProphetTuner
    if name == "RetailXGBoostForecaster":
        from src.models.xgboost_model import RetailXGBoostForecaster
        return RetailXGBoostForecaster
    raise AttributeError(f"module 'src.models' has no attribute {name!r}")
