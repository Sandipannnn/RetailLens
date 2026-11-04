"""Approximate prediction ranges based on observed holdout absolute errors."""
import numpy as np
import pandas as pd


def add_prediction_bounds(forecast: pd.DataFrame, actual, predicted) -> pd.DataFrame:
    """Add a symmetric 90% empirical error range; future coverage is not guaranteed."""
    errors = np.abs(np.asarray(actual, dtype=float) - np.asarray(predicted, dtype=float))
    if errors.size == 0 or not np.isfinite(errors).all():
        raise ValueError("Finite holdout errors are required to estimate prediction bounds.")
    width = float(np.quantile(errors, 0.9, method="higher"))
    result = forecast.copy()
    result["yhat_lower"] = np.maximum(0, result["yhat"] - width)
    result["yhat_upper"] = result["yhat"] + width
    return result
