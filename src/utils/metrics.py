"""
Time-Series Evaluation Metrics and Helpers.
Owner: Forecasting Track (RetailLens)
"""

from typing import Dict, Optional, Union
import numpy as np
import pandas as pd


def mean_absolute_error(
    y_true: Union[np.ndarray, pd.Series],
    y_pred: Union[np.ndarray, pd.Series],
) -> float:
    """Computes Mean Absolute Error (MAE)."""
    y_t = np.asarray(y_true, dtype=float)
    y_p = np.asarray(y_pred, dtype=float)
    return float(np.mean(np.abs(y_t - y_p)))


def root_mean_squared_error(
    y_true: Union[np.ndarray, pd.Series],
    y_pred: Union[np.ndarray, pd.Series],
) -> float:
    """Computes Root Mean Squared Error (RMSE)."""
    y_t = np.asarray(y_true, dtype=float)
    y_p = np.asarray(y_pred, dtype=float)
    return float(np.sqrt(np.mean((y_t - y_p) ** 2)))


def mean_absolute_percentage_error(
    y_true: Union[np.ndarray, pd.Series],
    y_pred: Union[np.ndarray, pd.Series],
    epsilon: float = 1e-5,
) -> float:
    """
    Computes Mean Absolute Percentage Error (MAPE) in percentage [0, 100].
    Epsilon is added to denominator to guard against division by zero.
    """
    y_t = np.asarray(y_true, dtype=float)
    y_p = np.asarray(y_pred, dtype=float)
    denom = np.where(np.abs(y_t) < epsilon, epsilon, y_t)
    return float(np.mean(np.abs((y_t - y_p) / denom)) * 100.0)


def symmetric_mean_absolute_percentage_error(
    y_true: Union[np.ndarray, pd.Series],
    y_pred: Union[np.ndarray, pd.Series],
    epsilon: float = 1e-5,
) -> float:
    """
    Computes Symmetric MAPE (sMAPE) in percentage [0, 100].
    """
    y_t = np.asarray(y_true, dtype=float)
    y_p = np.asarray(y_pred, dtype=float)
    denom = (np.abs(y_t) + np.abs(y_p)) / 2.0
    denom = np.where(denom < epsilon, epsilon, denom)
    return float(np.mean(np.abs(y_p - y_t) / denom) * 100.0)


def coverage_probability(
    y_true: Union[np.ndarray, pd.Series],
    y_lower: Union[np.ndarray, pd.Series],
    y_upper: Union[np.ndarray, pd.Series],
) -> float:
    """
    Calculates Empirical Coverage Probability (% of true values inside [lower, upper] interval).
    """
    y_t = np.asarray(y_true, dtype=float)
    y_l = np.asarray(y_lower, dtype=float)
    y_u = np.asarray(y_upper, dtype=float)
    in_bounds = (y_t >= y_l) & (y_t <= y_u)
    return float(np.mean(in_bounds) * 100.0)


def calculate_metrics(
    y_true: Union[np.ndarray, pd.Series],
    y_pred: Union[np.ndarray, pd.Series],
    y_lower: Optional[Union[np.ndarray, pd.Series]] = None,
    y_upper: Optional[Union[np.ndarray, pd.Series]] = None,
) -> Dict[str, float]:
    """
    Computes a comprehensive dictionary of evaluation metrics.
    """
    metrics = {
        "MAE": mean_absolute_error(y_true, y_pred),
        "RMSE": root_mean_squared_error(y_true, y_pred),
        "MAPE": mean_absolute_percentage_error(y_true, y_pred),
        "sMAPE": symmetric_mean_absolute_percentage_error(y_true, y_pred),
    }
    if y_lower is not None and y_upper is not None:
        metrics["Coverage"] = coverage_probability(y_true, y_lower, y_upper)
        # Average interval width
        metrics["Avg_Interval_Width"] = float(np.mean(np.asarray(y_upper) - np.asarray(y_lower)))
    return metrics
