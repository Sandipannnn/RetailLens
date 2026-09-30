"""
LSTM sequence builder — AI track feature utilities.
Lives in src/features/lstm_features.py so LSTM code can import from the
same package as the XGBoost feature engineering.
"""

from typing import Tuple
import numpy as np


def build_lstm_sequences(
    series: np.ndarray,
    window: int,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Converts a 1-D array into overlapping (X, y) sliding windows for LSTM.

    X shape: (n_samples, window, 1)
    y shape: (n_samples,)

    No data leakage: y[i] = series[window + i], so each target is strictly
    *after* its corresponding input window.

    Parameters
    ----------
    series : 1-D float array of sales values (already scaled if needed)
    window : Number of past time-steps per sample

    Returns
    -------
    (X, y) numpy arrays ready for Keras model.fit(X, y)
    """
    X, y = [], []
    for i in range(window, len(series)):
        X.append(series[i - window: i])
        y.append(series[i])
    X_arr = np.array(X, dtype=np.float32).reshape(-1, window, 1)
    y_arr = np.array(y, dtype=np.float32)
    return X_arr, y_arr
