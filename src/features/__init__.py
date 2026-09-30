"""
Features Package for RetailLens.

Modules:
  feature_engineering  : Calendar, lag, and rolling features for XGBoost (ML track)
  lstm_features        : Sliding-window sequence builder for LSTM (AI track)
"""

from src.features.feature_engineering import (
    add_calendar_features,
    add_lag_features,
    add_rolling_features,
    build_features,
    get_feature_columns,
    LAG_DAYS,
    ROLLING_WINDOWS,
    GROUP_COLS,
    DATE_COL,
    TARGET_COL,
)
from src.features.lstm_features import build_lstm_sequences

__all__ = [
    "add_calendar_features",
    "add_lag_features",
    "add_rolling_features",
    "build_features",
    "get_feature_columns",
    "build_lstm_sequences",
    "LAG_DAYS",
    "ROLLING_WINDOWS",
    "GROUP_COLS",
    "DATE_COL",
    "TARGET_COL",
]
