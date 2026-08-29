"""
Feature engineering utilities for RetailLens.

Builds the calendar, lag, and rolling-window features consumed by the
XGBoost forecasting track (src/models/xgboost_model.py, src/models/
ml_evaluation.py) and exercised directly by tests/test_ml_evaluation.py.

Owner: Sohel Mallik (Machine Learning & Evaluation Track)
"""

from typing import List
import pandas as pd

# ---------------------------------------------------------------------------
# Defaults — shared across the ML track so training and inference never
# drift out of sync with each other.
# ---------------------------------------------------------------------------
LAG_DAYS: List[int] = [1, 7, 14, 28]
ROLLING_WINDOWS: List[int] = [7, 14, 28]
GROUP_COLS: List[str] = ["store", "item"]
DATE_COL: str = "date"
TARGET_COL: str = "sales"

_CALENDAR_COLS: List[str] = [
    "day_of_week", "day_of_month", "week_of_year",
    "month", "quarter", "year", "is_weekend",
]


def add_calendar_features(df: pd.DataFrame, date_col: str = DATE_COL) -> pd.DataFrame:
    """Adds day_of_week, day_of_month, week_of_year, month, quarter, year, is_weekend."""
    out = df.copy()
    out[date_col] = pd.to_datetime(out[date_col])

    out["day_of_week"] = out[date_col].dt.dayofweek
    out["day_of_month"] = out[date_col].dt.day
    out["week_of_year"] = out[date_col].dt.isocalendar().week.astype(int)
    out["month"] = out[date_col].dt.month
    out["quarter"] = out[date_col].dt.quarter
    out["year"] = out[date_col].dt.year
    out["is_weekend"] = (out["day_of_week"] >= 5).astype(int)

    return out


def add_lag_features(
    df: pd.DataFrame,
    lag_days: List[int] = LAG_DAYS,
    group_cols: List[str] = GROUP_COLS,
    target_col: str = TARGET_COL,
    date_col: str = DATE_COL,
) -> pd.DataFrame:
    """
    Adds lag_{N} columns (sales N days earlier), computed per group
    (store-item series) so no series bleeds into another and the first
    N rows of each series correctly come out as NaN.
    """
    out = df.copy()
    out[date_col] = pd.to_datetime(out[date_col])

    group_keys = [c for c in group_cols if c in out.columns]
    sort_cols = group_keys + [date_col] if group_keys else [date_col]
    out = out.sort_values(sort_cols).reset_index(drop=True)

    for lag in lag_days:
        if group_keys:
            out[f"lag_{lag}"] = out.groupby(group_keys)[target_col].shift(lag)
        else:
            out[f"lag_{lag}"] = out[target_col].shift(lag)

    return out


def add_rolling_features(
    df: pd.DataFrame,
    windows: List[int] = ROLLING_WINDOWS,
    group_cols: List[str] = GROUP_COLS,
    target_col: str = TARGET_COL,
    date_col: str = DATE_COL,
) -> pd.DataFrame:
    """
    Adds rolling_mean_{W} and rolling_std_{W} columns computed over the
    W days *before* the current row. The target is shifted by 1 first,
    so rolling_mean_7 at row T never depends on sales[T] — no leakage.
    """
    out = df.copy()
    out[date_col] = pd.to_datetime(out[date_col])

    group_keys = [c for c in group_cols if c in out.columns]
    sort_cols = group_keys + [date_col] if group_keys else [date_col]
    out = out.sort_values(sort_cols).reset_index(drop=True)

    if group_keys:
        shifted = out.groupby(group_keys)[target_col].shift(1)
    else:
        shifted = out[target_col].shift(1)
    out["_shifted"] = shifted

    for w in windows:
        if group_keys:
            roll_mean = out.groupby(group_keys)["_shifted"].rolling(window=w, min_periods=w).mean()
            roll_std = out.groupby(group_keys)["_shifted"].rolling(window=w, min_periods=w).std()
            roll_mean.index = roll_mean.index.droplevel(list(range(len(group_keys))))
            roll_std.index = roll_std.index.droplevel(list(range(len(group_keys))))
            out[f"rolling_mean_{w}"] = roll_mean.reindex(out.index)
            out[f"rolling_std_{w}"] = roll_std.reindex(out.index)
        else:
            out[f"rolling_mean_{w}"] = out["_shifted"].rolling(window=w, min_periods=w).mean()
            out[f"rolling_std_{w}"] = out["_shifted"].rolling(window=w, min_periods=w).std()

    out = out.drop(columns=["_shifted"])
    return out


def build_features(
    df: pd.DataFrame,
    lag_days: List[int] = LAG_DAYS,
    rolling_windows: List[int] = ROLLING_WINDOWS,
    group_cols: List[str] = GROUP_COLS,
    target_col: str = TARGET_COL,
    date_col: str = DATE_COL,
) -> pd.DataFrame:
    """
    Builds the full feature set (calendar + lag + rolling) used to train
    and evaluate the XGBoost model, starting from a cleaned sales frame.
    """
    out = add_calendar_features(df, date_col=date_col)
    out = add_lag_features(out, lag_days=lag_days, group_cols=group_cols,
                            target_col=target_col, date_col=date_col)
    out = add_rolling_features(out, windows=rolling_windows, group_cols=group_cols,
                                target_col=target_col, date_col=date_col)
    return out


def get_feature_columns(
    lag_days: List[int] = LAG_DAYS,
    rolling_windows: List[int] = ROLLING_WINDOWS,
    include_store_item: bool = True,
) -> List[str]:
    """
    Returns the ordered list of feature column names produced by
    build_features(). Used for both training and inference so the
    column order never drifts between the two.
    """
    cols = list(_CALENDAR_COLS)

    if include_store_item:
        cols += [c for c in GROUP_COLS if c not in cols]

    cols += [f"lag_{lag}" for lag in lag_days]
    cols += [f"rolling_mean_{w}" for w in rolling_windows]
    cols += [f"rolling_std_{w}" for w in rolling_windows]

    return cols