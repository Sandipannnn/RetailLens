"""
XGBoost Forecasting Engine for RetailLens.

Implements a one-step-ahead XGBoost regression model for retail daily sales
forecasting. The model is trained on lag, rolling, and calendar features
derived from the feature_engineering module.

Evaluation protocol: one-step-ahead on a chronological hold-out test set.
The hold-out features are computed from actual historical sales values (as
would be available in a real production pipeline up to that date).  This is
the standard and defensible approach for multi-series tabular forecasting.

Model persistence uses joblib.
"""

import logging
import os
from typing import Dict, List, Optional, Tuple

import joblib
import numpy as np
import pandas as pd
from xgboost import XGBRegressor

from src.features.feature_engineering import (
    LAG_DAYS,
    ROLLING_WINDOWS,
    GROUP_COLS,
    DATE_COL,
    TARGET_COL,
    build_features,
    get_feature_columns,
)
from src.utils.metrics import calculate_metrics

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Default XGBoost hyperparameters — a reliable baseline, not over-tuned
# ---------------------------------------------------------------------------
DEFAULT_XGBOOST_PARAMS: Dict = {
    "n_estimators": 500,
    "max_depth": 6,
    "learning_rate": 0.05,
    "subsample": 0.8,
    "colsample_bytree": 0.8,
    "min_child_weight": 5,
    "reg_alpha": 0.1,
    "reg_lambda": 1.0,
    "random_state": 42,
    "n_jobs": -1,
    "verbosity": 0,
    "objective": "reg:squarederror",
}


class RetailXGBoostForecaster:
    """
    XGBoost regression model for retail daily sales forecasting.

    Usage
    -----
    forecaster = RetailXGBoostForecaster()
    metrics, comparison_df = forecaster.evaluate_holdout(df, holdout_days=90, store=1, item=1)
    forecaster.fit(df)           # retrain on full data
    future_df = forecaster.predict_future(df, horizon_days=90)
    """

    def __init__(
        self,
        xgb_params: Optional[Dict] = None,
        lag_days: List[int] = LAG_DAYS,
        rolling_windows: List[int] = ROLLING_WINDOWS,
        group_cols: List[str] = GROUP_COLS,
        target_col: str = TARGET_COL,
        date_col: str = DATE_COL,
    ):
        self.xgb_params = xgb_params or DEFAULT_XGBOOST_PARAMS.copy()
        self.lag_days = lag_days
        self.rolling_windows = rolling_windows
        self.group_cols = group_cols
        self.target_col = target_col
        self.date_col = date_col

        self.model: Optional[XGBRegressor] = None
        self.feature_cols: List[str] = get_feature_columns(
            lag_days=lag_days,
            rolling_windows=rolling_windows,
            include_store_item=True,
        )
        self.is_fitted: bool = False

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _build_and_clean(self, df: pd.DataFrame) -> pd.DataFrame:
        """Builds features and drops rows with NaN lag/rolling values."""
        featured = build_features(
            df,
            lag_days=self.lag_days,
            rolling_windows=self.rolling_windows,
            group_cols=self.group_cols,
            target_col=self.target_col,
            date_col=self.date_col,
        )
        # Only drop NaN in the feature columns we will actually use
        featured = featured.dropna(subset=self.feature_cols).reset_index(drop=True)
        return featured

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def fit(
        self,
        df: pd.DataFrame,
        store: Optional[int] = None,
        item: Optional[int] = None,
    ) -> "RetailXGBoostForecaster":
        """
        Builds features and fits the XGBoost model.

        Parameters
        ----------
        df    : Cleaned sales DataFrame (date, store, item, sales …)
        store : If given, filter to this store before training
        item  : If given, filter to this item before training
        """
        data = df.copy()
        if store is not None:
            data = data[data["store"] == store]
        if item is not None:
            data = data[data["item"] == item]

        if data.empty:
            raise ValueError(f"No data found for store={store}, item={item}.")

        featured = self._build_and_clean(data)

        if featured.empty:
            raise ValueError("No rows remain after dropping NaN lag features. "
                             "Ensure sufficient historical data is available.")

        X = featured[self.feature_cols].values
        y = featured[self.target_col].values

        self.model = XGBRegressor(**self.xgb_params)
        self.model.fit(X, y)
        self.is_fitted = True
        logger.info("XGBoost model fitted on %d rows, %d features.", len(X), X.shape[1])
        return self

    def evaluate_holdout(
        self,
        df: pd.DataFrame,
        holdout_days: int = 90,
        store: Optional[int] = None,
        item: Optional[int] = None,
    ) -> Tuple[Dict[str, float], pd.DataFrame]:
        """
        Chronological hold-out evaluation.

        The split is performed on the raw time-series before feature
        engineering.  Features for the hold-out rows are derived from
        historical values (including the training period), ensuring no
        leakage of future target values.

        Leakage guard
        -------------
        Only rows whose *date* falls in the training window are used to
        compute lags/rolling stats for training.  For test rows, lag_N at
        date T uses actual sales at T-N — all of which lie in the past and
        were already observed before the test period begins.

        Parameters
        ----------
        df           : Full cleaned DataFrame
        holdout_days : Number of trailing calendar days to reserve for testing
        store        : Optional store filter
        item         : Optional item filter

        Returns
        -------
        (metrics_dict, comparison_DataFrame)
            metrics_dict keys: MAE, RMSE, MAPE, sMAPE
            comparison_DataFrame columns: date, [store], [item], actual_sales,
                                          predicted_sales
        """
        data = df.copy()
        if store is not None:
            data = data[data["store"] == store]
        if item is not None:
            data = data[data["item"] == item]

        if data.empty:
            raise ValueError(f"No data found for store={store}, item={item}.")

        data[self.date_col] = pd.to_datetime(data[self.date_col])
        data = data.sort_values(self.group_cols + [self.date_col]).reset_index(drop=True)

        max_date = data[self.date_col].max()
        split_date = max_date - pd.Timedelta(days=holdout_days - 1)

        train_mask = data[self.date_col] < split_date
        test_mask = data[self.date_col] >= split_date

        if train_mask.sum() == 0:
            raise ValueError(f"No training rows before split date {split_date}.")
        if test_mask.sum() == 0:
            raise ValueError(f"No test rows from {split_date} onward.")

        # Build features on the FULL dataset (history is shared between
        # train/test for lag computation — but the TARGET in test rows is
        # never used as a feature for any row).
        featured = build_features(
            data,
            lag_days=self.lag_days,
            rolling_windows=self.rolling_windows,
            group_cols=self.group_cols,
            target_col=self.target_col,
            date_col=self.date_col,
        )

        train_df = featured[featured[self.date_col] < split_date].dropna(subset=self.feature_cols)
        test_df = featured[featured[self.date_col] >= split_date].dropna(subset=self.feature_cols)

        if train_df.empty:
            raise ValueError("Training set is empty after feature engineering and NaN removal.")
        if test_df.empty:
            raise ValueError("Test set is empty after feature engineering and NaN removal.")

        # Verify chronological order — guard against accidental future leakage
        assert train_df[self.date_col].max() < test_df[self.date_col].min(), (
            "LEAKAGE DETECTED: Training dates overlap with test dates."
        )

        X_train = train_df[self.feature_cols].values
        y_train = train_df[self.target_col].values
        X_test = test_df[self.feature_cols].values
        y_test = test_df[self.target_col].values

        model = XGBRegressor(**self.xgb_params)
        model.fit(X_train, y_train)

        y_pred = np.maximum(0.0, model.predict(X_test))

        metrics = calculate_metrics(y_true=y_test, y_pred=y_pred)

        comparison = pd.DataFrame({
            self.date_col: test_df[self.date_col].values,
            "actual_sales": y_test,
            "predicted_sales": y_pred,
        })
        if store is not None:
            comparison["store"] = store
        if item is not None:
            comparison["item"] = item

        # Store trained model for subsequent calls
        self.model = model
        self.is_fitted = True

        return metrics, comparison

    def get_feature_importance(self) -> pd.DataFrame:
        """
        Returns a DataFrame of feature importances sorted descending.

        Requires the model to be fitted first.

        Returns
        -------
        DataFrame with columns: feature, importance (normalised gain score)
        """
        if not self.is_fitted or self.model is None:
            raise RuntimeError("Model must be fitted before accessing feature importance.")

        scores = self.model.feature_importances_
        importance_df = pd.DataFrame({
            "feature": self.feature_cols,
            "importance": scores,
        }).sort_values("importance", ascending=False).reset_index(drop=True)
        return importance_df

    def save(self, path: str) -> None:
        """Persists the trained model to disk using joblib."""
        if not self.is_fitted or self.model is None:
            raise RuntimeError("No fitted model to save.")
        os.makedirs(os.path.dirname(path), exist_ok=True)
        joblib.dump(self.model, path)
        logger.info("XGBoost model saved to %s", path)

    @classmethod
    def load(cls, path: str, **kwargs) -> "RetailXGBoostForecaster":
        """
        Loads a previously saved XGBoost model and returns a fitted forecaster.

        Parameters
        ----------
        path   : File path to the saved joblib model
        kwargs : Additional keyword arguments forwarded to __init__
        """
        if not os.path.exists(path):
            raise FileNotFoundError(f"Model file not found: {path}")
        instance = cls(**kwargs)
        instance.model = joblib.load(path)
        instance.is_fitted = True
        logger.info("XGBoost model loaded from %s", path)
        return instance

    def predict_future(
        self,
        df: pd.DataFrame,
        horizon_days: int = 90,
        store: Optional[int] = None,
        item: Optional[int] = None,
    ) -> pd.DataFrame:
        """
        Generates a future forecast for the specified horizon using the
        trained model.  This method uses the *actual* last observed sales
        values to compute lag features for the first horizon step, then
        rolls forward using predicted values (recursive strategy).

        Note: This is a multi-step recursive forecast.  Prediction error
        accumulates over longer horizons; this is expected behaviour.

        Parameters
        ----------
        df           : Full cleaned DataFrame (used to seed lags)
        horizon_days : Number of future days to forecast
        store        : Store to forecast (required if multiple stores in df)
        item         : Item to forecast (required if multiple items in df)

        Returns
        -------
        DataFrame with columns: ds (date), yhat (predicted sales)
        """
        if not self.is_fitted or self.model is None:
            raise RuntimeError("Model must be fitted before predicting.")

        data = df.copy()
        if store is not None:
            data = data[data["store"] == store]
        if item is not None:
            data = data[data["item"] == item]

        if data.empty:
            raise ValueError(f"No seed data found for store={store}, item={item}.")

        data[self.date_col] = pd.to_datetime(data[self.date_col])
        data = data.sort_values(self.date_col).reset_index(drop=True)

        last_date = data[self.date_col].max()
        future_dates = pd.date_range(
            start=last_date + pd.Timedelta(days=1),
            periods=horizon_days,
            freq="D",
        )

        # Seed history: keep only sales column indexed by date
        history = data.set_index(self.date_col)[self.target_col].copy()

        predictions = []
        for future_date in future_dates:
            row = {self.date_col: future_date}

            # Calendar features
            row["day_of_week"] = future_date.dayofweek
            row["day_of_month"] = future_date.day
            row["week_of_year"] = future_date.isocalendar()[1]
            row["month"] = future_date.month
            row["quarter"] = (future_date.month - 1) // 3 + 1
            row["year"] = future_date.year
            row["is_weekend"] = int(future_date.dayofweek >= 5)

            # Store / item identifiers
            if store is not None:
                row["store"] = store
            if item is not None:
                row["item"] = item

            # Lag features
            for lag in self.lag_days:
                lag_date = future_date - pd.Timedelta(days=lag)
                row[f"lag_{lag}"] = float(history.get(lag_date, np.nan))

            # Rolling features (mean and std over the last W values before future_date)
            for w in self.rolling_windows:
                window_vals = [
                    history.get(future_date - pd.Timedelta(days=d), np.nan)
                    for d in range(1, w + 1)
                ]
                valid = [v for v in window_vals if not np.isnan(v)]
                row[f"rolling_mean_{w}"] = float(np.mean(valid)) if valid else np.nan
                row[f"rolling_std_{w}"] = float(np.std(valid, ddof=1)) if len(valid) >= 2 else np.nan

            features = np.array([[row.get(c, np.nan) for c in self.feature_cols]])
            yhat = float(np.maximum(0.0, self.model.predict(features)[0]))
            predictions.append(yhat)

            # Add prediction to history for subsequent lag computation
            history[future_date] = yhat

        return pd.DataFrame({
            "ds": future_dates,
            "yhat": predictions,
            "store": store,
            "item": item,
        })
