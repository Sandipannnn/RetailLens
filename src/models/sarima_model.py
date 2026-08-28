"""
SARIMA (Seasonal AutoRegressive Integrated Moving Average) Forecaster for RetailLens.
Provides classical statistical time-series benchmarking against Prophet.

Role: Time-Series Forecasting
"""

import os
import logging
from typing import Dict, List, Optional, Tuple, Union
import numpy as np
import pandas as pd
from statsmodels.tsa.statespace.sarimax import SARIMAX, SARIMAXResults

from src.utils.metrics import calculate_metrics

logging.getLogger("statsmodels").setLevel(logging.WARNING)


class RetailSARIMAForecaster:
    """
    SARIMA forecasting wrapper using statsmodels.tsa.statespace.sarimax.SARIMAX.
    """

    def __init__(
        self,
        order: Tuple[int, int, int] = (1, 1, 1),
        seasonal_order: Tuple[int, int, int, int] = (1, 1, 1, 7),
        trend: Optional[str] = "c",
        enforce_stationarity: bool = False,
        enforce_invertibility: bool = False,
        interval_width: float = 0.95,
    ):
        self.order = order
        self.seasonal_order = seasonal_order
        self.trend = trend
        self.enforce_stationarity = enforce_stationarity
        self.enforce_invertibility = enforce_invertibility
        self.interval_width = interval_width

        self.fitted_model: Optional[SARIMAXResults] = None
        self.is_fitted: bool = False
        self.last_date: Optional[pd.Timestamp] = None
        self.store_id: Optional[int] = None
        self.item_id: Optional[int] = None

    @staticmethod
    def prepare_series(
        df: pd.DataFrame,
        store: Optional[int] = None,
        item: Optional[int] = None,
    ) -> pd.Series:
        """
        Extracts a clean pd.Series indexed by continuous daily DatetimeIndex.
        """
        data = df.copy()
        if store is not None and "store" in data.columns:
            data = data[data["store"] == store]
        if item is not None and "item" in data.columns:
            data = data[data["item"] == item]

        if "sales" in data.columns and "date" in data.columns:
            data = data.groupby("date", as_index=False)[["sales"]].sum()
            data["date"] = pd.to_datetime(data["date"])
            data = data.sort_values("date").set_index("date")
            series = data["sales"].asfreq("D")
        elif "ds" in data.columns and "y" in data.columns:
            data["ds"] = pd.to_datetime(data["ds"])
            data = data.sort_values("ds").set_index("ds")
            series = data["y"].asfreq("D")
        else:
            raise ValueError("DataFrame must contain 'date'/'sales' or 'ds'/'y' columns.")

        # Interpolate or forward fill any missing daily points
        if series.isnull().any():
            series = series.interpolate(method="time").ffill().bfill()

        return series

    def fit(
        self,
        df: pd.DataFrame,
        store: Optional[int] = None,
        item: Optional[int] = None,
    ) -> "RetailSARIMAForecaster":
        """
        Fits SARIMAX on the target time series.
        """
        series = self.prepare_series(df, store=store, item=item)
        self.store_id = store
        self.item_id = item
        self.last_date = series.index.max()

        model = SARIMAX(
            series,
            order=self.order,
            seasonal_order=self.seasonal_order,
            trend=self.trend,
            enforce_stationarity=self.enforce_stationarity,
            enforce_invertibility=self.enforce_invertibility,
        )
        self.fitted_model = model.fit(disp=False)
        self.is_fitted = True
        return self

    def predict(
        self,
        horizon_days: int = 90,
        clip_non_negative: bool = True,
    ) -> pd.DataFrame:
        """
        Generates out-of-sample multi-step forecasts with prediction intervals.
        """
        if not self.is_fitted or self.fitted_model is None or self.last_date is None:
            raise RuntimeError("Model must be fitted before predicting.")

        alpha = 1.0 - self.interval_width
        forecast_res = self.fitted_model.get_forecast(steps=horizon_days)
        yhat = forecast_res.predicted_mean
        conf_int = forecast_res.conf_int(alpha=alpha)

        future_dates = pd.date_range(
            start=self.last_date + pd.Timedelta(days=1),
            periods=horizon_days,
            freq="D",
        )

        lower_col = conf_int.columns[0]
        upper_col = conf_int.columns[1]

        yhat_vals = yhat.values
        lower_vals = conf_int[lower_col].values
        upper_vals = conf_int[upper_col].values

        if clip_non_negative:
            yhat_vals = np.maximum(0, yhat_vals)
            lower_vals = np.maximum(0, lower_vals)
            upper_vals = np.maximum(0, upper_vals)

        forecast_df = pd.DataFrame({
            "ds": future_dates,
            "yhat": yhat_vals,
            "yhat_lower": lower_vals,
            "yhat_upper": upper_vals,
        })

        if self.store_id is not None:
            forecast_df["store"] = self.store_id
        if self.item_id is not None:
            forecast_df["item"] = self.item_id

        return forecast_df

    def forecast_horizons(
        self,
        horizons: List[int] = [30, 60, 90],
        clip_non_negative: bool = True,
    ) -> Dict[int, pd.DataFrame]:
        """
        Generates forecasts sliced by requested horizon windows.
        """
        max_h = max(horizons)
        full_fc = self.predict(horizon_days=max_h, clip_non_negative=clip_non_negative)
        return {h: full_fc.head(h).copy().reset_index(drop=True) for h in horizons}

    def evaluate_holdout(
        self,
        df: pd.DataFrame,
        holdout_days: int = 90,
        store: Optional[int] = None,
        item: Optional[int] = None,
    ) -> Tuple[Dict[str, float], pd.DataFrame]:
        """
        Trains on historical data excluding holdout window and computes metrics against ground truth.
        """
        series = self.prepare_series(df, store=store, item=item)
        if len(series) <= holdout_days:
            raise ValueError(f"Series length ({len(series)}) must exceed holdout days ({holdout_days})")

        train_series = series.iloc[:-holdout_days]
        test_series = series.iloc[-holdout_days:]

        model = SARIMAX(
            train_series,
            order=self.order,
            seasonal_order=self.seasonal_order,
            trend=self.trend,
            enforce_stationarity=self.enforce_stationarity,
            enforce_invertibility=self.enforce_invertibility,
        )
        fitted = model.fit(disp=False)

        alpha = 1.0 - self.interval_width
        fc_res = fitted.get_forecast(steps=holdout_days)
        yhat = np.maximum(0, fc_res.predicted_mean.values)
        conf_int = fc_res.conf_int(alpha=alpha)
        lower = np.maximum(0, conf_int.iloc[:, 0].values)
        upper = np.maximum(0, conf_int.iloc[:, 1].values)

        comparison = pd.DataFrame({
            "ds": test_series.index,
            "y": test_series.values,
            "yhat": yhat,
            "yhat_lower": lower,
            "yhat_upper": upper,
        })

        metrics = calculate_metrics(
            y_true=comparison["y"],
            y_pred=comparison["yhat"],
            y_lower=comparison["yhat_lower"],
            y_upper=comparison["yhat_upper"],
        )
        return metrics, comparison

    def save_forecast(self, forecast: pd.DataFrame, output_path: str) -> None:
        """Saves forecast output to CSV."""
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        forecast.to_csv(output_path, index=False)
