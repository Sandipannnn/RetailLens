"""
Prophet Forecasting Engine for RetailLens.
Implements single-series and batch retail forecasting with multi-step horizons (30/60/90 days),
uncertainty quantification (confidence intervals), and seasonal component extraction.

Owner: Rudra Pratap Singh (Time-Series Forecasting Track)
"""

import os
import logging
from typing import Dict, List, Optional, Tuple, Union
import numpy as np
import pandas as pd
from prophet import Prophet
import matplotlib.pyplot as plt

from src.utils.metrics import calculate_metrics

# Suppress Prophet / cmdstanpy verbose logs
logging.getLogger("prophet").setLevel(logging.WARNING)
logging.getLogger("cmdstanpy").setLevel(logging.WARNING)


class RetailProphetForecaster:
    """
    High-level wrapper for Facebook Prophet tailored for retail demand forecasting.
    """

    def __init__(
        self,
        growth: str = "linear",
        changepoint_prior_scale: float = 0.05,
        seasonality_prior_scale: float = 10.0,
        holidays_prior_scale: float = 10.0,
        seasonality_mode: str = "additive",
        changepoint_range: float = 0.8,
        interval_width: float = 0.95,
        yearly_seasonality: Union[bool, str, int] = "auto",
        weekly_seasonality: Union[bool, str, int] = "auto",
        daily_seasonality: Union[bool, str, int] = False,
        country_holidays: Optional[str] = "US",
    ):
        self.growth = growth
        self.changepoint_prior_scale = changepoint_prior_scale
        self.seasonality_prior_scale = seasonality_prior_scale
        self.holidays_prior_scale = holidays_prior_scale
        self.seasonality_mode = seasonality_mode
        self.changepoint_range = changepoint_range
        self.interval_width = interval_width
        self.yearly_seasonality = yearly_seasonality
        self.weekly_seasonality = weekly_seasonality
        self.daily_seasonality = daily_seasonality
        self.country_holidays = country_holidays

        self.model: Optional[Prophet] = None
        self.is_fitted: bool = False
        self.last_train_date: Optional[pd.Timestamp] = None
        self.store_id: Optional[int] = None
        self.item_id: Optional[int] = None

    def _init_model(self) -> Prophet:
        """Instantiates a fresh Prophet model configured with instance parameters."""
        model = Prophet(
            growth=self.growth,
            changepoint_prior_scale=self.changepoint_prior_scale,
            seasonality_prior_scale=self.seasonality_prior_scale,
            holidays_prior_scale=self.holidays_prior_scale,
            seasonality_mode=self.seasonality_mode,
            changepoint_range=self.changepoint_range,
            interval_width=self.interval_width,
            yearly_seasonality=self.yearly_seasonality,
            weekly_seasonality=self.weekly_seasonality,
            daily_seasonality=self.daily_seasonality,
        )
        if self.country_holidays:
            try:
                model.add_country_holidays(country_name=self.country_holidays)
            except Exception as e:
                logging.warning(f"Could not add country holidays '{self.country_holidays}': {e}")
        return model

    @staticmethod
    def prepare_data(
        df: pd.DataFrame,
        store: Optional[int] = None,
        item: Optional[int] = None,
    ) -> pd.DataFrame:
        """
        Filters and formats raw/processed DataFrame into Prophet standard format ('ds', 'y').
        """
        data = df.copy()
        if store is not None and "store" in data.columns:
            data = data[data["store"] == store]
        if item is not None and "item" in data.columns:
            data = data[data["item"] == item]

        # If aggregated over multiple series
        if "sales" in data.columns:
            if "date" in data.columns:
                data = data.groupby("date", as_index=False)["sales"].sum()
                data = data.rename(columns={"date": "ds", "sales": "y"})
        elif "ds" not in data.columns or "y" not in data.columns:
            raise ValueError("DataFrame must contain 'date'/'sales' or 'ds'/'y' columns.")

        data["ds"] = pd.to_datetime(data["ds"])
        data["y"] = pd.to_numeric(data["y"])
        data = data.sort_values("ds").reset_index(drop=True)
        return data

    def fit(
        self,
        df: pd.DataFrame,
        store: Optional[int] = None,
        item: Optional[int] = None,
    ) -> "RetailProphetForecaster":
        """
        Fits the Prophet model on the specified series or pre-formatted DataFrame.
        """
        data = self.prepare_data(df, store=store, item=item)
        self.store_id = store
        self.item_id = item
        self.last_train_date = data["ds"].max()

        self.model = self._init_model()
        self.model.fit(data)
        self.is_fitted = True
        return self

    def predict(
        self,
        horizon_days: int = 90,
        freq: str = "D",
        include_history: bool = False,
        clip_non_negative: bool = True,
    ) -> pd.DataFrame:
        """
        Generates out-of-sample multi-step forecasts with confidence intervals.
        
        Returns:
            DataFrame with columns ['ds', 'yhat', 'yhat_lower', 'yhat_upper', ...]
        """
        if not self.is_fitted or self.model is None:
            raise RuntimeError("Model must be fitted before predicting.")

        future = self.model.make_future_dataframe(
            periods=horizon_days,
            freq=freq,
            include_history=include_history,
        )
        forecast = self.model.predict(future)

        if clip_non_negative:
            for col in ["yhat", "yhat_lower", "yhat_upper"]:
                if col in forecast.columns:
                    forecast[col] = forecast[col].clip(lower=0)

        # Attach metadata tags
        if self.store_id is not None:
            forecast["store"] = self.store_id
        if self.item_id is not None:
            forecast["item"] = self.item_id

        return forecast

    def forecast_horizons(
        self,
        horizons: List[int] = [30, 60, 90],
        clip_non_negative: bool = True,
    ) -> Dict[int, pd.DataFrame]:
        """
        Convenience method to generate multi-step forecasts for standard horizons (e.g., 30, 60, 90 days).
        
        Returns:
            Dictionary mapping horizon (e.g. 30, 60, 90) -> Forecast DataFrame
        """
        max_h = max(horizons)
        full_forecast = self.predict(horizon_days=max_h, include_history=False, clip_non_negative=clip_non_negative)
        
        results = {}
        for h in horizons:
            results[h] = full_forecast.head(h).copy().reset_index(drop=True)
        return results

    def evaluate_holdout(
        self,
        df: pd.DataFrame,
        holdout_days: int = 90,
        store: Optional[int] = None,
        item: Optional[int] = None,
    ) -> Tuple[Dict[str, float], pd.DataFrame]:
        """
        Trains on history up to (max_date - holdout_days) and evaluates on the holdout test window.
        
        Returns:
            (metrics_dict, holdout_comparison_df)
        """
        data = self.prepare_data(df, store=store, item=item)
        if len(data) <= holdout_days:
            raise ValueError(f"Data length ({len(data)}) must be strictly greater than holdout_days ({holdout_days})")

        train_data = data.iloc[:-holdout_days].copy()
        test_data = data.iloc[-holdout_days:].copy()

        eval_model = self._init_model()
        eval_model.fit(train_data)

        future = eval_model.make_future_dataframe(periods=holdout_days, freq="D", include_history=False)
        forecast = eval_model.predict(future)

        # Clip non-negative
        for col in ["yhat", "yhat_lower", "yhat_upper"]:
            forecast[col] = forecast[col].clip(lower=0)

        comparison = test_data.merge(forecast[["ds", "yhat", "yhat_lower", "yhat_upper"]], on="ds", how="inner")
        metrics = calculate_metrics(
            y_true=comparison["y"],
            y_pred=comparison["yhat"],
            y_lower=comparison["yhat_lower"],
            y_upper=comparison["yhat_upper"],
        )
        return metrics, comparison

    def extract_components(self, forecast: pd.DataFrame) -> Dict[str, pd.DataFrame]:
        """
        Extracts individual decomposed time-series components (trend, weekly, yearly, holidays).
        """
        components = {}
        if "trend" in forecast.columns:
            components["trend"] = forecast[["ds", "trend"]].copy()
        if "weekly" in forecast.columns:
            components["weekly"] = forecast[["ds", "weekly"]].copy()
        if "yearly" in forecast.columns:
            components["yearly"] = forecast[["ds", "yearly"]].copy()
        if "holidays" in forecast.columns:
            components["holidays"] = forecast[["ds", "holidays"]].copy()
        return components

    def save_forecast(
        self,
        forecast: pd.DataFrame,
        output_path: str,
    ) -> None:
        """Saves generated forecast DataFrame to disk."""
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        cols_to_save = [c for c in ["ds", "store", "item", "yhat", "yhat_lower", "yhat_upper", "trend", "weekly", "yearly"] if c in forecast.columns]
        forecast[cols_to_save].to_csv(output_path, index=False)
        logging.info(f"Saved forecast to {output_path}")


def batch_forecast_prophet(
    df: pd.DataFrame,
    series_list: List[Tuple[int, int]],
    horizons: List[int] = [30, 60, 90],
    output_dir: str = "data/processed/forecasts",
) -> pd.DataFrame:
    """
    Executes Prophet forecasting across multiple (store, item) series and aggregates the results.
    """
    os.makedirs(output_dir, exist_ok=True)
    all_forecasts = []
    max_h = max(horizons)

    for store, item in series_list:
        forecaster = RetailProphetForecaster()
        forecaster.fit(df, store=store, item=item)
        fc = forecaster.predict(horizon_days=max_h, include_history=False)
        all_forecasts.append(fc)

    combined = pd.concat(all_forecasts, ignore_index=True)
    combined_path = os.path.join(output_dir, "prophet_all_horizons.csv")
    combined.to_csv(combined_path, index=False)
    return combined
