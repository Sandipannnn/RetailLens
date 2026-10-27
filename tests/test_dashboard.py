"""Focused regression tests for dashboard data and model-selection behavior."""

import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import pandas as pd

from app import streamlit_app


class TestDashboardForecastHandling(unittest.TestCase):
    def test_future_forecasts_exclude_holdout_prediction_rows(self):
        with tempfile.TemporaryDirectory(dir=Path.cwd()) as temp_dir:
            root = Path(temp_dir)
            future_dir = root / "forecasts"
            future_dir.mkdir()
            pd.DataFrame({"ds": ["2026-01-01"], "yhat": [10]}).to_csv(
                future_dir / "future.csv", index=False
            )
            pd.DataFrame({
                "date": ["2025-12-31"], "actual_sales": [8], "predicted_sales": [9]
            }).to_csv(future_dir / "holdout.csv", index=False)

            with patch.object(streamlit_app, "FORECAST_DIR", future_dir):
                forecasts = streamlit_app.load_forecasts.__wrapped__()

        self.assertEqual(len(forecasts), 1)
        self.assertEqual(forecasts.iloc[0]["_source"], "future")

    def test_holdout_loader_normalizes_actual_and_prediction_columns(self):
        with tempfile.TemporaryDirectory(dir=Path.cwd()) as temp_dir:
            prediction_dir = Path(temp_dir)
            pd.DataFrame({
                "date": ["2025-12-31"], "actual_sales": [8], "predicted_sales": [9],
                "store": [1], "item": [1],
            }).to_csv(prediction_dir / "xgboost_test_predictions.csv", index=False)

            with patch.object(streamlit_app, "ML_PREDICTIONS_DIR", prediction_dir):
                holdout = streamlit_app.load_holdout_predictions.__wrapped__()

        self.assertEqual(len(holdout), 1)
        self.assertTrue({"date", "actual_sales", "predicted_sales"}.issubset(holdout.columns))
        self.assertEqual(holdout.iloc[0]["predicted_sales"], 9)

    def test_best_model_uses_mae_then_rmse(self):
        comparison = pd.DataFrame({
            "Model": ["Lower RMSE", "Lower MAE", "Tie on MAE"],
            "MAE": [4.0, 2.0, 2.0],
            "RMSE": [4.0, 5.0, 6.0],
            "MAPE (%)": [10.0, 12.0, 8.0],
        })
        self.assertEqual(streamlit_app.best_model(comparison), "Lower MAE")

    def test_best_model_forecast_uses_the_resolved_ml_model(self):
        forecasts = pd.DataFrame({
            "_source": ["prophet_tuned_forecast", "xgboost_future_forecast"],
            "yhat": [12, 15],
        })
        selected = streamlit_app.select_forecast_model(
            forecasts, "Best Model", pd.DataFrame(), selected_best="XGBoost"
        )
        self.assertEqual(selected["_source"].tolist(), ["xgboost_future_forecast"])

    def test_interval_metrics_are_unavailable_without_bounds(self):
        with tempfile.TemporaryDirectory(dir=Path.cwd()) as temp_dir:
            prediction_dir = Path(temp_dir)
            pd.DataFrame({
                "date": ["2025-12-31"], "actual_sales": [8], "predicted_sales": [9]
            }).to_csv(prediction_dir / "xgboost_test_predictions.csv", index=False)
            comparison = pd.DataFrame({
                "Model": ["XGBoost"], "Coverage (%)": [0.0], "Avg Interval Width": [0.0]
            })
            with patch.object(streamlit_app, "ML_PREDICTIONS_DIR", prediction_dir):
                displayed = streamlit_app.comparison_for_display(comparison)

        self.assertTrue(displayed[["Coverage (%)", "Avg Interval Width"]].isna().all().all())
        self.assertEqual(comparison.loc[0, "Coverage (%)"], 0.0)


if __name__ == "__main__":
    unittest.main()
