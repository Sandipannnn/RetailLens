"""
Comprehensive Unit Test Suite for Time-Series Forecasting Modules.
Tests Prophet, SARIMA, Multi-Step Horizons, Confidence Intervals, Tuning, and Metrics.
"""

import unittest
import numpy as np
import pandas as pd

from src.data_generator import generate_sample_sales_data
from src.utils.metrics import (
    mean_absolute_error,
    root_mean_squared_error,
    mean_absolute_percentage_error,
    coverage_probability,
    calculate_metrics,
)
from src.models.prophet_model import RetailProphetForecaster
from src.models.sarima_model import RetailSARIMAForecaster
from src.models.hyperparameter_tuner import ProphetTuner


class TestTimeSeriesMetrics(unittest.TestCase):
    def setUp(self):
        self.y_true = np.array([10.0, 20.0, 30.0, 40.0])
        self.y_pred = np.array([12.0, 18.0, 33.0, 39.0])
        self.y_lower = np.array([8.0, 15.0, 25.0, 35.0])
        self.y_upper = np.array([15.0, 25.0, 35.0, 45.0])

    def test_mae_and_rmse(self):
        mae = mean_absolute_error(self.y_true, self.y_pred)
        rmse = root_mean_squared_error(self.y_true, self.y_pred)
        # diffs: [2, 2, 3, 1], mean = 2.0
        self.assertAlmostEqual(mae, 2.0, places=4)
        # sq diffs: [4, 4, 9, 1] -> mean=4.5 -> sqrt(4.5) ~ 2.1213
        self.assertAlmostEqual(rmse, np.sqrt(4.5), places=4)

    def test_coverage(self):
        cov = coverage_probability(self.y_true, self.y_lower, self.y_upper)
        self.assertEqual(cov, 100.0)

        # One value out of bound
        y_true_out = np.array([5.0, 20.0, 30.0, 40.0])
        cov_out = coverage_probability(y_true_out, self.y_lower, self.y_upper)
        self.assertEqual(cov_out, 75.0)

    def test_calculate_metrics_dict(self):
        metrics = calculate_metrics(self.y_true, self.y_pred, self.y_lower, self.y_upper)
        self.assertIn("MAE", metrics)
        self.assertIn("RMSE", metrics)
        self.assertIn("MAPE", metrics)
        self.assertIn("Coverage", metrics)
        self.assertIn("Avg_Interval_Width", metrics)


class TestDataGenerator(unittest.TestCase):
    def test_generate_sample_sales(self):
        df = generate_sample_sales_data(
            num_stores=2,
            num_items=2,
            start_date="2016-01-01",
            end_date="2016-03-31",
            output_path=None,
        )
        self.assertEqual(set(df.columns), {"date", "store", "item", "sales"})
        self.assertEqual(df["store"].nunique(), 2)
        self.assertEqual(df["item"].nunique(), 2)
        self.assertTrue((df["sales"] >= 0).all())


class TestForecastingModels(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        # 1.5 years of synthetic daily data for fast test execution
        cls.df = generate_sample_sales_data(
            num_stores=1,
            num_items=1,
            start_date="2015-01-01",
            end_date="2016-06-30",
            output_path=None,
        )

    def test_prophet_multistep_and_intervals(self):
        forecaster = RetailProphetForecaster(
            seasonality_mode="additive",
            changepoint_prior_scale=0.05,
            interval_width=0.90,
            yearly_seasonality=True,
            weekly_seasonality=True,
        )
        forecaster.fit(self.df, store=1, item=1)
        self.assertTrue(forecaster.is_fitted)

        # Multi-step horizons: 30, 60, 90 days
        horizons = forecaster.forecast_horizons([30, 60, 90])
        self.assertEqual(len(horizons[30]), 30)
        self.assertEqual(len(horizons[60]), 60)
        self.assertEqual(len(horizons[90]), 90)

        fc_90 = horizons[90]
        self.assertIn("yhat", fc_90.columns)
        self.assertIn("yhat_lower", fc_90.columns)
        self.assertIn("yhat_upper", fc_90.columns)

        # Assert interval logic: lower <= yhat <= upper (or with small numerical tolerance)
        self.assertTrue((fc_90["yhat_lower"] <= fc_90["yhat"] + 1e-3).all())
        self.assertTrue((fc_90["yhat"] <= fc_90["yhat_upper"] + 1e-3).all())

        # Extract components
        comps = forecaster.extract_components(fc_90)
        self.assertIn("trend", comps)
        self.assertIn("weekly", comps)

    def test_prophet_holdout_evaluation(self):
        forecaster = RetailProphetForecaster(interval_width=0.95)
        metrics, comparison = forecaster.evaluate_holdout(self.df, holdout_days=30, store=1, item=1)
        self.assertEqual(len(comparison), 30)
        self.assertIn("MAE", metrics)
        self.assertIn("RMSE", metrics)
        self.assertGreater(metrics["MAE"], 0)

    def test_sarima_forecasting_and_intervals(self):
        sarima = RetailSARIMAForecaster(
            order=(1, 1, 0),
            seasonal_order=(1, 0, 0, 7),
            interval_width=0.90,
        )
        sarima.fit(self.df, store=1, item=1)
        self.assertTrue(sarima.is_fitted)

        horizons = sarima.forecast_horizons([30, 60, 90])
        self.assertEqual(len(horizons[30]), 30)
        self.assertEqual(len(horizons[60]), 60)
        self.assertEqual(len(horizons[90]), 90)

        fc = horizons[30]
        self.assertTrue((fc["yhat_lower"] <= fc["yhat"] + 1e-3).all())
        self.assertTrue((fc["yhat"] <= fc["yhat_upper"] + 1e-3).all())

    def test_prophet_tuning(self):
        param_grid = {
            "changepoint_prior_scale": [0.01, 0.1],
            "seasonality_mode": ["additive", "multiplicative"],
        }
        tuner = ProphetTuner(param_grid=param_grid, metric="mae")
        best_params, results_df = tuner.grid_search_cv(self.df, store=1, item=1, use_holdout_fallback=True)
        self.assertIn("changepoint_prior_scale", best_params)
        self.assertEqual(len(results_df), 4)

        best_forecaster = tuner.get_best_forecaster()
        self.assertIsInstance(best_forecaster, RetailProphetForecaster)


if __name__ == "__main__":
    unittest.main()
