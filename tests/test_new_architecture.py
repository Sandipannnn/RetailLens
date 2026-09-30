"""
RetailLens — Comprehensive Test Suite.
Tests the new LSTM + XGBoost architecture.

Run from project root:
    python -m pytest tests/ -v
    python -m unittest tests/test_new_architecture.py -v
"""

import os
import sys
import tempfile
import unittest

import numpy as np
import pandas as pd

# Ensure project root is in path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.data_generator import generate_sample_sales_data
from src.utils.metrics import (
    mean_absolute_error,
    root_mean_squared_error,
    mean_absolute_percentage_error,
    calculate_metrics,
)
from src.features.feature_engineering import (
    add_calendar_features,
    add_lag_features,
    add_rolling_features,
    build_features,
    get_feature_columns,
    LAG_DAYS,
    ROLLING_WINDOWS,
)
from src.features.lstm_features import build_lstm_sequences
from src.models.xgboost_model import RetailXGBoostForecaster
from src.models.model_selector import ModelSelector
from src.inventory import compute_inventory_recommendation, InventoryReport


# ---------------------------------------------------------------------------
# Fixtures — shared synthetic dataset
# ---------------------------------------------------------------------------

def make_df(n_stores=1, n_items=1, years=2):
    """Creates a small synthetic dataset for fast testing."""
    end = f"{2014 + years}-12-31"
    return generate_sample_sales_data(
        num_stores=n_stores,
        num_items=n_items,
        start_date="2013-01-01",
        end_date=end,
        output_path=None,
    )


# ---------------------------------------------------------------------------
# 1. Metrics
# ---------------------------------------------------------------------------

class TestMetrics(unittest.TestCase):
    def setUp(self):
        self.y_true = np.array([10.0, 20.0, 30.0, 40.0])
        self.y_pred = np.array([12.0, 18.0, 33.0, 39.0])

    def test_mae(self):
        mae = mean_absolute_error(self.y_true, self.y_pred)
        # diffs [2,2,3,1] -> mean 2.0
        self.assertAlmostEqual(mae, 2.0, places=4)

    def test_rmse(self):
        rmse = root_mean_squared_error(self.y_true, self.y_pred)
        expected = np.sqrt(np.mean([4, 4, 9, 1]))
        self.assertAlmostEqual(rmse, expected, places=4)

    def test_mape_positive(self):
        mape = mean_absolute_percentage_error(self.y_true, self.y_pred)
        self.assertGreater(mape, 0)
        self.assertLess(mape, 50)

    def test_calculate_metrics_keys(self):
        m = calculate_metrics(self.y_true, self.y_pred)
        for key in ("MAE", "RMSE", "MAPE", "sMAPE"):
            self.assertIn(key, m)
        self.assertGreater(m["MAE"], 0)


# ---------------------------------------------------------------------------
# 2. Data Generator
# ---------------------------------------------------------------------------

class TestDataGenerator(unittest.TestCase):
    def test_schema(self):
        df = make_df(n_stores=2, n_items=3)
        self.assertEqual(set(df.columns), {"date", "store", "item", "sales"})
        self.assertEqual(df["store"].nunique(), 2)
        self.assertEqual(df["item"].nunique(), 3)
        self.assertTrue((df["sales"] >= 0).all())

    def test_no_nulls(self):
        df = make_df()
        self.assertEqual(df.isnull().sum().sum(), 0)


# ---------------------------------------------------------------------------
# 3. Feature Engineering
# ---------------------------------------------------------------------------

class TestFeatureEngineering(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.df = make_df()

    def test_calendar_features(self):
        df = add_calendar_features(self.df)
        for col in ["day_of_week", "month", "year", "is_weekend"]:
            self.assertIn(col, df.columns)

    def test_lag_features(self):
        df = add_lag_features(self.df)
        self.assertIn("lag_1", df.columns)
        self.assertIn("lag_7", df.columns)
        self.assertIn("lag_28", df.columns)

    def test_rolling_features(self):
        df = add_rolling_features(self.df)
        self.assertIn("rolling_mean_7", df.columns)
        self.assertIn("rolling_std_14", df.columns)

    def test_build_features_no_na_after_drop(self):
        df = build_features(self.df)
        feat_cols = [c for c in get_feature_columns() if c in df.columns]
        df_clean = df.dropna(subset=feat_cols)
        self.assertGreater(len(df_clean), 0)

    def test_lag_is_shifted_by_one(self):
        """lag_1 at row i must equal sales at row i-1 — no leakage."""
        df = add_lag_features(self.df)
        df = df.sort_values(["store", "item", "date"]).reset_index(drop=True)
        series = df[(df["store"] == 1) & (df["item"] == 1)].copy()
        # First valid lag_1 row (index 1 in the series)
        row_idx = series.index[1]
        prev_idx = series.index[0]
        self.assertAlmostEqual(
            float(series.loc[row_idx, "lag_1"]),
            float(series.loc[prev_idx, "sales"]),
            places=4,
        )

    def test_get_feature_columns(self):
        cols = get_feature_columns()
        self.assertIn("lag_1", cols)
        self.assertIn("rolling_mean_7", cols)
        self.assertIn("day_of_week", cols)

    def test_lstm_sequences_shape(self):
        series = np.arange(100, dtype=np.float32)
        window = 14
        X, y = build_lstm_sequences(series, window)
        self.assertEqual(X.shape, (100 - window, window, 1))
        self.assertEqual(y.shape, (100 - window,))
        # No leakage: y[i] = series[window + i]
        np.testing.assert_allclose(y[0], series[window])
        np.testing.assert_allclose(y[-1], series[-1])


# ---------------------------------------------------------------------------
# 4. XGBoost Model
# ---------------------------------------------------------------------------

class TestXGBoostForecaster(unittest.TestCase):
    """Tests for the upstream RetailXGBoostForecaster (ML track)."""
    @classmethod
    def setUpClass(cls):
        cls.df = make_df(years=3)
        cls.xgb = RetailXGBoostForecaster()
        cls.xgb.fit(cls.df, store=1, item=1)

    def test_fit_sets_is_fitted(self):
        self.assertTrue(self.xgb.is_fitted)
        self.assertIsNotNone(self.xgb.model)

    def test_predict_future_returns_dataframe(self):
        fc = self.xgb.predict_future(self.df, horizon_days=30, store=1, item=1)
        self.assertIsInstance(fc, pd.DataFrame)
        self.assertEqual(len(fc), 30)
        self.assertIn("yhat", fc.columns)
        self.assertIn("ds", fc.columns)

    def test_predict_future_non_negative(self):
        fc = self.xgb.predict_future(self.df, horizon_days=30, store=1, item=1)
        self.assertTrue((fc["yhat"] >= 0).all())

    def test_evaluate_holdout_metrics(self):
        metrics, comparison = self.xgb.evaluate_holdout(
            self.df, holdout_days=30, store=1, item=1
        )
        self.assertIn("MAE", metrics)
        self.assertIn("RMSE", metrics)
        self.assertGreater(metrics["MAE"], 0)
        self.assertEqual(len(comparison), 30)
        self.assertIn("actual_sales", comparison.columns)
        self.assertIn("predicted_sales", comparison.columns)

    def test_feature_importance(self):
        fi = self.xgb.get_feature_importance()
        self.assertIsInstance(fi, pd.DataFrame)
        self.assertIn("feature", fi.columns)
        self.assertIn("importance", fi.columns)
        self.assertGreater(len(fi), 0)

    def test_save_and_load(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            path = os.path.join(tmpdir, "xgb_test.joblib")
            xgb = RetailXGBoostForecaster()
            xgb.fit(self.df, store=1, item=1)
            xgb.save(path)
            self.assertTrue(os.path.exists(path))

            xgb2 = RetailXGBoostForecaster.load(path)
            self.assertTrue(xgb2.is_fitted)
            fc = xgb2.predict_future(self.df, horizon_days=10, store=1, item=1)
            self.assertEqual(len(fc), 10)


# ---------------------------------------------------------------------------
# 5. LSTM Model (lightweight test using minimal epochs)
# ---------------------------------------------------------------------------

class TestLSTMForecaster(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        try:
            from src.models.lstm_model import LSTMForecaster
            cls.LSTMForecaster = LSTMForecaster
            cls.df = make_df(years=2)
            # Use minimal settings for fast testing
            cls.lstm = LSTMForecaster(
                window=7,
                lstm_units_1=8,
                lstm_units_2=4,
                epochs=2,
                batch_size=32,
                patience=1,
            )
            cls.lstm.fit(cls.df, store=1, item=1, val_days=30)
            cls.available = True
        except ImportError:
            cls.available = False

    def _skip_if_unavailable(self):
        if not self.available:
            self.skipTest("TensorFlow not available")

    def test_fit_sets_is_fitted(self):
        self._skip_if_unavailable()
        self.assertTrue(self.lstm.is_fitted)

    def test_predict_returns_dataframe(self):
        self._skip_if_unavailable()
        fc = self.lstm.predict(horizon_days=30)
        self.assertIsInstance(fc, pd.DataFrame)
        self.assertEqual(len(fc), 30)
        self.assertIn("yhat", fc.columns)
        self.assertIn("ds", fc.columns)

    def test_predict_non_negative(self):
        self._skip_if_unavailable()
        fc = self.lstm.predict(horizon_days=30, clip_non_negative=True)
        self.assertTrue((fc["yhat"] >= 0).all())

    def test_forecast_horizons(self):
        self._skip_if_unavailable()
        horizons = self.lstm.forecast_horizons([30, 60, 90])
        self.assertEqual(len(horizons[30]), 30)
        self.assertEqual(len(horizons[60]), 60)
        self.assertEqual(len(horizons[90]), 90)

    def test_holdout_metrics(self):
        self._skip_if_unavailable()
        lstm = self.LSTMForecaster(
            window=7, lstm_units_1=8, lstm_units_2=4, epochs=2, patience=1
        )
        metrics, comparison = lstm.evaluate_holdout(
            self.df, holdout_days=30, store=1, item=1
        )
        self.assertIn("MAE", metrics)
        self.assertIn("RMSE", metrics)
        self.assertGreater(metrics["MAE"], 0)

    def test_save_and_load(self):
        self._skip_if_unavailable()
        with tempfile.TemporaryDirectory() as tmpdir:
            lstm = self.LSTMForecaster(
                window=7, lstm_units_1=8, lstm_units_2=4,
                epochs=2, patience=1, model_dir=tmpdir
            )
            lstm.fit(self.df, store=1, item=1, val_days=30)
            lstm.save(1, 1)

            self.assertTrue(self.LSTMForecaster.model_exists(1, 1, tmpdir))

            lstm2 = self.LSTMForecaster(model_dir=tmpdir)
            lstm2.load(1, 1)
            self.assertTrue(lstm2.is_fitted)


# ---------------------------------------------------------------------------
# 6. Model Selector
# ---------------------------------------------------------------------------

class TestModelSelector(unittest.TestCase):
    def _make_dummy_model(self):
        return object()  # Placeholder

    def test_select_lower_rmse_wins(self):
        selector = ModelSelector()
        selector.register("ModelA", {"MAE": 5.0, "RMSE": 10.0, "MAPE": 5.0}, "A")
        selector.register("ModelB", {"MAE": 4.0, "RMSE": 8.0, "MAPE": 4.0}, "B")
        best_name, best_model, best_metrics = selector.select()
        self.assertEqual(best_name, "ModelB")
        self.assertEqual(best_model, "B")

    def test_select_tie_broken_by_mae(self):
        selector = ModelSelector()
        selector.register("ModelA", {"MAE": 5.0, "RMSE": 8.0, "MAPE": 5.0}, "A")
        selector.register("ModelB", {"MAE": 4.0, "RMSE": 8.0, "MAPE": 4.0}, "B")
        best_name, _, _ = selector.select()
        self.assertEqual(best_name, "ModelB")

    def test_comparison_table(self):
        selector = ModelSelector()
        selector.register("LSTM (AI)", {"MAE": 5.0, "RMSE": 9.0, "MAPE": 5.0, "sMAPE": 5.0}, "lstm")
        selector.register("XGBoost (ML)", {"MAE": 4.0, "RMSE": 7.0, "MAPE": 4.0, "sMAPE": 4.0}, "xgb")
        df = selector.comparison_table()
        self.assertEqual(len(df), 2)
        self.assertIn("Model", df.columns)
        self.assertIn("RMSE", df.columns)
        # Should be sorted ascending by RMSE
        self.assertEqual(df.iloc[0]["Model"], "XGBoost (ML)")

    def test_no_models_raises(self):
        selector = ModelSelector()
        with self.assertRaises(RuntimeError):
            selector.select()


# ---------------------------------------------------------------------------
# 7. Inventory Recommendation
# ---------------------------------------------------------------------------

class TestInventoryRecommendation(unittest.TestCase):
    def _make_forecast_df(self, n=90, daily_demand=50.0):
        dates = pd.date_range("2018-01-01", periods=n, freq="D")
        return pd.DataFrame({"ds": dates, "yhat": [daily_demand] * n})

    def test_basic_calculation(self):
        fc = self._make_forecast_df(n=90, daily_demand=50.0)
        report = compute_inventory_recommendation(
            forecast_df=fc,
            current_stock=200.0,
            store=1,
            item=1,
            model_name="TestModel",
            lead_time_days=7,
            cycle_stock_days=14,
        )
        self.assertIsInstance(report, InventoryReport)
        # Lead-time demand = 7 * 50 = 350
        self.assertAlmostEqual(report.expected_lead_time_demand, 350.0, places=1)
        # Safety stock >= 0 (constant demand => std=0 => safety_stock=0, which is correct)
        self.assertGreaterEqual(report.safety_stock, 0)
        # Reorder point = lead_time_demand + safety_stock > 0
        self.assertGreater(report.reorder_point, 0)

    def test_low_stock_alert_triggered(self):
        fc = self._make_forecast_df(n=90, daily_demand=50.0)
        report = compute_inventory_recommendation(
            forecast_df=fc,
            current_stock=10.0,   # Very low
            store=1, item=1,
            lead_time_days=7,
            cycle_stock_days=14,
        )
        self.assertTrue(report.low_stock_alert)

    def test_no_alert_when_stock_adequate(self):
        fc = self._make_forecast_df(n=90, daily_demand=10.0)
        report = compute_inventory_recommendation(
            forecast_df=fc,
            current_stock=5000.0,  # Very high
            store=1, item=1,
            lead_time_days=7,
            cycle_stock_days=14,
        )
        self.assertFalse(report.low_stock_alert)

    def test_overstock_alert_triggered(self):
        fc = self._make_forecast_df(n=90, daily_demand=5.0)
        report = compute_inventory_recommendation(
            forecast_df=fc,
            current_stock=10000.0,  # Way too much
            store=1, item=1,
            lead_time_days=7,
            cycle_stock_days=14,
        )
        self.assertTrue(report.overstock_alert)

    def test_to_dict_keys(self):
        fc = self._make_forecast_df()
        report = compute_inventory_recommendation(
            forecast_df=fc, current_stock=300.0, store=1, item=1
        )
        d = report.to_dict()
        for key in ["Reorder Point", "Safety Stock", "Suggested Reorder Qty",
                    "Low-Stock Alert", "Overstock Alert"]:
            self.assertIn(key, d)

    def test_non_negative_reorder_qty(self):
        fc = self._make_forecast_df(n=90, daily_demand=50.0)
        report = compute_inventory_recommendation(
            forecast_df=fc, current_stock=50000.0, store=1, item=1
        )
        self.assertGreaterEqual(report.suggested_reorder_qty, 0)

    def test_missing_yhat_raises(self):
        bad_df = pd.DataFrame({"ds": pd.date_range("2018-01-01", periods=10), "val": [1] * 10})
        with self.assertRaises(ValueError):
            compute_inventory_recommendation(bad_df, 100.0, 1, 1)

    def test_empty_forecast_raises(self):
        empty_df = pd.DataFrame({"yhat": []})
        with self.assertRaises(ValueError):
            compute_inventory_recommendation(empty_df, 100.0, 1, 1)


# ---------------------------------------------------------------------------
# Run
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    unittest.main(verbosity=2)
