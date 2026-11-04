import numpy as np
import pandas as pd
import pytest

from src.historical_comparison import historical_comparison


def test_bounds_use_earlier_calibration_not_displayed_actuals(monkeypatch):
    dates = pd.date_range('2017-01-01', periods=365)
    data = pd.DataFrame({'date': dates, 'store': 2, 'item': 3, 'sales': 10})

    class Model:
        def evaluate_holdout(self, series, holdout_days, store, item):
            assert holdout_days == 180
            assert store == 2 and item == 3
            assert len(series) == 365
            return {}, pd.DataFrame({'date': dates[-180:],
                'actual_sales': [12] * 90 + [1000] * 90, 'predicted_sales': 10})

    monkeypatch.setattr('src.historical_comparison.RetailXGBoostForecaster', Model)
    result = historical_comparison(data, 2, 3)
    assert len(result) == 90
    assert result.date.min() == dates[-90]
    assert (result.yhat_lower == 8).all()
    assert (result.yhat_upper == 12).all()
    assert np.isfinite(result[['actual_sales', 'yhat', 'yhat_lower', 'yhat_upper']]).all().all()


def test_short_or_missing_series_is_not_fabricated():
    data = pd.DataFrame({'date': pd.date_range('2017-01-01', periods=10),
                         'store': 1, 'item': 1, 'sales': 10})
    with pytest.raises(ValueError, match='365'):
        historical_comparison(data, 2, 1)
