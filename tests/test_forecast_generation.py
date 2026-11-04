import numpy as np
import pandas as pd

from src.generate_missing_forecasts import valid_forecast, write_status


def test_resume_only_accepts_complete_finite_future_forecasts(tmp_path):
    path = tmp_path / "forecast.csv"
    cutoff = pd.Timestamp("2017-12-31")
    frame = pd.DataFrame({"ds": pd.date_range("2018-01-01", periods=90), "yhat": 12.0})
    assert not valid_forecast(path, cutoff)

    frame.to_csv(path, index=False)
    assert valid_forecast(path, cutoff)
    frame.iloc[:30].to_csv(path, index=False)
    assert not valid_forecast(path, cutoff)
    frame.loc[0, "yhat"] = np.inf
    frame.to_csv(path, index=False)
    assert not valid_forecast(path, cutoff)
    frame["yhat"] = 12.0
    frame["ds"] -= pd.Timedelta(days=90)
    frame.to_csv(path, index=False)
    assert not valid_forecast(path, cutoff)


def test_progress_lock_does_not_abort_training(tmp_path, monkeypatch):
    from pathlib import Path
    monkeypatch.setattr("src.generate_missing_forecasts.time.sleep", lambda _: None)
    def locked(*args):
        raise PermissionError("temporarily locked")
    monkeypatch.setattr(Path, "replace", locked)
    assert write_status({"completed": 4}, tmp_path / "status.json") is False

