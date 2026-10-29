"""Regression tests for the FastAPI sales, forecast, and inventory endpoints."""

import asyncio
import json
from pathlib import Path

import httpx
import pandas as pd
import pytest

from api import main as api_module


class APIClient:
    def get(self, path: str, params: dict[str, int | str] | None = None) -> httpx.Response:
        async def send_request() -> httpx.Response:
            transport = httpx.ASGITransport(app=api_module.app)
            async with httpx.AsyncClient(transport=transport, base_url="http://testserver") as client:
                return await client.get(path, params=params)

        return asyncio.run(send_request())


@pytest.fixture
def client() -> APIClient:
    api_module._sales_frame.cache_clear()
    test_client = APIClient()
    yield test_client
    api_module._sales_frame.cache_clear()


def _write_sales_csv(path: Path) -> None:
    pd.DataFrame({
        "date": ["2024-01-01", "2024-01-02", "2024-01-01"],
        "store": [1, 1, 2],
        "item": [1, 1, 1],
        "sales": [10, 12, 8],
    }).to_csv(path, index=False)


def test_health_endpoint(client):
    response = client.get("/health")

    assert response.status_code == 200
    assert response.json() == {"status": "ok"}


def test_sales_options_returns_store_and_item_choices(client, tmp_path, monkeypatch):
    sales_path = tmp_path / "sales.csv"
    _write_sales_csv(sales_path)
    monkeypatch.setattr(api_module, "DATA_CANDIDATES", (sales_path,))

    response = client.get("/sales/options", params={"store": 1})

    assert response.status_code == 200
    assert response.json() == {
        "stores": [1, 2],
        "items": [1],
        "date_start": "2024-01-01",
        "date_end": "2024-01-02",
    }


def test_sales_summary_and_pagination_use_filters(client, tmp_path, monkeypatch):
    sales_path = tmp_path / "sales.csv"
    _write_sales_csv(sales_path)
    monkeypatch.setattr(api_module, "DATA_CANDIDATES", (sales_path,))

    summary = client.get("/sales/summary", params={"store": 1, "item": 1})
    page = client.get("/sales", params={
        "store": 1,
        "item": 1,
        "limit": 1,
        "offset": 1,
    })

    assert summary.status_code == 200
    assert summary.json()["records"] == 2
    assert summary.json()["total_sales"] == 22
    assert page.status_code == 200
    assert page.json()["total"] == 2
    assert page.json()["records"] == [{
        "date": "2024-01-02",
        "store": 1,
        "item": 1,
        "sales": 12,
    }]


def test_sales_reject_invalid_or_reversed_date_ranges(client, tmp_path, monkeypatch):
    sales_path = tmp_path / "sales.csv"
    _write_sales_csv(sales_path)
    monkeypatch.setattr(api_module, "DATA_CANDIDATES", (sales_path,))

    malformed = client.get("/sales", params={"start_date": "not-a-date"})
    reversed_range = client.get(
        "/sales/summary",
        params={"start_date": "2024-01-02", "end_date": "2024-01-01"},
    )

    assert malformed.status_code == 422
    assert reversed_range.status_code == 422


def test_forecasts_exclude_holdout_and_filter_by_series(client, tmp_path, monkeypatch):
    forecast_dir = tmp_path / "forecasts"
    forecast_dir.mkdir()
    pd.DataFrame({"ds": ["2024-02-01"], "yhat": [11]}).to_csv(
        forecast_dir / "lstm_s1_i1.csv", index=False
    )
    pd.DataFrame({"ds": ["2024-02-01"], "yhat": [20]}).to_csv(
        forecast_dir / "xgboost_s2_i1.csv", index=False
    )
    pd.DataFrame({
        "date": ["2024-01-31"],
        "actual_sales": [10],
        "predicted_sales": [12],
    }).to_csv(forecast_dir / "holdout_s1_i1.csv", index=False)
    monkeypatch.setattr(api_module, "FORECAST_DIR", forecast_dir)

    response = client.get("/forecasts", params={"store": 1, "item": 1})

    assert response.status_code == 200
    payload = response.json()
    assert payload["total"] == 1
    assert payload["records"][0]["store"] == 1
    assert payload["records"][0]["item"] == 1
    assert payload["records"][0]["yhat"] == 11


def test_model_comparison_includes_current_per_series_report(client, tmp_path, monkeypatch):
    report_dir = tmp_path / "reports"
    report_dir.mkdir()
    report = {
        "store": 1,
        "item": 1,
        "best_model": "LSTM (AI)",
        "comparison": [{"Model": "LSTM (AI)", "RMSE": 5.5}],
    }
    (report_dir / "model_comparison_s1_i1.json").write_text(json.dumps(report))
    (report_dir / "model_comparison_s2_i1.json").write_text(json.dumps({
        "store": 2, "item": 1, "best_model": "XGBoost (ML)", "comparison": [],
    }))
    monkeypatch.setattr(api_module, "REPORT_DIR", report_dir)

    response = client.get("/models/comparison", params={"store": 1, "item": 1})

    assert response.status_code == 200
    reports = response.json()["training_reports"]
    assert len(reports) == 1
    assert reports[0]["store"] == 1
    assert reports[0]["report"]["best_model"] == "LSTM (AI)"


def test_model_evaluation_includes_current_report_and_saved_artifacts(client, tmp_path, monkeypatch):
    report_dir = tmp_path / "reports"
    prediction_dir = tmp_path / "predictions"
    report_dir.mkdir()
    prediction_dir.mkdir()
    (report_dir / "model_comparison_s1_i1.json").write_text(json.dumps({
        "store": 1,
        "item": 1,
        "best_model": "LSTM (AI)",
    }))
    metrics_path = tmp_path / "metrics.json"
    metrics_path.write_text(json.dumps({"best_model": "XGBoost"}))
    importance_path = tmp_path / "importance.csv"
    pd.DataFrame({"feature": ["lag_7"], "importance": [1.0]}).to_csv(
        importance_path, index=False
    )
    pd.DataFrame({"actual": [10], "prediction": [11]}).to_csv(
        prediction_dir / "xgboost_test_predictions.csv", index=False
    )
    monkeypatch.setattr(api_module, "REPORT_DIR", report_dir)
    monkeypatch.setattr(api_module, "ML_METRICS_JSON", metrics_path)
    monkeypatch.setattr(api_module, "FEATURE_IMPORTANCE_CSV", importance_path)
    monkeypatch.setattr(api_module, "PREDICTION_DIR", prediction_dir)

    response = client.get("/models/evaluation", params={"store": 1, "item": 1})

    assert response.status_code == 200
    payload = response.json()
    assert payload["training_reports"][0]["report"]["best_model"] == "LSTM (AI)"
    assert payload["metrics"] == {"best_model": "XGBoost"}
    assert payload["feature_importance"][0]["feature"] == "lag_7"
    assert payload["holdout_predictions"]["xgboost_test_predictions.csv"] == {
        "available": True,
        "records": 1,
    }


def test_inventory_recommendation_uses_selected_best_forecast(client, tmp_path, monkeypatch):
    report_dir = tmp_path / "reports"
    forecast_dir = tmp_path / "forecasts"
    report_dir.mkdir()
    forecast_dir.mkdir()
    (report_dir / "model_comparison_s1_i1.json").write_text(json.dumps({
        "store": 1, "item": 1, "best_model": "LSTM (AI)",
    }))
    pd.DataFrame({
        "ds": pd.date_range("2024-02-01", periods=30, freq="D"),
        "yhat": [10.0] * 30,
    }).to_csv(forecast_dir / "lstm_s1_i1.csv", index=False)
    pd.DataFrame({
        "ds": pd.date_range("2024-02-01", periods=30, freq="D"),
        "yhat": [100.0] * 30,
    }).to_csv(forecast_dir / "xgboost_s1_i1.csv", index=False)
    monkeypatch.setattr(api_module, "REPORT_DIR", report_dir)
    monkeypatch.setattr(api_module, "FORECAST_DIR", forecast_dir)

    response = client.get("/inventory/recommendation", params={
        "store": 1,
        "item": 1,
        "current_stock": 5,
    })

    assert response.status_code == 200
    payload = response.json()
    assert payload["forecast_model"] == "lstm"
    assert payload["recommendation"]["Lead Time Demand"] == 70
    assert payload["recommendation"]["Low-Stock Alert"] is True


def test_inventory_endpoint_reports_missing_training_report(client, tmp_path, monkeypatch):
    report_dir = tmp_path / "reports"
    report_dir.mkdir()
    monkeypatch.setattr(api_module, "REPORT_DIR", report_dir)

    response = client.get("/inventory/recommendation", params={
        "store": 1,
        "item": 1,
        "current_stock": 20,
    })

    assert response.status_code == 404
    assert "No trained model report" in response.json()["detail"]
