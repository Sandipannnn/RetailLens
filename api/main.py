"""Read-only FastAPI endpoints for the existing RetailLens sales artifacts."""
from __future__ import annotations

import json
from functools import lru_cache
from pathlib import Path
from typing import Any

import pandas as pd
from fastapi import FastAPI, HTTPException, Query

ROOT = Path(__file__).resolve().parents[1]
DATA_CANDIDATES = (
    ROOT / "data" / "processed" / "sales_clean.csv",
    ROOT / "data" / "raw" / "train.csv",
)
FORECAST_DIR = ROOT / "data" / "processed" / "forecasts"
ML_METRICS_JSON = ROOT / "outputs" / "metrics" / "model_comparison.json"
ML_METRICS_CSV = ROOT / "outputs" / "metrics" / "model_comparison.csv"
TS_METRICS_JSON = ROOT / "reports" / "ts_model_comparison.json"
TS_METRICS_CSV = ROOT / "reports" / "ts_model_comparison.csv"
FEATURE_IMPORTANCE_CSV = ROOT / "outputs" / "metrics" / "xgboost_feature_importance.csv"

app = FastAPI(
    title="RetailLens Sales Analytics API",
    description=(
        "Read-only access to RetailLens historical sales, exported future forecasts, "
        "and saved model evaluation artifacts. This API never trains models or writes to project data."
    ),
    version="1.0.0",
)


def _json_safe(value: Any) -> Any:
    """Convert pandas/numpy values into JSON-compatible Python values."""
    if isinstance(value, dict):
        return {str(key): _json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(item) for item in value]
    if pd.isna(value):
        return None
    if hasattr(value, "item"):
        return _json_safe(value.item())
    if hasattr(value, "isoformat"):
        return value.isoformat()
    return value


@lru_cache(maxsize=1)
def _sales_frame() -> tuple[pd.DataFrame, Path]:
    for path in DATA_CANDIDATES:
        if path.is_file():
            try:
                frame = pd.read_csv(path)
            except (OSError, UnicodeError, pd.errors.ParserError) as exc:
                raise HTTPException(status_code=500, detail=f"Unable to read sales data {path.name}: {exc}") from exc
            required = {"date", "store", "item", "sales"}
            missing = sorted(required.difference(frame.columns))
            if missing:
                raise HTTPException(status_code=500, detail=f"Sales data is missing required columns: {', '.join(missing)}")
            frame["date"] = pd.to_datetime(frame["date"], errors="coerce")
            frame["store"] = pd.to_numeric(frame["store"], errors="coerce")
            frame["item"] = pd.to_numeric(frame["item"], errors="coerce")
            frame["sales"] = pd.to_numeric(frame["sales"], errors="coerce")
            frame = frame.dropna(subset=["date", "store", "item", "sales"]).sort_values(["date", "store", "item"])
            return frame, path
    raise HTTPException(status_code=404, detail="No sales CSV found. Expected data/processed/sales_clean.csv or data/raw/train.csv.")


def _filtered_sales(store: int | None, item: int | None, start_date: str | None, end_date: str | None) -> pd.DataFrame:
    frame, _ = _sales_frame()
    if start_date:
        frame = frame[frame["date"] >= pd.Timestamp(start_date)]
    if end_date:
        frame = frame[frame["date"] < pd.Timestamp(end_date) + pd.Timedelta(days=1)]
    if store is not None:
        frame = frame[frame["store"] == store]
    if item is not None:
        frame = frame[frame["item"] == item]
    return frame


def _read_json(path: Path) -> Any:
    if not path.is_file():
        return None
    try:
        with path.open(encoding="utf-8") as stream:
            return json.load(stream)
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise HTTPException(status_code=500, detail=f"Unable to read {path.relative_to(ROOT)}: {exc}") from exc


def _read_csv_records(path: Path) -> list[dict[str, Any]] | None:
    if not path.is_file():
        return None
    try:
        return _json_safe(pd.read_csv(path).to_dict(orient="records"))
    except (OSError, UnicodeError, pd.errors.ParserError) as exc:
        raise HTTPException(status_code=500, detail=f"Unable to read {path.relative_to(ROOT)}: {exc}") from exc


@app.get("/health", tags=["service"])
def health() -> dict[str, str]:
    """Report that the API process is responding."""
    return {"status": "ok"}


@app.get("/sales/summary", tags=["sales"])
def sales_summary(
    store: int | None = Query(default=None, ge=1),
    item: int | None = Query(default=None, ge=1),
    start_date: str | None = None,
    end_date: str | None = None,
) -> dict[str, Any]:
    """Return sales totals, averages, date coverage, and row counts for optional filters."""
    try:
        data = _filtered_sales(store, item, start_date, end_date)
    except (ValueError, TypeError) as exc:
        raise HTTPException(status_code=422, detail=f"Invalid date filter: {exc}") from exc
    return {
        "filters": {"store": store, "item": item, "start_date": start_date, "end_date": end_date},
        "records": int(len(data)),
        "total_sales": float(data["sales"].sum()) if not data.empty else 0.0,
        "average_daily_sales_per_record": float(data["sales"].mean()) if not data.empty else None,
        "minimum_sales": float(data["sales"].min()) if not data.empty else None,
        "maximum_sales": float(data["sales"].max()) if not data.empty else None,
        "stores": int(data["store"].nunique()),
        "items": int(data["item"].nunique()),
        "date_start": data["date"].min().date().isoformat() if not data.empty else None,
        "date_end": data["date"].max().date().isoformat() if not data.empty else None,
    }


@app.get("/sales", tags=["sales"])
def historical_sales(
    store: int | None = Query(default=None, ge=1),
    item: int | None = Query(default=None, ge=1),
    start_date: str | None = None,
    end_date: str | None = None,
    limit: int = Query(default=1000, ge=1, le=10000),
    offset: int = Query(default=0, ge=0),
) -> dict[str, Any]:
    """Return paginated historical observations with optional store/item/date filters."""
    try:
        data = _filtered_sales(store, item, start_date, end_date)
    except (ValueError, TypeError) as exc:
        raise HTTPException(status_code=422, detail=f"Invalid date filter: {exc}") from exc
    total = len(data)
    records = data.iloc[offset : offset + limit].copy()
    records["date"] = records["date"].dt.strftime("%Y-%m-%d")
    return {"total": int(total), "limit": limit, "offset": offset, "records": records.to_dict(orient="records")}


@app.get("/forecasts", tags=["forecasts"])
def forecasts(
    store: int | None = Query(default=None, ge=1),
    item: int | None = Query(default=None, ge=1),
    limit: int = Query(default=1000, ge=1, le=10000),
    offset: int = Query(default=0, ge=0),
) -> dict[str, Any]:
    """Return saved future forecast rows; holdout prediction files are deliberately excluded."""
    files = sorted(FORECAST_DIR.glob("*.csv")) if FORECAST_DIR.is_dir() else []
    parts: list[pd.DataFrame] = []
    for path in files:
        try:
            frame = pd.read_csv(path)
        except (OSError, UnicodeError, pd.errors.ParserError) as exc:
            raise HTTPException(status_code=500, detail=f"Unable to read forecast {path.name}: {exc}") from exc
        if frame.empty or {"actual", "actual_sales", "actuals"}.intersection(str(col).lower() for col in frame.columns):
            continue
        frame["_source"] = path.stem
        parts.append(frame)
    if not parts:
        return {"available": False, "message": "No exported future forecast CSV files are available.", "total": 0, "limit": limit, "offset": offset, "records": []}
    combined = pd.concat(parts, ignore_index=True)
    store_col = next((c for c in combined.columns if str(c).lower() in {"store", "store_id", "location", "branch"}), None)
    item_col = next((c for c in combined.columns if str(c).lower() in {"item", "product", "product_id", "sku"}), None)
    if store is not None and store_col:
        combined = combined[pd.to_numeric(combined[store_col], errors="coerce") == store]
    if item is not None and item_col:
        combined = combined[pd.to_numeric(combined[item_col], errors="coerce") == item]
    total = len(combined)
    return {"available": True, "total": int(total), "limit": limit, "offset": offset, "records": _json_safe(combined.iloc[offset : offset + limit].to_dict(orient="records"))}


@app.get("/models/comparison", tags=["models"])
def model_comparison() -> dict[str, Any]:
    """Return the saved time-series comparison report and ML model comparison separately."""
    return {
        "time_series": {"json": _read_json(TS_METRICS_JSON), "csv": _read_csv_records(TS_METRICS_CSV)},
        "machine_learning": {"json": _read_json(ML_METRICS_JSON), "csv": _read_csv_records(ML_METRICS_CSV)},
    }


@app.get("/models/evaluation", tags=["models"])
def model_evaluation() -> dict[str, Any]:
    """Return saved ML evaluation metrics, prediction-file summaries, and feature importance."""
    metrics = _read_json(ML_METRICS_JSON)
    prediction_summaries: dict[str, Any] = {}
    prediction_dir = ROOT / "outputs" / "predictions"
    for name in ("prophet_test_predictions.csv", "xgboost_test_predictions.csv"):
        path = prediction_dir / name
        rows = _read_csv_records(path)
        prediction_summaries[name] = {"available": rows is not None, "records": len(rows) if rows is not None else 0}
    return {
        "metrics": metrics,
        "feature_importance": _read_csv_records(FEATURE_IMPORTANCE_CSV),
        "holdout_predictions": prediction_summaries,
        "note": "Prediction artifacts are historical holdout evaluations and are not future forecasts.",
    }
