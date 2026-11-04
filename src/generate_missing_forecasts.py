"""Resume missing per-series LSTM/XGBoost exports: python -m src.generate_missing_forecasts."""
from __future__ import annotations

import json
import os
from pathlib import Path
import subprocess
import sys
import time

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
OUTPUT = ROOT / "data/processed/forecasts"
STATUS = OUTPUT / "generation_status.json"


def write_status(state: dict, path: Path = STATUS) -> bool:
    """A temporarily locked progress file must not stop model generation."""
    temp = path.with_suffix(".tmp")
    for attempt in range(5):
        try:
            temp.write_text(json.dumps(state, indent=2))
            temp.replace(path)
            return True
        except PermissionError:
            time.sleep(0.2 * (attempt + 1))
    print("Progress file is locked; continuing training. See generation.log.", flush=True)
    return False


def valid_forecast(path: Path, last_date: pd.Timestamp) -> bool:
    if not path.is_file():
        return False
    try:
        frame = pd.read_csv(path)
        dates = pd.to_datetime(frame["ds"])
        values = pd.to_numeric(frame["yhat"])
        expected = pd.date_range(last_date + pd.Timedelta(days=1), periods=90)
        return (len(frame) == 90 and list(dates) == list(expected)
                and bool(np.isfinite(values).all()) and bool((values >= 0).all()))
    except (OSError, ValueError, KeyError):
        return False


def main() -> None:
    OUTPUT.mkdir(parents=True, exist_ok=True)
    data = pd.read_csv(ROOT / "data/processed/sales_clean.csv", parse_dates=["date"])
    series = data.groupby(["store", "item"])["date"].max()
    state = {"total": len(series), "completed": 0, "failed": [], "current": None,
             "status": "running", "started_at": time.time()}

    def save_status():
        state["updated_at"] = time.time()
        write_status(state)

    env = os.environ.copy()
    env.update({"TF_NUM_INTRAOP_THREADS": "2", "TF_NUM_INTEROP_THREADS": "2",
                "OMP_NUM_THREADS": "2", "PYTHONUNBUFFERED": "1"})
    for (store, item), last_date in series.items():
        paths = [OUTPUT / f"{model}_s{store}_i{item}.csv" for model in ("lstm", "xgboost")]
        if not all(valid_forecast(path, last_date) for path in paths):
            state["current"] = {"store": int(store), "item": int(item)}
            save_status()
            print(f"Generating store {store}, item {item}", flush=True)
            # Separate processes release TensorFlow memory between series.
            result = subprocess.run(
                [sys.executable, "-m", "src.train", "--store", str(store), "--item", str(item)],
                cwd=ROOT, env=env,
            )
            if result.returncode or not all(valid_forecast(path, last_date) for path in paths):
                state["failed"].append(state["current"])
                save_status()
                continue
        state["completed"] += 1
        save_status()
        print(f"Complete: {state['completed']}/{state['total']}", flush=True)
    state["current"] = None
    state["status"] = "complete" if not state["failed"] else "completed_with_errors"
    state["finished_at"] = time.time()
    save_status()
    if state["failed"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
