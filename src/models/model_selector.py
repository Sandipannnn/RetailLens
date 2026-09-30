"""
Model Selector for RetailLens.
Compares LSTM (AI) and XGBoost (ML) on MAE/RMSE/MAPE and selects the
best-performing model for downstream forecasting.

Selection logic:
- Primary metric: RMSE (lower is better)
- Falls back to MAE if RMSE is tied
- Does NOT hard-code a winner — result is determined by actual evaluation

Owner: RetailLens ML Track
"""

import logging
from typing import Dict, Optional, Tuple

import pandas as pd

logger = logging.getLogger(__name__)

# Primary metric used for model selection (lower = better)
PRIMARY_METRIC: str = "RMSE"
SECONDARY_METRIC: str = "MAE"


class ModelSelector:
    """
    Compares LSTM and XGBoost forecasting models on holdout evaluation metrics
    and returns the name and fitted instance of the best model.

    Usage:
        selector = ModelSelector()
        selector.register("LSTM", lstm_metrics, lstm_forecaster)
        selector.register("XGBoost", xgb_metrics, xgb_forecaster)
        best_name, best_model = selector.select()
    """

    def __init__(
        self,
        primary_metric: str = PRIMARY_METRIC,
        secondary_metric: str = SECONDARY_METRIC,
    ):
        self.primary_metric = primary_metric.upper()
        self.secondary_metric = secondary_metric.upper()
        self._registry: Dict[str, Tuple[Dict[str, float], object]] = {}

    def register(
        self,
        model_name: str,
        metrics: Dict[str, float],
        forecaster: object,
    ) -> None:
        """
        Registers a model with its evaluation metrics.

        Args:
            model_name: Human-readable name (e.g., "LSTM", "XGBoost")
            metrics: Dictionary with at least "MAE", "RMSE", "MAPE" keys
            forecaster: Fitted model instance
        """
        self._registry[model_name] = (metrics, forecaster)
        logger.info(
            f"Registered model '{model_name}': "
            f"RMSE={metrics.get('RMSE', float('nan')):.4f}, "
            f"MAE={metrics.get('MAE', float('nan')):.4f}, "
            f"MAPE={metrics.get('MAPE', float('nan')):.2f}%"
        )

    def select(self) -> Tuple[str, object, Dict[str, float]]:
        """
        Returns (best_model_name, best_forecaster, best_metrics).
        Selection is based on the primary metric (lower = better).
        """
        if not self._registry:
            raise RuntimeError("No models registered. Call register() before select().")

        best_name = None
        best_score = float("inf")
        best_forecaster = None
        best_metrics = {}

        for name, (metrics, forecaster) in self._registry.items():
            score = metrics.get(self.primary_metric, float("inf"))
            if score < best_score:
                best_score = score
                best_name = name
                best_forecaster = forecaster
                best_metrics = metrics
            elif score == best_score:
                # Tie-break on secondary metric
                sec_score = metrics.get(self.secondary_metric, float("inf"))
                sec_best = best_metrics.get(self.secondary_metric, float("inf"))
                if sec_score < sec_best:
                    best_name = name
                    best_forecaster = forecaster
                    best_metrics = metrics

        logger.info(
            f"Selected model: '{best_name}' "
            f"({self.primary_metric}={best_metrics.get(self.primary_metric, float('nan')):.4f})"
        )
        return best_name, best_forecaster, best_metrics

    def comparison_table(self) -> pd.DataFrame:
        """
        Returns a DataFrame comparing all registered models.

        Columns: Model, MAE, RMSE, MAPE, sMAPE
        """
        rows = []
        for name, (metrics, _) in self._registry.items():
            rows.append({
                "Model": name,
                "MAE": round(metrics.get("MAE", float("nan")), 4),
                "RMSE": round(metrics.get("RMSE", float("nan")), 4),
                "MAPE (%)": round(metrics.get("MAPE", float("nan")), 2),
                "sMAPE (%)": round(metrics.get("sMAPE", float("nan")), 2),
            })
        df = pd.DataFrame(rows)
        if not df.empty and self.primary_metric in df.columns:
            df = df.sort_values(self.primary_metric).reset_index(drop=True)
        return df
