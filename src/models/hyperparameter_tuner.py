"""
Automated Hyperparameter Tuning for Prophet Forecaster.
Supports Grid Search with Rolling-Origin Cross-Validation and Optuna Bayesian Optimization.

Owner: Rudra Pratap Singh (Time-Series Forecasting Track)
"""

import itertools
import logging
from typing import Any, Dict, List, Optional, Tuple, Union
import numpy as np
import pandas as pd
from prophet import Prophet
from prophet.diagnostics import cross_validation, performance_metrics

from src.models.prophet_model import RetailProphetForecaster
from src.utils.metrics import calculate_metrics

logging.getLogger("prophet").setLevel(logging.WARNING)
logging.getLogger("cmdstanpy").setLevel(logging.WARNING)


class ProphetTuner:
    """
    Hyperparameter optimization engine for RetailProphetForecaster.
    """

    DEFAULT_PARAM_GRID = {
        "changepoint_prior_scale": [0.01, 0.05, 0.1, 0.5],
        "seasonality_prior_scale": [0.1, 1.0, 10.0],
        "seasonality_mode": ["additive", "multiplicative"],
        "changepoint_range": [0.8, 0.9],
    }

    def __init__(
        self,
        param_grid: Optional[Dict[str, List[Any]]] = None,
        metric: str = "rmse",  # 'rmse', 'mae', 'mape'
        cv_initial: str = "730 days",   # 2 years initial train
        cv_period: str = "180 days",    # Cutoff step
        cv_horizon: str = "90 days",    # Forecast horizon
    ):
        self.param_grid = param_grid or self.DEFAULT_PARAM_GRID
        self.metric = metric.lower()
        self.cv_initial = cv_initial
        self.cv_period = cv_period
        self.cv_horizon = cv_horizon

        self.best_params_: Optional[Dict[str, Any]] = None
        self.best_score_: Optional[float] = None
        self.tuning_results_: Optional[pd.DataFrame] = None

    def grid_search_cv(
        self,
        df: pd.DataFrame,
        store: Optional[int] = None,
        item: Optional[int] = None,
        max_trials: Optional[int] = None,
        use_holdout_fallback: bool = True,
    ) -> Tuple[Dict[str, Any], pd.DataFrame]:
        """
        Runs Grid Search evaluating each combination via Prophet's rolling-origin cross-validation.
        If the dataset history is too short for rolling CV, falls back to fast holdout split.
        """
        data = RetailProphetForecaster.prepare_data(df, store=store, item=item)

        # Generate all parameter combinations
        keys = list(self.param_grid.keys())
        values = list(self.param_grid.values())
        all_combinations = [dict(zip(keys, v)) for v in itertools.product(*values)]

        if max_trials and max_trials < len(all_combinations):
            import random
            random.seed(42)
            combinations = random.sample(all_combinations, max_trials)
        else:
            combinations = all_combinations

        results = []
        best_score = float("inf")
        best_params = {}

        total = len(combinations)
        logging.info(f"Starting Prophet tuning across {total} parameter configurations...")

        for i, params in enumerate(combinations, 1):
            try:
                # Check if data allows standard rolling CV
                data_days = (data["ds"].max() - data["ds"].min()).days
                if data_days >= 900 and not use_holdout_fallback:
                    m = Prophet(**params)
                    m.fit(data)
                    df_cv = cross_validation(
                        m,
                        initial=self.cv_initial,
                        period=self.cv_period,
                        horizon=self.cv_horizon,
                        parallel=None,
                        disable_tqdm=True,
                    )
                    df_p = performance_metrics(df_cv, rolling_window=1)
                    score = float(df_p[self.metric].mean())
                else:
                    # Fast temporal holdout validation (last 90 days)
                    holdout_days = 90
                    train_data = data.iloc[:-holdout_days]
                    test_data = data.iloc[-holdout_days:]

                    m = Prophet(**params)
                    m.fit(train_data)
                    future = m.make_future_dataframe(periods=holdout_days, freq="D", include_history=False)
                    forecast = m.predict(future)

                    y_true = test_data["y"].values
                    y_pred = np.maximum(0, forecast["yhat"].values)

                    metrics = calculate_metrics(y_true, y_pred)
                    score = metrics[self.metric.upper()]

                record = {**params, self.metric: score}
                results.append(record)

                if score < best_score:
                    best_score = score
                    best_params = params

            except Exception as e:
                logging.warning(f"Trial {i}/{total} failed with params {params}: {e}")
                continue

        results_df = pd.DataFrame(results).sort_values(self.metric).reset_index(drop=True)
        self.best_params_ = best_params
        self.best_score_ = best_score
        self.tuning_results_ = results_df

        return best_params, results_df

    def tune_with_optuna(
        self,
        df: pd.DataFrame,
        store: Optional[int] = None,
        item: Optional[int] = None,
        n_trials: int = 20,
        random_seed: int = 42,
    ) -> Tuple[Dict[str, Any], float]:
        """
        Bayesian optimization using Optuna for efficient hyperparameter search.
        """
        try:
            import optuna
            optuna.logging.set_verbosity(optuna.logging.WARNING)
        except ImportError:
            raise ImportError("Optuna is required for tune_with_optuna(). Please install optuna.")

        data = RetailProphetForecaster.prepare_data(df, store=store, item=item)
        holdout_days = 90
        train_data = data.iloc[:-holdout_days]
        test_data = data.iloc[-holdout_days:]

        def objective(trial: optuna.Trial) -> float:
            changepoint_prior_scale = trial.suggest_float("changepoint_prior_scale", 0.001, 0.5, log=True)
            seasonality_prior_scale = trial.suggest_float("seasonality_prior_scale", 0.01, 20.0, log=True)
            seasonality_mode = trial.suggest_categorical("seasonality_mode", ["additive", "multiplicative"])
            changepoint_range = trial.suggest_float("changepoint_range", 0.75, 0.95)

            m = Prophet(
                changepoint_prior_scale=changepoint_prior_scale,
                seasonality_prior_scale=seasonality_prior_scale,
                seasonality_mode=seasonality_mode,
                changepoint_range=changepoint_range,
                yearly_seasonality=True,
                weekly_seasonality=True,
                daily_seasonality=False,
            )
            m.fit(train_data)
            future = m.make_future_dataframe(periods=holdout_days, freq="D", include_history=False)
            forecast = m.predict(future)

            y_true = test_data["y"].values
            y_pred = np.maximum(0, forecast["yhat"].values)
            metrics = calculate_metrics(y_true, y_pred)
            return metrics[self.metric.upper()]

        sampler = optuna.samplers.TPESampler(seed=random_seed)
        study = optuna.create_study(direction="minimize", sampler=sampler)
        study.optimize(objective, n_trials=n_trials)

        self.best_params_ = study.best_params
        self.best_score_ = study.best_value
        return study.best_params, study.best_value

    def get_best_forecaster(self, **kwargs) -> RetailProphetForecaster:
        """Returns a RetailProphetForecaster initialized with the tuned hyperparameters."""
        if not self.best_params_:
            raise RuntimeError("Hyperparameters have not been tuned yet. Call grid_search_cv() or tune_with_optuna().")
        params = {**self.best_params_, **kwargs}
        return RetailProphetForecaster(**params)
