"""
LSTM (Long Short-Term Memory) Forecasting Model for RetailLens.
AI / Deep Learning component of the forecasting system.

Architecture:
- Univariate time-series approach: uses a sliding window of past sales
- Two stacked LSTM layers with Dropout for regularization
- Dense output for single-step-ahead prediction
- Per-series MinMaxScaler (fitted ONLY on training data)
- Supports multi-step forecast generation via recursive prediction
- Saves/loads model + scaler to disk to avoid retraining on dashboard load

Owner: RetailLens AI Track
"""

import logging
import os
import pickle
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

from src.features.lstm_features import build_lstm_sequences
from src.utils.metrics import calculate_metrics

logger = logging.getLogger(__name__)
logging.getLogger("tensorflow").setLevel(logging.ERROR)
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "3")

# Default LSTM hyperparameters — tuned for accuracy and fast CPU inference
WINDOW_SIZE: int = 35          # 5-week look-back (captures weekly seasonal cycles)
LSTM_UNITS_1: int = 32         # Lightweight single-layer LSTM: fast CPU training, prevents overfitting
LSTM_UNITS_2: int = 0          # 0 = single layer; >0 adds stacked LSTM layer
DROPOUT_RATE: float = 0.1      # Regularization
EPOCHS: int = 25               # Efficient training with early stopping
BATCH_SIZE: int = 64           # Optimized CPU batch throughput
PATIENCE: int = 4              # Early stopping patience
LEARNING_RATE: float = 0.005   # Optimized Adam learning rate for faster convergence
LOSS: str = "mse"              # Training loss function

MODEL_DIR: str = "data/processed/models"


def _load_keras():
    """Lazy-load Keras to avoid slow startup when LSTM is not needed."""
    try:
        from tensorflow import keras  # type: ignore[import-untyped,import-not-found]
        return keras
    except ImportError as e:
        raise ImportError(
            "TensorFlow is required for the LSTM model. "
            "Install it with: pip install tensorflow-cpu"
        ) from e


class LSTMForecaster:
    """
    LSTM-based demand forecasting model (AI / Deep Learning).

    Fits a per-series sliding-window LSTM that takes the last `window`
    days of sales and predicts the next day's demand.

    Multi-step forecasting is done recursively: each prediction is fed
    back as input to produce the next prediction.
    """

    MODEL_NAME: str = "LSTM"

    def __init__(
        self,
        window: int = WINDOW_SIZE,
        lstm_units_1: int = LSTM_UNITS_1,
        lstm_units_2: int = LSTM_UNITS_2,
        dropout: float = DROPOUT_RATE,
        epochs: int = EPOCHS,
        batch_size: int = BATCH_SIZE,
        patience: int = PATIENCE,
        learning_rate: float = LEARNING_RATE,
        loss: str = LOSS,
        model_dir: str = MODEL_DIR,
    ):
        self.window = window
        self.lstm_units_1 = lstm_units_1
        self.lstm_units_2 = lstm_units_2
        self.dropout = dropout
        self.epochs = epochs
        self.batch_size = batch_size
        self.patience = patience
        self.learning_rate = learning_rate
        self.loss = loss
        self.model_dir = model_dir

        self._model = None          # Keras model
        self._scaler = None         # MinMaxScaler, fitted on training data only
        self.is_fitted: bool = False
        self.store_id: Optional[int] = None
        self.item_id: Optional[int] = None
        self.last_train_date: Optional[pd.Timestamp] = None
        self._last_window: Optional[np.ndarray] = None  # seed for recursive forecast

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _build_model(self, keras):
        """Constructs and compiles the Keras LSTM model."""
        layers = [keras.layers.Input(shape=(self.window, 1))]
        if self.lstm_units_2 and self.lstm_units_2 > 0:
            layers.append(keras.layers.LSTM(self.lstm_units_1, return_sequences=True))
            if self.dropout > 0:
                layers.append(keras.layers.Dropout(self.dropout))
            layers.append(keras.layers.LSTM(self.lstm_units_2, return_sequences=False))
        else:
            layers.append(keras.layers.LSTM(self.lstm_units_1, return_sequences=False))

        if self.dropout > 0:
            layers.append(keras.layers.Dropout(self.dropout))
        layers.append(keras.layers.Dense(16, activation="relu"))
        layers.append(keras.layers.Dense(1))

        model = keras.Sequential(layers)
        optimizer = keras.optimizers.Adam(learning_rate=self.learning_rate)
        model.compile(optimizer=optimizer, loss=self.loss)
        return model

    def _make_scaler(self):
        """Returns a fresh MinMaxScaler."""
        from sklearn.preprocessing import MinMaxScaler
        return MinMaxScaler(feature_range=(0, 1))

    def _model_path(self, store: int, item: int) -> str:
        return os.path.join(self.model_dir, f"lstm_s{store}_i{item}.keras")

    def _scaler_path(self, store: int, item: int) -> str:
        return os.path.join(self.model_dir, f"lstm_scaler_s{store}_i{item}.pkl")

    # ------------------------------------------------------------------
    # Prepare series from raw DataFrame
    # ------------------------------------------------------------------

    @staticmethod
    def prepare_series(
        df: pd.DataFrame,
        store: Optional[int] = None,
        item: Optional[int] = None,
    ) -> pd.Series:
        """
        Extracts a sorted, daily pd.Series of sales for (store, item).
        """
        data = df.copy()
        if store is not None and "store" in data.columns:
            data = data[data["store"] == store]
        if item is not None and "item" in data.columns:
            data = data[data["item"] == item]

        if "sales" in data.columns and "date" in data.columns:
            data["date"] = pd.to_datetime(data["date"])
            data = data.sort_values("date")
            series = data.set_index("date")["sales"].asfreq("D")
        else:
            raise ValueError("DataFrame must contain 'date' and 'sales' columns.")

        if series.isnull().any():
            series = series.interpolate(method="time").ffill().bfill()

        return series

    # ------------------------------------------------------------------
    # Training
    # ------------------------------------------------------------------

    def fit(
        self,
        df: pd.DataFrame,
        store: Optional[int] = None,
        item: Optional[int] = None,
        val_days: int = 90,
    ) -> "LSTMForecaster":
        """
        Fits the LSTM on training data.

        - Scaler is fitted ONLY on train split to avoid data leakage.
        - Validation set uses the last val_days of the training series.
        - Early stopping monitors val_loss.
        """
        keras = _load_keras()

        series = self.prepare_series(df, store=store, item=item)
        self.store_id = store
        self.item_id = item
        self.last_train_date = series.index.max()

        values = series.values.reshape(-1, 1).astype(np.float32)

        # Chronological split: train / val
        split_idx = max(self.window + 1, len(values) - val_days)
        train_vals = values[:split_idx]
        val_vals = values[split_idx - self.window:]  # include window prefix

        # Scaler fitted on train ONLY
        self._scaler = self._make_scaler()
        train_scaled = self._scaler.fit_transform(train_vals)
        val_scaled = self._scaler.transform(val_vals)

        # Build sequences
        X_train, y_train = build_lstm_sequences(train_scaled.ravel(), self.window)
        X_val, y_val = build_lstm_sequences(val_scaled.ravel(), self.window)

        if len(X_train) == 0:
            raise ValueError(
                f"Not enough data to build LSTM sequences. "
                f"Need at least {self.window + 1} rows; got {len(train_vals)}."
            )

        # Build and train model
        self._model = self._build_model(keras)
        cb_early_stop = keras.callbacks.EarlyStopping(
            monitor="val_loss", patience=self.patience,
            restore_best_weights=True, verbose=0,
        )
        cb_lr = keras.callbacks.ReduceLROnPlateau(
            monitor="val_loss", factor=0.5, patience=max(2, self.patience // 2), verbose=0
        )

        self._model.fit(
            X_train, y_train,
            validation_data=(X_val, y_val),
            epochs=self.epochs,
            batch_size=self.batch_size,
            callbacks=[cb_early_stop, cb_lr],
            verbose=0,
        )

        # Cache last training window (scaled) for recursive forecasting
        all_scaled = self._scaler.transform(values)
        self._last_window = all_scaled[-self.window:].ravel()
        self.is_fitted = True
        return self

    # ------------------------------------------------------------------
    # Prediction
    # ------------------------------------------------------------------

    def predict(
        self,
        horizon_days: int = 90,
        clip_non_negative: bool = True,
    ) -> pd.DataFrame:
        """
        Generates out-of-sample forecasts by recursively feeding predictions
        back into the LSTM window. Fast inference via predict_on_batch.

        Returns DataFrame with columns:
            ds, yhat, store (optional), item (optional)
        """
        if not self.is_fitted or self._model is None or self._scaler is None:
            raise RuntimeError("Model must be fitted before predicting.")
        if self._last_window is None:
            raise RuntimeError("No window seed available; call fit() first.")
        if self.last_train_date is None:
            raise RuntimeError("last_train_date is not set; call fit() first.")

        curr_window = self._last_window.copy()
        predictions_scaled = []

        for _ in range(horizon_days):
            x_input = curr_window[-self.window:].reshape(1, self.window, 1).astype(np.float32)
            pred_scaled = float(self._model.predict_on_batch(x_input)[0, 0])
            predictions_scaled.append(pred_scaled)
            curr_window = np.append(curr_window, pred_scaled)

        # Inverse-transform to original scale
        preds_array = np.array(predictions_scaled, dtype=np.float32).reshape(-1, 1)
        preds_original = self._scaler.inverse_transform(preds_array).ravel()

        if clip_non_negative:
            preds_original = np.maximum(0, preds_original)

        future_dates = pd.date_range(
            start=self.last_train_date + pd.Timedelta(days=1),
            periods=horizon_days,
            freq="D",
        )
        forecast_df = pd.DataFrame({"ds": future_dates, "yhat": preds_original})

        if self.store_id is not None:
            forecast_df["store"] = self.store_id
        if self.item_id is not None:
            forecast_df["item"] = self.item_id

        return forecast_df

    def forecast_horizons(
        self,
        horizons: List[int] = [30, 60, 90],
        clip_non_negative: bool = True,
    ) -> Dict[int, pd.DataFrame]:
        """Multi-step forecasts for standard horizons (30/60/90 days)."""
        max_h = max(horizons)
        full_fc = self.predict(horizon_days=max_h, clip_non_negative=clip_non_negative)
        return {h: full_fc.head(h).copy().reset_index(drop=True) for h in horizons}

    # ------------------------------------------------------------------
    # Holdout Evaluation
    # ------------------------------------------------------------------

    def evaluate_holdout(
        self,
        df: pd.DataFrame,
        holdout_days: int = 90,
        store: Optional[int] = None,
        item: Optional[int] = None,
        mode: str = "one_step",
        retrain: bool = True,
    ) -> Tuple[Dict[str, float], pd.DataFrame]:
        """
        Time-based holdout evaluation:
        - Trains on data up to (max_date - holdout_days)
        - Supports two evaluation modes:
            * 'one_step' (default): Rolling 1-step ahead prediction using actual
              historical values up to each test day (matching XGBoost's evaluation protocol).
            * 'recursive': Multi-step autoregressive recursive forecast without ground truth.
        - Computes MAE, RMSE, MAPE, sMAPE on test window.
        - Keeps self._model fitted so downstream prediction does not require retraining.

        Returns: (metrics_dict, holdout_comparison_df)
        """
        series = self.prepare_series(df, store=store, item=item)
        if len(series) <= holdout_days + self.window:
            raise ValueError(
                f"Series too short for holdout evaluation. "
                f"Need > {holdout_days + self.window} rows."
            )

        self.store_id = store
        self.item_id = item
        self.last_train_date = series.index.max()

        train_series_vals = series.iloc[: -holdout_days].values.reshape(-1, 1).astype(np.float32)
        test_series = series.iloc[-holdout_days:]

        # Check if we should retrain or use existing model
        should_train = retrain or (self._model is None) or (self._scaler is None) or (not self.is_fitted)

        if should_train:
            scaler = self._make_scaler()
            train_scaled = scaler.fit_transform(train_series_vals).ravel()

            X_train, y_train = build_lstm_sequences(train_scaled, self.window)
            if len(X_train) == 0:
                raise ValueError("Not enough data for holdout evaluation.")

            keras = _load_keras()
            eval_model = self._build_model(keras)

            # Chronological 85/15 validation split within train partition for early stopping
            val_split = max(self.window + 1, int(len(X_train) * 0.85))
            X_tr, y_tr = X_train[:val_split], y_train[:val_split]
            X_va, y_va = X_train[val_split:], y_train[val_split:]

            cb_early_stop = keras.callbacks.EarlyStopping(
                monitor="val_loss", patience=self.patience, restore_best_weights=True, verbose=0
            )
            cb_lr = keras.callbacks.ReduceLROnPlateau(
                monitor="val_loss", factor=0.5, patience=max(2, self.patience // 2), verbose=0
            )

            eval_model.fit(
                X_tr, y_tr,
                validation_data=(X_va, y_va),
                epochs=self.epochs,
                batch_size=self.batch_size,
                callbacks=[cb_early_stop, cb_lr],
                verbose=0,
            )

            self._model = eval_model
            self._scaler = scaler
            self.is_fitted = True
        else:
            if self._scaler is None or self._model is None:
                raise RuntimeError("Model and scaler must be fitted before evaluate_holdout when retrain=False.")
            scaler = self._scaler
            eval_model = self._model
            train_scaled = scaler.transform(train_series_vals).ravel()

        all_scaled = scaler.transform(series.values.reshape(-1, 1).astype(np.float32)).ravel()
        self._last_window = all_scaled[-self.window:].ravel()

        if mode == "one_step":
            # Vectorized 1-step prediction across all holdout days
            X_test_list = [
                all_scaled[i - self.window : i]
                for i in range(len(train_series_vals), len(series))
            ]
            X_test = np.array(X_test_list, dtype=np.float32).reshape(-1, self.window, 1)
            preds_scaled = eval_model.predict_on_batch(X_test).ravel()
            preds = scaler.inverse_transform(preds_scaled.reshape(-1, 1)).ravel()
        else:
            # Recursive prediction
            window = train_scaled[-self.window:].copy()
            preds_scaled = []
            for _ in range(holdout_days):
                x_in = window[-self.window:].reshape(1, self.window, 1).astype(np.float32)
                p = float(eval_model.predict_on_batch(x_in)[0, 0])
                preds_scaled.append(p)
                window = np.append(window, p)

            preds = scaler.inverse_transform(
                np.array(preds_scaled, dtype=np.float32).reshape(-1, 1)
            ).ravel()

        preds = np.maximum(0, preds)

        comparison = pd.DataFrame({
            "ds": test_series.index,
            "y": test_series.values,
            "yhat": preds,
        })
        metrics = calculate_metrics(
            y_true=comparison["y"],
            y_pred=comparison["yhat"],
        )
        return metrics, comparison

    # ------------------------------------------------------------------
    # Save / Load
    # ------------------------------------------------------------------

    def save(self, store: Optional[int] = None, item: Optional[int] = None) -> None:
        """Persists the Keras model and scaler to disk."""
        if not self.is_fitted or self._model is None:
            raise RuntimeError("Cannot save an unfitted model.")
        s = store if store is not None else self.store_id
        i = item if item is not None else self.item_id
        if s is None or i is None:
            raise ValueError("store_id and item_id must be set before saving.")
        os.makedirs(self.model_dir, exist_ok=True)
        self._model.save(self._model_path(s, i))
        with open(self._scaler_path(s, i), "wb") as f:
            pickle.dump(self._scaler, f)
        logger.info(f"LSTM model saved for store={s}, item={i}")

    def load(self, store: int, item: int) -> "LSTMForecaster":
        """Loads a previously saved Keras model and scaler from disk."""
        keras = _load_keras()
        model_path = self._model_path(store, item)
        scaler_path = self._scaler_path(store, item)

        if not os.path.exists(model_path):
            raise FileNotFoundError(f"LSTM model not found: {model_path}")
        if not os.path.exists(scaler_path):
            raise FileNotFoundError(f"LSTM scaler not found: {scaler_path}")

        self._model = keras.models.load_model(model_path)
        with open(scaler_path, "rb") as f:
            self._scaler = pickle.load(f)
        self.store_id = store
        self.item_id = item
        self.is_fitted = True
        return self

    @staticmethod
    def model_exists(store: int, item: int, model_dir: str = MODEL_DIR) -> bool:
        """Returns True if a saved model exists for this (store, item) pair."""
        return os.path.exists(
            os.path.join(model_dir, f"lstm_s{store}_i{item}.keras")
        )

    def save_forecast(self, forecast: pd.DataFrame, output_path: str) -> None:
        """Saves forecast DataFrame to CSV."""
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        forecast.to_csv(output_path, index=False)
        logger.info(f"LSTM forecast saved to {output_path}")
