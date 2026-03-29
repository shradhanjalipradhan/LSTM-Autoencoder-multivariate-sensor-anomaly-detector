"""
Data preprocessing pipeline.

Responsibilities
----------------
1. Fit / transform StandardScaler on training split only.
2. Create overlapping sliding-window sequences (shape: N × seq_len × n_features).
3. Return train / val / test splits as PyTorch tensors.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import torch
from sklearn.preprocessing import StandardScaler


SENSOR_COLS = ["pressure_bar", "flow_rate_lps", "temperature_c", "vibration_ms2"]


class SensorPreprocessor:
    """Preprocessing helper for the 4-channel water-leak sensor dataset.

    Parameters
    ----------
    seq_len : int
        Sliding-window length in timesteps.
    train_frac : float
        Fraction of data used for training (default 0.70).
    val_frac : float
        Fraction of data used for validation (default 0.15).
        The remainder (1 - train_frac - val_frac) becomes the test set.
    step : int
        Stride of the sliding window (default 1).
    """

    def __init__(
        self,
        seq_len: int = 50,
        train_frac: float = 0.70,
        val_frac: float = 0.15,
        step: int = 1,
    ):
        self.seq_len = seq_len
        self.train_frac = train_frac
        self.val_frac = val_frac
        self.step = step
        self.scaler = StandardScaler()

    # ------------------------------------------------------------------
    def fit_transform(
        self, df: pd.DataFrame
    ) -> tuple[
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        np.ndarray,
        np.ndarray,
        np.ndarray,
    ]:
        """Fit scaler on training split, then build windowed tensors.

        Returns
        -------
        X_train, X_val, X_test : torch.Tensor  (N, seq_len, n_features)
        y_train, y_val, y_test : np.ndarray     (N,)  window-level labels
                                                 (1 if any label==1 in window)
        """
        values = df[SENSOR_COLS].values.astype(np.float32)
        labels = df["label"].values.astype(np.int32)

        n = len(values)
        train_end = int(n * self.train_frac)
        val_end = int(n * (self.train_frac + self.val_frac))

        # Fit scaler only on training data to prevent data leakage
        self.scaler.fit(values[:train_end])
        scaled = self.scaler.transform(values)

        X, y = self._make_windows(scaled, labels)

        # Split by approximate fraction of the original series
        # Each window[i] ends at timestep  i * step + seq_len - 1
        window_end_ts = np.arange(len(X)) * self.step + self.seq_len - 1

        train_mask = window_end_ts < train_end
        val_mask = (window_end_ts >= train_end) & (window_end_ts < val_end)
        test_mask = window_end_ts >= val_end

        def _t(arr: np.ndarray) -> torch.Tensor:
            return torch.from_numpy(arr)

        X_train = _t(X[train_mask])
        X_val = _t(X[val_mask])
        X_test = _t(X[test_mask])

        y_train = y[train_mask]
        y_val = y[val_mask]
        y_test = y[test_mask]

        print(
            f"[Preprocessor] windows -> train:{len(X_train)}  "
            f"val:{len(X_val)}  test:{len(X_test)}"
        )
        return X_train, X_val, X_test, y_train, y_val, y_test

    # ------------------------------------------------------------------
    def transform(self, df: pd.DataFrame) -> torch.Tensor:
        """Transform a raw DataFrame using the already-fitted scaler."""
        values = df[SENSOR_COLS].values.astype(np.float32)
        scaled = self.scaler.transform(values)
        labels = np.zeros(len(scaled), dtype=np.int32)
        X, _ = self._make_windows(scaled, labels)
        return torch.from_numpy(X)

    # ------------------------------------------------------------------
    def _make_windows(
        self, scaled: np.ndarray, labels: np.ndarray
    ) -> tuple[np.ndarray, np.ndarray]:
        n = len(scaled)
        starts = range(0, n - self.seq_len + 1, self.step)
        X = np.stack([scaled[s : s + self.seq_len] for s in starts])
        y = np.array(
            [int(labels[s : s + self.seq_len].any()) for s in starts],
            dtype=np.int32,
        )
        return X, y
