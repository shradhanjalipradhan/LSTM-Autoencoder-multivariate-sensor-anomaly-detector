"""
Unit tests for the water-leak anomaly detection pipeline.
Run with: python -m pytest tests/ -v
"""

import numpy as np
import pandas as pd
import torch
import pytest

from data.generate_sensor_data import generate_sensor_data, SENSOR_COLS, LEAK_EVENTS
from models.lstm_autoencoder import LSTMAutoencoder
from pipeline.preprocessing import SensorPreprocessor
from pipeline.alert_engine import AlertEngine, AlertState


# ---------------------------------------------------------------------------
# Data generation
# ---------------------------------------------------------------------------

class TestDataGeneration:
    def test_shape(self):
        df = generate_sensor_data(n_timesteps=200)
        assert len(df) == 200
        assert list(df.columns) == ["timestep", *SENSOR_COLS, "label"]

    def test_sensor_columns_present(self):
        df = generate_sensor_data(n_timesteps=100)
        for col in SENSOR_COLS:
            assert col in df.columns

    def test_labels_binary(self):
        df = generate_sensor_data(n_timesteps=200)
        assert set(df["label"].unique()).issubset({0, 1})

    def test_normal_pressure_range(self):
        df = generate_sensor_data(n_timesteps=500, seed=0)
        normal = df[df["label"] == 0]
        assert normal["pressure_bar"].between(3.0, 7.0).all()

    def test_reproducibility(self):
        df1 = generate_sensor_data(seed=7)
        df2 = generate_sensor_data(seed=7)
        assert (df1.values == df2.values).all()


# ---------------------------------------------------------------------------
# LSTM Autoencoder
# ---------------------------------------------------------------------------

class TestLSTMAutoencoder:
    def _make_model(self, seq_len=20):
        return LSTMAutoencoder(
            n_features=4, hidden_dim=16, latent_dim=4, n_layers=1, seq_len=seq_len
        )

    def test_forward_output_shape(self):
        model = self._make_model(seq_len=20)
        x = torch.randn(8, 20, 4)
        out = model(x)
        assert out.shape == (8, 20, 4)

    def test_encode_shape(self):
        model = self._make_model(seq_len=20)
        x = torch.randn(8, 20, 4)
        latent = model.encode(x)
        assert latent.shape == (8, 4)  # (batch, latent_dim)

    def test_reconstruction_error_shape(self):
        model = self._make_model(seq_len=20)
        x = torch.randn(8, 20, 4)
        errs = model.reconstruction_error(x)
        assert errs.shape == (8,)

    def test_reconstruction_error_nonneg(self):
        model = self._make_model(seq_len=20)
        x = torch.randn(16, 20, 4)
        errs = model.reconstruction_error(x)
        assert (errs >= 0).all()


# ---------------------------------------------------------------------------
# Preprocessor
# ---------------------------------------------------------------------------

class TestSensorPreprocessor:
    def _make_df(self, n=300):
        return generate_sensor_data(n_timesteps=n, seed=1)

    def test_split_sizes(self):
        df = self._make_df(300)
        prep = SensorPreprocessor(seq_len=10, step=1)
        X_tr, X_v, X_te, *_ = prep.fit_transform(df)
        total = len(X_tr) + len(X_v) + len(X_te)
        # total windows = n - seq_len + 1
        assert total == 300 - 10 + 1

    def test_tensor_dtype(self):
        df = self._make_df(200)
        prep = SensorPreprocessor(seq_len=10, step=5)
        X_tr, X_v, X_te, *_ = prep.fit_transform(df)
        for X in (X_tr, X_v, X_te):
            assert X.dtype == torch.float32

    def test_window_shape(self):
        df = self._make_df(200)
        prep = SensorPreprocessor(seq_len=15, step=5)
        X_tr, *_ = prep.fit_transform(df)
        assert X_tr.shape[1] == 15
        assert X_tr.shape[2] == 4


# ---------------------------------------------------------------------------
# Alert Engine
# ---------------------------------------------------------------------------

class TestAlertEngine:
    def test_initial_state_normal(self):
        engine = AlertEngine()
        assert engine.state == AlertState.NORMAL

    def test_escalates_to_suspicious(self):
        engine = AlertEngine(suspicious_window=3, alert_window=6, recovery_window=5)
        for i in range(3):
            engine.step(i, error=1.0, is_anomaly=True)
        assert engine.state == AlertState.SUSPICIOUS

    def test_escalates_to_alert(self):
        engine = AlertEngine(suspicious_window=3, alert_window=6, recovery_window=5)
        for i in range(9):
            engine.step(i, error=1.0, is_anomaly=True)
        assert engine.state == AlertState.ALERT

    def test_escalates_to_confirmed(self):
        engine = AlertEngine(suspicious_window=3, alert_window=6, recovery_window=5)
        for i in range(20):
            engine.step(i, error=1.0, is_anomaly=True)
        assert engine.state == AlertState.CONFIRMED

    def test_recovery_to_normal(self):
        engine = AlertEngine(suspicious_window=3, alert_window=6, recovery_window=3)
        # Escalate
        for i in range(3):
            engine.step(i, error=1.0, is_anomaly=True)
        assert engine.state == AlertState.SUSPICIOUS
        # Recover
        for i in range(3, 6):
            engine.step(i, error=0.0, is_anomaly=False)
        assert engine.state == AlertState.NORMAL

    def test_run_returns_array(self):
        engine = AlertEngine()
        errors = np.ones(10, dtype=np.float32)
        anomalies = np.array([1, 0, 1, 0, 1, 1, 1, 1, 1, 1], dtype=np.int32)
        states = engine.run(errors, anomalies)
        assert len(states) == 10

    def test_events_recorded_on_transition(self):
        engine = AlertEngine(suspicious_window=2, alert_window=4, recovery_window=3)
        for i in range(2):
            engine.step(i, error=1.0, is_anomaly=True)
        assert len(engine.events) >= 1
        assert engine.events[0].new_state == AlertState.SUSPICIOUS


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
