"""
main.py — Water Leak Detection via LSTM Autoencoder

End-to-end pipeline:
  1. Generate 6000-timestep synthetic sensor data (4 channels, 3 leak events)
  2. Pre-process: scale + sliding-window  (70 / 15 / 15 split)
  3. Train LSTM Autoencoder on normal windows
  4. Threshold reconstruction error on val-normal windows
  5. Predict anomalies on test set
  6. Run state-machine alert engine
  7. Save three plots to outputs/

Usage
-----
    python main.py
"""

import warnings
warnings.filterwarnings("ignore")

from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use("Agg")          # headless backend — no display needed
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import torch

from data.generate_sensor_data import generate_sensor_data, SENSOR_COLS, LEAK_EVENTS
from models.lstm_autoencoder import LSTMAutoencoder
from pipeline.preprocessing import SensorPreprocessor
from pipeline.detector import AnomalyDetector
from pipeline.alert_engine import AlertEngine, AlertState


# ── Config ─────────────────────────────────────────────────────────────────
N_TIMESTEPS   = 6000
SEQ_LEN       = 50
STEP          = 1          # stride between consecutive windows
HIDDEN_DIM    = 64
LATENT_DIM    = 16
N_LAYERS      = 2
DROPOUT       = 0.2
LR            = 1e-3
BATCH_SIZE    = 256
N_EPOCHS      = 60
PATIENCE      = 8
THRESHOLD_K   = 3.0        # σ multiplier for anomaly threshold
OUTPUTS_DIR   = Path("outputs")
SEED          = 42

torch.manual_seed(SEED)
np.random.seed(SEED)
OUTPUTS_DIR.mkdir(exist_ok=True)


# ── 1. Generate data ────────────────────────────────────────────────────────
print("=" * 60)
print("  Water Leak Detection — LSTM Autoencoder Pipeline")
print("=" * 60)
print("\n[1/6] Generating sensor data...")
df = generate_sensor_data(n_timesteps=N_TIMESTEPS, seed=SEED)
print(f"      {len(df)} timesteps, {df['label'].sum()} anomalous timesteps")


# ── 2. Pre-process ──────────────────────────────────────────────────────────
print("\n[2/6] Pre-processing...")
prep = SensorPreprocessor(seq_len=SEQ_LEN, train_frac=0.70, val_frac=0.15, step=STEP)
X_train, X_val, X_test, y_train, y_val, y_test = prep.fit_transform(df)
print(f"      X_train:{X_train.shape}  X_val:{X_val.shape}  X_test:{X_test.shape}")


# ── 3. Train ────────────────────────────────────────────────────────────────
print("\n[3/6] Training LSTM Autoencoder...")
detector = AnomalyDetector(
    n_features  = len(SENSOR_COLS),
    hidden_dim  = HIDDEN_DIM,
    latent_dim  = LATENT_DIM,
    n_layers    = N_LAYERS,
    seq_len     = SEQ_LEN,
    dropout     = DROPOUT,
    lr          = LR,
    batch_size  = BATCH_SIZE,
    n_epochs    = N_EPOCHS,
    patience    = PATIENCE,
    threshold_k = THRESHOLD_K,
)
train_losses = detector.fit(X_train, y_train, X_val, y_val)


# ── 4. Evaluate ─────────────────────────────────────────────────────────────
print("\n[4/6] Evaluating on test set...")
metrics = detector.evaluate(X_test, y_test, split_name="test")

test_errors, test_anomalies = detector.predict(X_test)
# Compute val errors for plot
val_errors, _ = detector.predict(X_val)


# ── 5. Alert engine ─────────────────────────────────────────────────────────
print("\n[5/6] Running state-machine alert engine on test windows...")
engine = AlertEngine(suspicious_window=3, alert_window=6, recovery_window=5)
# offset by train+val window count so timesteps align with original series
train_val_windows = len(X_train) + len(X_val)
state_arr = engine.run(test_errors, test_anomalies, timestep_offset=train_val_windows)
engine.summary()


# ── 6. Save plots ───────────────────────────────────────────────────────────
print("\n[6/6] Saving plots to outputs/...")

STATE_COLORS = {
    AlertState.NORMAL.value - 1    : "#4caf50",   # green
    AlertState.SUSPICIOUS.value - 1: "#ff9800",   # orange
    AlertState.ALERT.value - 1     : "#f44336",   # red
    AlertState.CONFIRMED.value - 1 : "#9c27b0",   # purple
}
STATE_LABELS = ["NORMAL", "SUSPICIOUS", "ALERT", "CONFIRMED"]

# ── Plot 1: Raw sensor data with leak events annotated ──────────────────────
fig, axes = plt.subplots(4, 1, figsize=(14, 10), sharex=True)
fig.suptitle("Multivariate Sensor Data — Water Leak Events", fontsize=14, fontweight="bold")
colors = ["#1f77b4", "#ff7f0e", "#2ca02c", "#d62728"]
for ax, col, color in zip(axes, SENSOR_COLS, colors):
    ax.plot(df["timestep"], df[col], lw=0.6, color=color, label=col)
    ax.set_ylabel(col, fontsize=9)
    for start, dur in LEAK_EVENTS:
        ax.axvspan(start, min(start + dur, N_TIMESTEPS), alpha=0.18, color="crimson")
    ax.legend(loc="upper right", fontsize=8)
axes[-1].set_xlabel("Timestep")
# Legend entry for leak events
leak_patch = mpatches.Patch(color="crimson", alpha=0.4, label="Leak event")
fig.legend(handles=[leak_patch], loc="lower center", ncol=1, fontsize=9)
plt.tight_layout(rect=[0, 0.03, 1, 1])
out1 = OUTPUTS_DIR / "01_sensor_data.png"
fig.savefig(out1, dpi=120)
plt.close(fig)
print(f"  Saved -> {out1}")


# ── Plot 2: Reconstruction error (val + test) with threshold ────────────────
all_errors = np.concatenate([val_errors, test_errors])
# Approximate window timesteps (window i ends at timestep i * STEP + SEQ_LEN - 1)
val_ts_start  = len(X_train) * STEP + SEQ_LEN - 1
test_ts_start = (len(X_train) + len(X_val)) * STEP + SEQ_LEN - 1
val_ts   = np.arange(len(val_errors))  * STEP + val_ts_start
test_ts  = np.arange(len(test_errors)) * STEP + test_ts_start

fig, ax = plt.subplots(figsize=(14, 4))
ax.plot(val_ts,  val_errors,  lw=0.7, color="#1f77b4", alpha=0.7, label="Val error")
ax.plot(test_ts, test_errors, lw=0.7, color="#ff7f0e", alpha=0.7, label="Test error")
ax.axhline(detector.threshold, color="red", lw=1.5, ls="--",
           label=f"Threshold = {detector.threshold:.4f}")
for start, dur in LEAK_EVENTS:
    ax.axvspan(start, min(start + dur, N_TIMESTEPS), alpha=0.18, color="crimson")
ax.set_xlabel("Timestep")
ax.set_ylabel("Reconstruction Error (MSE)")
ax.set_title("LSTM Autoencoder — Reconstruction Error")
ax.legend(fontsize=9)
plt.tight_layout()
out2 = OUTPUTS_DIR / "02_reconstruction_error.png"
fig.savefig(out2, dpi=120)
plt.close(fig)
print(f"  Saved -> {out2}")


# ── Plot 3: Alert state timeline ────────────────────────────────────────────
fig, axes = plt.subplots(2, 1, figsize=(14, 6), sharex=True)

# Top: reconstruction error
ax = axes[0]
ax.plot(test_ts, test_errors, lw=0.7, color="#ff7f0e", label="Recon error")
ax.axhline(detector.threshold, color="red", lw=1.4, ls="--",
           label=f"Threshold = {detector.threshold:.4f}")
for start, dur in LEAK_EVENTS:
    ax.axvspan(start, min(start + dur, N_TIMESTEPS), alpha=0.15, color="crimson")
ax.set_ylabel("Recon Error")
ax.set_title("Alert Engine — State Machine Timeline")
ax.legend(fontsize=8)

# Bottom: state timeline (coloured step-plot)
ax = axes[1]
for i in range(len(state_arr) - 1):
    ax.fill_between(
        [test_ts[i], test_ts[i + 1]],
        [0, 0],
        [state_arr[i] + 1, state_arr[i] + 1],
        color=STATE_COLORS[state_arr[i]],
        step="pre",
        linewidth=0,
    )
ax.set_yticks([1, 2, 3, 4])
ax.set_yticklabels(STATE_LABELS, fontsize=8)
ax.set_xlabel("Timestep")
ax.set_ylabel("Alert State")

# Legend patches
patches = [
    mpatches.Patch(color=STATE_COLORS[i], label=STATE_LABELS[i]) for i in range(4)
]
ax.legend(handles=patches, loc="upper left", fontsize=8, ncol=4)
plt.tight_layout()
out3 = OUTPUTS_DIR / "03_alert_states.png"
fig.savefig(out3, dpi=120)
plt.close(fig)
print(f"  Saved -> {out3}")


# ── Summary ─────────────────────────────────────────────────────────────────
print("\n" + "=" * 60)
print("  PIPELINE COMPLETE")
print("=" * 60)
print(f"  Threshold         : {detector.threshold:.6f}")
print(f"  Test Precision    : {metrics['precision']:.4f}")
print(f"  Test Recall       : {metrics['recall']:.4f}")
print(f"  Test F1           : {metrics['f1']:.4f}")
print(f"  Test Accuracy     : {metrics['accuracy']:.4f}")
print(f"  Alert transitions : {len(engine.events)}")
print(f"  Outputs saved to  : {OUTPUTS_DIR.resolve()}/")
print("=" * 60)
