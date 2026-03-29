# LSTM Autoencoder — Multivariate Sensor Anomaly Detector

> Real-time water leak detection in commercial buildings using unsupervised reconstruction-based anomaly detection on multivariate sensor streams.

[![Python](https://img.shields.io/badge/Python-3.10+-blue.svg)](https://python.org)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-red.svg)](https://pytorch.org)
[![License](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)

---

## Overview

Water leaks in large buildings are **rare events** — a typical commercial building experiences one confirmed leak per quarter. This makes supervised classification impractical due to severe class imbalance and sparse ground truth. This project implements an **unsupervised reconstruction-based approach**: train exclusively on normal operating data, then flag windows where reconstruction error exceeds a calibrated threshold as anomalous.

The system monitors four correlated sensor channels simultaneously and uses a **finite state machine confirmation layer** to suppress false positives from transient noise — matching how a real facilities management system should behave.

```
Sensor streams → Sliding window (seq=50) → LSTM Autoencoder → Reconstruction error → FSM → Alert
```

---

## Results

First run on 6,000-step synthetic building sensor data with 3 injected leak events:

| Metric | Value |
|--------|-------|
| Threshold | 0.1368 (μ + 3σ on val-normal) |
| Precision | **0.748** |
| Recall | **0.894** |
| F1 Score | **0.814** |
| Accuracy | **0.923** |
| Alert transitions | 14 — all 3 leak events detected |

All 3 leak events escalated to **CONFIRMED** state. The model trained for 60 epochs and converged cleanly. The high recall (0.894) reflects the design priority: in a building management context, missing a real leak is far more costly than a false alarm.

---

## Key Design Decisions

### Why LSTM Autoencoder over Isolation Forest or threshold rules?

Isolation Forest treats each timestep independently — it has no concept of temporal structure. A real water leak is not a single anomalous reading; it is a **pattern that evolves over 30–80 timesteps**: pressure drops first, flow rate increases downstream 5–10 timesteps later, temperature follows. The LSTM encoder learns the joint temporal distribution of all four channels, so a coordinated multi-channel pattern produces high reconstruction error even when each individual channel looks borderline.

### Why unsupervised?

A supervised classifier requires labeled leak events for training. In a real building deployment, you may have **zero confirmed historical leaks** when you first deploy. The reconstruction approach requires only normal operating data to bootstrap — available from day one.

### Why a finite state machine confirmation layer?

A single high-reconstruction window generates too many false alerts. The FSM requires the anomaly score to exceed the threshold for **3 consecutive windows** before alerting, and 5 windows at 1.5× threshold for confirmation. This suppresses transient sensor noise while maintaining low detection latency on real leak signatures.

```
NORMAL ──[score > threshold, 1 window]──────────► SUSPICIOUS
SUSPICIOUS ──[score > threshold, 3 consecutive]──► ALERT
ALERT ──[score > 1.5× threshold, 5 windows]──────► CONFIRMED
Any state ──[score below threshold, 5 windows]───► NORMAL
```

### Why joint multivariate reconstruction?

Modeling all four channels jointly means the autoencoder learns **cross-channel correlations** that define normal operation. A sensor fault (single channel spike, no neighbours affected) produces low reconstruction error in the other channels — the model correctly ignores it. A real leak creates correlated anomalies across pressure, flow, temperature, and vibration simultaneously — the joint reconstruction error is high. This is the core mechanism that separates leaks from noise.

---

## Architecture

```
Input: (batch, seq_len=50, features=4)
        │
        ▼
┌─────────────────────────────┐
│  Encoder                    │
│  LSTM layer 1 (hidden=64)   │
│  LSTM layer 2 (hidden=64)   │
│  Linear → latent (dim=16)   │
└─────────────────────────────┘
        │
        ▼ latent representation (dim=16)
        │
┌─────────────────────────────┐
│  Decoder                    │
│  Linear → hidden (dim=64)   │
│  LSTM layer 1 (hidden=64)   │
│  LSTM layer 2 (hidden=64)   │
│  Linear → output (dim=4)    │
└─────────────────────────────┘
        │
        ▼
Output: (batch, seq_len=50, features=4)
        │
        ▼
Reconstruction error: MSE per window across all 4 channels
```

---

## Sensor Data

Four channels simulating a commercial building pipe network:

| Channel | Normal range | Leak signature |
|---------|-------------|----------------|
| `pressure_bar` | 2.5–4.5 bar | Drops 15–25% at source |
| `flow_rate_lps` | 0.8–1.2 L/s | Increases 20–35% downstream |
| `temperature_c` | 15–20°C | Slight increase |
| `vibration_ms2` | 0.01–0.05 m/s² | Spike at leak onset |

All channels include sinusoidal day/night drift (24hr cycle) and Gaussian noise. Three leak events injected at known timesteps. Two single-sensor fault events included to test false positive suppression.

---

## Project Structure

```
LSTM-Autoencoder-multivariate-sensor-anomaly-detector/
├── data/
│   └── generate_sensor_data.py      # 6000-step synthetic 4-channel data, 3 leak events
├── models/
│   ├── __init__.py
│   └── lstm_autoencoder.py          # Encoder-Decoder LSTM (hidden=64, latent=16)
├── pipeline/
│   ├── __init__.py
│   ├── preprocessing.py             # StandardScaler + sliding windows (seq=50)
│   ├── detector.py                  # Train + threshold calibration + predict
│   └── alert_engine.py              # NORMAL→SUSPICIOUS→ALERT→CONFIRMED FSM
├── notebooks/
│   └── anomaly_detection_demo.ipynb # Full walkthrough with inline plots
├── tests/
│   ├── __init__.py
│   └── test_detector.py             # 20 unit tests across all modules
├── outputs/
│   ├── 01_sensor_data.png           # 4-channel time series + leak event shading
│   ├── 02_reconstruction_error.png  # Reconstruction error + threshold line
│   └── 03_alert_states.png          # FSM state colour timeline
├── main.py                          # End-to-end pipeline
└── requirements.txt
```

---

## Quickstart

```bash
git clone https://github.com/shradhanjalipradhan/LSTM-Autoencoder-multivariate-sensor-anomaly-detector
cd LSTM-Autoencoder-multivariate-sensor-anomaly-detector
pip install -r requirements.txt
python main.py
```

Runs in under 5 minutes on CPU. All plots saved to `outputs/`.

---

## Output Plots

**`01_sensor_data.png`** — All four sensor channels over 6,000 timesteps with leak event regions shaded. Shows the correlated multi-channel signature of each injected leak vs. single-channel sensor fault events.

**`02_reconstruction_error.png`** — Per-window reconstruction error with the calibrated threshold line (μ + 3σ on validation normal data). The three leak events are clearly visible as spikes above the threshold.

**`03_alert_states.png`** — FSM state timeline colour-coded by state (NORMAL / SUSPICIOUS / ALERT / CONFIRMED). Shows the confirmation latency from first threshold crossing to CONFIRMED state for each leak event.

---

## What Would Be Different in Production

1. **Real sensor ingestion** — Replace synthetic CSV with an MQTT subscriber (Paho) feeding a sliding buffer. Handle clock drift, out-of-order messages, and variable sampling rates at the ingestion boundary.

2. **Adaptive thresholding** — The μ + 3σ threshold drifts as building occupancy changes seasonally. In production, retrain the threshold monthly on recent normal data rather than fixing it at deployment time.

3. **Graph-structured localization** — This model detects *that* something is wrong. A companion GNN model over the pipe network topology identifies *where* — which sensor zone to inspect. See [pipe-network-gnn-localizer](https://github.com/shradhanjalipradhan/pipe-network-gnn-localizer) *(coming soon)*.

4. **Model monitoring** — Track input feature distribution (Population Stability Index) over time. Significant drift triggers a retraining job so the model's definition of "normal" stays current with the building's operating patterns.

5. **Multi-building transfer** — Fine-tune the threshold and scaler per building while keeping the LSTM weights frozen. The temporal dynamics of leak signatures are universal; the per-channel baselines differ by building.

---

## Tech Stack

- Python 3.10
- PyTorch 2.0
- scikit-learn, pandas, numpy
- matplotlib, seaborn
- pytest (20 unit tests)
- No GPU required

---

## Related Projects

- [pipe-network-gnn-localizer](https://github.com/shradhanjalipradhan/pipe-network-gnn-localizer) — GraphSAGE-based fault localization on pipe network topology *(companion project, coming soon)*

---

*Built as a technical proof-of-concept for water leak detection in commercial buildings using unsupervised anomaly detection on multivariate IoT sensor streams.*
