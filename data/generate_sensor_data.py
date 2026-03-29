"""
Generate synthetic multivariate sensor data for a water distribution pipeline.
Injects 3 leak events at timesteps 1500, 3500, and 5800.

Channels:
    pressure_bar   - pipe pressure in bar
    flow_rate_lps  - volumetric flow rate in litres/second
    temperature_c  - fluid temperature in degrees Celsius
    vibration_ms2  - pipe vibration in m/s²
"""

import numpy as np
import pandas as pd
from pathlib import Path


SENSOR_COLS = ["pressure_bar", "flow_rate_lps", "temperature_c", "vibration_ms2"]

# Leak events: (start_timestep, duration_timesteps)
LEAK_EVENTS = [
    (1500, 100),
    (3500, 150),
    (5800, 120),
]


def generate_sensor_data(n_timesteps: int = 6000, seed: int = 42) -> pd.DataFrame:
    """Return a DataFrame with synthetic sensor readings and ground-truth labels."""
    rng = np.random.default_rng(seed)
    t = np.arange(n_timesteps, dtype=float)

    # --- Normal operating signals (slow sinusoids + white noise) ---
    pressure = (
        5.0
        + 0.30 * np.sin(2 * np.pi * t / 100)
        + rng.normal(0, 0.05, n_timesteps)
    )
    flow_rate = (
        2.50
        + 0.20 * np.sin(2 * np.pi * t / 80 + 0.5)
        + rng.normal(0, 0.03, n_timesteps)
    )
    temperature = (
        20.0
        + 2.00 * np.sin(2 * np.pi * t / 500)
        + rng.normal(0, 0.10, n_timesteps)
    )
    vibration = (
        0.50
        + 0.10 * np.sin(2 * np.pi * t / 60)
        + rng.normal(0, 0.02, n_timesteps)
    )

    labels = np.zeros(n_timesteps, dtype=np.int32)

    # --- Inject leak anomalies ---
    for start, duration in LEAK_EVENTS:
        end = min(start + duration, n_timesteps)
        ramp = np.linspace(0, 1, end - start)

        pressure[start:end] -= 1.50 * ramp          # pressure drops
        flow_rate[start:end] += 0.80 * ramp          # flow spikes (leak path)
        temperature[start:end] += 3.00 * ramp        # temperature rises
        vibration[start:end] += 1.50 * ramp          # vibration intensifies

        labels[start:end] = 1

    df = pd.DataFrame(
        {
            "timestep": np.arange(n_timesteps, dtype=int),
            "pressure_bar": pressure,
            "flow_rate_lps": flow_rate,
            "temperature_c": temperature,
            "vibration_ms2": vibration,
            "label": labels,
        }
    )
    return df


def save_sensor_data(output_path: str = "data/sensor_data.csv") -> pd.DataFrame:
    df = generate_sensor_data()
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(output_path, index=False)
    print(f"[DataGen] Saved {len(df)} timesteps → {output_path}")
    return df


if __name__ == "__main__":
    save_sensor_data()
