"""
State-machine alert engine for water-leak detection.

States
------
NORMAL      → baseline, no anomaly signal
SUSPICIOUS  → anomaly detected but below confirmation threshold
ALERT       → sustained anomaly, operator action recommended
CONFIRMED   → leak confirmed, immediate response required

Transitions
-----------
NORMAL      --[anomaly]--> SUSPICIOUS
SUSPICIOUS  --[anomaly x suspicious_window]--> ALERT
SUSPICIOUS  --[normal  x recovery_window ]--> NORMAL
ALERT       --[anomaly x alert_window    ]--> CONFIRMED
ALERT       --[normal  x recovery_window ]--> SUSPICIOUS
CONFIRMED   --[normal  x recovery_window ]--> NORMAL
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum, auto
from typing import List


class AlertState(Enum):
    NORMAL = auto()
    SUSPICIOUS = auto()
    ALERT = auto()
    CONFIRMED = auto()


@dataclass
class AlertEvent:
    timestep: int
    prev_state: AlertState
    new_state: AlertState
    reconstruction_error: float
    is_anomaly: bool

    def __str__(self) -> str:
        direction = "^" if self.new_state.value > self.prev_state.value else "v"
        return (
            f"t={self.timestep:5d}  {direction}  "
            f"{self.prev_state.name} -> {self.new_state.name}  "
            f"(err={self.reconstruction_error:.6f})"
        )


class AlertEngine:
    """Consume a stream of (error, is_anomaly) pairs and maintain state.

    Parameters
    ----------
    suspicious_window : int
        Consecutive anomalous windows needed to escalate NORMAL→SUSPICIOUS
        and SUSPICIOUS→ALERT.
    alert_window : int
        Consecutive anomalous windows needed to escalate ALERT→CONFIRMED.
    recovery_window : int
        Consecutive normal windows needed to de-escalate one step.
    """

    def __init__(
        self,
        suspicious_window: int = 3,
        alert_window: int = 6,
        recovery_window: int = 5,
    ):
        self.suspicious_window = suspicious_window
        self.alert_window = alert_window
        self.recovery_window = recovery_window

        self._state = AlertState.NORMAL
        self._consecutive_anomaly = 0
        self._consecutive_normal = 0

        self.state_history: List[AlertState] = []
        self.events: List[AlertEvent] = []

    # ------------------------------------------------------------------
    @property
    def state(self) -> AlertState:
        return self._state

    # ------------------------------------------------------------------
    def step(self, timestep: int, error: float, is_anomaly: bool) -> AlertState:
        """Process one window and return the current state after transition."""
        prev = self._state

        if is_anomaly:
            self._consecutive_anomaly += 1
            self._consecutive_normal = 0
        else:
            self._consecutive_normal += 1
            self._consecutive_anomaly = 0

        new_state = self._transition(is_anomaly)
        self._state = new_state
        self.state_history.append(new_state)

        if new_state != prev:
            event = AlertEvent(
                timestep=timestep,
                prev_state=prev,
                new_state=new_state,
                reconstruction_error=error,
                is_anomaly=is_anomaly,
            )
            self.events.append(event)
            print(f"[AlertEngine] {event}")

        return new_state

    # ------------------------------------------------------------------
    def _transition(self, is_anomaly: bool) -> AlertState:
        s = self._state

        if is_anomaly:
            if s == AlertState.NORMAL:
                if self._consecutive_anomaly >= self.suspicious_window:
                    return AlertState.SUSPICIOUS
            elif s == AlertState.SUSPICIOUS:
                if self._consecutive_anomaly >= self.alert_window:
                    return AlertState.ALERT
            elif s == AlertState.ALERT:
                if self._consecutive_anomaly >= self.alert_window:
                    return AlertState.CONFIRMED
            # CONFIRMED stays CONFIRMED while anomaly persists
        else:
            # Recovery de-escalates one step at a time
            if s != AlertState.NORMAL:
                if self._consecutive_normal >= self.recovery_window:
                    steps = {
                        AlertState.SUSPICIOUS: AlertState.NORMAL,
                        AlertState.ALERT: AlertState.SUSPICIOUS,
                        AlertState.CONFIRMED: AlertState.NORMAL,
                    }
                    new_state = steps[s]
                    # Reset counter so next de-escalation needs a fresh run
                    self._consecutive_normal = 0
                    return new_state

        return s  # no transition

    # ------------------------------------------------------------------
    def run(
        self,
        errors: "np.ndarray",
        anomalies: "np.ndarray",
        timestep_offset: int = 0,
    ) -> "np.ndarray":
        """Process all windows and return the integer state array.

        Parameters
        ----------
        errors    : (N,) reconstruction errors
        anomalies : (N,) binary anomaly flags
        timestep_offset : first window's timestep index

        Returns
        -------
        states : (N,) int array  (AlertState.value - 1 so NORMAL=0 …)
        """
        import numpy as np

        state_arr = np.empty(len(errors), dtype=np.int32)
        for i, (err, anom) in enumerate(zip(errors, anomalies)):
            s = self.step(
                timestep=timestep_offset + i,
                error=float(err),
                is_anomaly=bool(anom),
            )
            state_arr[i] = s.value - 1  # 0-based
        return state_arr

    # ------------------------------------------------------------------
    def summary(self) -> None:
        print(f"\n[AlertEngine] Total state transitions: {len(self.events)}")
        for event in self.events:
            print(f"  {event}")
