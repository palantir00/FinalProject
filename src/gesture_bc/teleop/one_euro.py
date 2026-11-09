"""One-Euro filter (Casiez, Roussel, Vogel, CHI 2012): smoothing of noisy real-time signals.

Ported from the earlier Gesture-to-Geometry project (app/src/cv/filters/oneEuro.ts).

An ordinary low-pass filter must choose between jitter and lag: a low cutoff frequency
removes jitter, but fast movements then trail behind. The One-Euro filter adapts the cutoff
to the speed of the signal:
    hand (almost) still  -> cutoff = min_cutoff_hz             (strong smoothing, no jitter)
    hand moving fast     -> cutoff = min_cutoff_hz + beta*speed (little smoothing, little lag)
The speed itself is a low-pass filtered derivative (cutoff d_cutoff_hz), so noise does not
open the filter.
"""

import numpy as np


def smoothing_factor(dt: float, cutoff_hz: float | np.ndarray) -> float | np.ndarray:
    """Weight of a new sample in an exponential low-pass filter with this cutoff frequency."""
    tau = 1.0 / (2.0 * np.pi * cutoff_hz)  # time constant of the filter [s]
    return 1.0 / (1.0 + tau / dt)


class OneEuroFilter:
    """Filters a vector signal; every element is filtered independently."""

    # Time steps are clamped to this range [s], as in the TypeScript version: two frames with
    # the same timestamp (dt = 0) or a long pause must not break the filter.
    MIN_DT, MAX_DT = 1.0 / 240.0, 1.0

    def __init__(self, min_cutoff_hz: float = 1.0, beta: float = 0.0, d_cutoff_hz: float = 1.0):
        self.min_cutoff_hz = min_cutoff_hz
        self.beta = beta
        self.d_cutoff_hz = d_cutoff_hz
        self.reset()

    def reset(self) -> None:
        """Forget the history: the next sample passes through unchanged."""
        self._value: np.ndarray | None = None  # last filtered value
        self._speed: np.ndarray | None = None  # last filtered derivative [units/s]
        self._time_s = 0.0

    def __call__(self, x: np.ndarray, time_s: float) -> np.ndarray:
        x = np.asarray(x, dtype=float)
        if self._value is None:
            self._value, self._speed, self._time_s = x.copy(), np.zeros_like(x), time_s
            return self._value.copy()

        dt = float(np.clip(time_s - self._time_s, self.MIN_DT, self.MAX_DT))
        self._time_s = time_s
        # 1) Smoothed speed of the signal (the paper starts it at 0).
        raw_speed = (x - self._value) / dt
        self._speed += smoothing_factor(dt, self.d_cutoff_hz) * (raw_speed - self._speed)
        # 2) Cutoff grows with the speed; then an ordinary low-pass step with that cutoff.
        cutoff_hz = self.min_cutoff_hz + self.beta * np.abs(self._speed)
        self._value += smoothing_factor(dt, cutoff_hz) * (x - self._value)
        return self._value.copy()
