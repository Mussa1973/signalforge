"""
Signal analysis utilities: smoothing, FFT, statistics.
"""

import numpy as np
import pandas as pd


class SignalAnalyzer:
    """Computes statistics and spectra for a cleaned signal."""

    def __init__(self, df: pd.DataFrame):
        self.df = df
        self.time = df["time"].values
        self.value = df["value"].values

    # ---------- STATS ----------
    def stats(self) -> dict:
        v = self.value
        return {
            "samples": len(v),
            "duration_s": float(self.time[-1] - self.time[0]) if len(self.time) > 1 else 0.0,
            "mean": float(np.mean(v)),
            "std": float(np.std(v)),
            "min": float(np.min(v)),
            "max": float(np.max(v)),
            "rms": float(np.sqrt(np.mean(v**2))),
        }

    # ---------- SMOOTHING ----------
    def moving_average(self, window: int =2 ) -> np.ndarray:
        """Simple moving average (centered)."""
        if window < 2:
            return self.value.copy()
        kernel = np.ones(window) / window
        return np.convolve(self.value, kernel, mode="same")

    # ---------- FFT ----------
    def fft(self) -> tuple[np.ndarray, np.ndarray]:
        """Return (frequencies, magnitudes) of the signal."""
        n = len(self.value)
        if n < 2:
            return np.array([0.0]), np.array([0.0])

        # Estimate sample rate from median time delta
        dt = np.median(np.diff(self.time))
        if dt <= 0:
            dt = 1.0
        fs = 1.0 / dt  # sample rate in Hz

        # Remove mean to avoid DC spike
        centered = self.value - np.mean(self.value)
        freqs = np.fft.rfftfreq(n, d=dt)
        mags = np.abs(np.fft.rfft(centered)) / n
        return freqs, mags