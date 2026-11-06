"""
Data cleaning engine for laboratory signal CSV files.

Handles common real-world CSV problems:
- Missing values
- Duplicate timestamps
- Outliers (spikes)
- Non-numeric garbage
- Unsorted timestamps
- Time gaps (resampling)
"""

import pandas as pd
import numpy as np


class SignalCleaner:
    """Cleans dirty laboratory signal data from CSV files."""

    def __init__(self, filepath: str):
        self.filepath = filepath
        self.raw_df: pd.DataFrame | None = None
        self.clean_df: pd.DataFrame | None = None
        self.report: dict = {}

    # ---------- LOAD ----------
    def load(self) -> pd.DataFrame:
        df = None
        last_err = None
        for sep in [",", ";", "\t"]:
            try:
                df = pd.read_csv(
                    self.filepath, sep=sep, encoding="utf-8-sig",
                    on_bad_lines="skip", engine="python",
                )
                if df is not None and df.shape[1] >= 2:
                    break
            except Exception as e:
                last_err = e
                df = None

        if df is None or df.shape[1] < 2:
            raise ValueError(f"Could not parse '{self.filepath}'. Last error: {last_err}")

        df.columns = [str(c).strip().lower().replace("\ufeff", "") for c in df.columns]

        time_col = self._detect_column(df, ["time", "timestamp", "t", "seconds", "date"])
        value_col = self._detect_column(
            df, ["envelope", "value", "signal", "amplitude", "voltage", "reading", "measurement"]
        )

        if time_col is None or value_col is None:
            raise ValueError(f"Could not detect time/value columns. Found: {list(df.columns)}")

        # Keep the envelope as the primary value. If a phase column exists, we
        # store it separately for future analysis but don't plot it by default.
        self.phase_df = None
        if "phase_deg" in df.columns:
            self.phase_df = df[["time", "phase_deg"]].copy() if time_col == "time" else None

        df = df[[time_col, value_col]].rename(columns={time_col: "time", value_col: "value"})
        self.raw_df = df
        self.report["raw_rows"] = len(df)
        return df

    @staticmethod
    def _detect_column(df: pd.DataFrame, candidates: list[str]) -> str | None:
        for c in df.columns:
            if any(cand in c for cand in candidates):
                return c
        return None

    # ---------- CLEAN ----------
    def clean(self) -> pd.DataFrame:
        """Run the full cleaning pipeline."""
        if self.raw_df is None:
            self.load()

        df = self.raw_df.copy()
        issues = {}

        # 1. Coerce to numeric (drop garbage rows)
        df["time"] = pd.to_numeric(df["time"], errors="coerce")
        df["value"] = pd.to_numeric(df["value"], errors="coerce")
        before = len(df)
        df = df.dropna(subset=["time", "value"])
        issues["non_numeric_dropped"] = before - len(df)

        # 2. Sort by time
        df = df.sort_values("time").reset_index(drop=True)

        # 3. Remove duplicate timestamps (keep mean)
        before = len(df)
        df = df.groupby("time", as_index=False)["value"].mean()
        issues["duplicates_merged"] = before - len(df)

        # 4. Interpolate small gaps (up to 5 consecutive NaN)
        df["value"] = df["value"].interpolate(limit=5)

        # 5. Outlier removal via rolling MAD (Median Absolute Deviation)
        df = self._remove_outliers(df)
        issues["outliers_removed"] = self._last_outliers

        # 6. Final safety: drop any remaining NaN
        df = df.dropna().reset_index(drop=True)

        self.clean_df = df
        self.report.update(issues)
        self.report["clean_rows"] = len(df)
        return df

    def _remove_outliers(self, df: pd.DataFrame, window: int = 15, threshold: float = 4.0) -> pd.DataFrame:
        """Remove spikes using rolling MAD. Robust to noisy lab data."""
        values = df["value"].values
        n = len(values)
        if n < window:
            self._last_outliers = 0
            return df

        rolling_med = pd.Series(values).rolling(window, center=True, min_periods=1).median().values
        mad = pd.Series(np.abs(values - rolling_med)).rolling(window, center=True, min_periods=1).median().values
        # Avoid divide-by-zero
        mad = np.where(mad == 0, 1e-9, mad)
        modified_z = 0.6745 * (values - rolling_med) / mad

        mask = np.abs(modified_z) < threshold
        # Always keep endpoints
        mask[0] = mask[-1] = True
        self._last_outliers = int((~mask).sum())
        return df[mask].reset_index(drop=True)

    # ---------- EXPORT ----------
    def export(self, out_path: str):
        """Save cleaned data to CSV."""
        if self.clean_df is None:
            raise RuntimeError("Run .clean() before exporting.")
        self.clean_df.to_csv(out_path, index=False)