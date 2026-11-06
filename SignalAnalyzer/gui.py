"""
Tkinter GUI for the Laboratory Signal Analyzer.
"""

import tkinter as tk
from tkinter import ttk, filedialog, messagebox

import matplotlib
matplotlib.use("TkAgg")
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg, NavigationToolbar2Tk
from matplotlib.figure import Figure
from SignalAnalyzer.cleaner import SignalCleaner
from SignalAnalyzer.analyzer import SignalAnalyzer


class SignalAnalyzerApp:
    """Main GUI application window."""

    def __init__(self):
        self.root = tk.Tk()
        self.root.title("Signal Forge - Python-powered Communication Signal Analyzer")
        self.root.geometry("1100x750")

        self.cleaner: SignalCleaner | None = None
        self.analyzer: SignalAnalyzer | None = None

        self._build_ui()

    # ---------- UI ----------
    def _build_ui(self):
        # Top toolbar
        top = ttk.Frame(self.root, padding=8)
        top.pack(side=tk.TOP, fill=tk.X)

        ttk.Button(top, text="📂 Load CSV", command=self._load_file).pack(side=tk.LEFT, padx=4)
        ttk.Button(top, text="🧹 Clean Data", command=self._clean_data).pack(side=tk.LEFT, padx=4)
        ttk.Button(top, text="💾 Export Clean CSV", command=self._export).pack(side=tk.LEFT, padx=4)

        ttk.Label(top, text="Smoothing window:").pack(side=tk.LEFT, padx=(20, 4))
        self.smooth_var = tk.IntVar(value=1)
        ttk.Spinbox(top, from_=1, to=51, width=5, textvariable=self.smooth_var,
                    command=self._refresh).pack(side=tk.LEFT)

        ttk.Label(top, text="   Show:").pack(side=tk.LEFT)
        self.view_var = tk.StringVar(value="Raw + Clean + Smoothed")
        combo = ttk.Combobox(top, textvariable=self.view_var, state="readonly", width=28,
                             values=["Raw", "Clean", "Clean + Smoothed", "Raw + Clean + Smoothed"])
        combo.pack(side=tk.LEFT, padx=4)
        combo.bind("<<ComboboxSelected>>", lambda e: self._refresh())

        ttk.Button(top, text="🔬 FFT View", command=self._show_fft).pack(side=tk.LEFT, padx=10)

        # Status
        self.status_var = tk.StringVar(value="Ready. Load a CSV to begin.")
        status = ttk.Label(self.root, textvariable=self.status_var, relief=tk.SUNKEN, anchor="w")
        status.pack(side=tk.BOTTOM, fill=tk.X)

        # Notebook with tabs
        self.notebook = ttk.Notebook(self.root)
        self.notebook.pack(fill=tk.BOTH, expand=True, padx=8, pady=8)

        # Time-domain tab
        self.time_frame = ttk.Frame(self.notebook)
        self.notebook.add(self.time_frame, text="Time Domain")
        self._build_time_canvas()

        # Stats tab
        self.stats_frame = ttk.Frame(self.notebook)
        self.notebook.add(self.stats_frame, text="Statistics")
        self.stats_text = tk.Text(self.stats_frame, wrap="word", font=("Courier", 11))
        self.stats_text.pack(fill=tk.BOTH, expand=True, padx=8, pady=8)

    def _build_time_canvas(self):
        self.fig = Figure(figsize=(10, 5), dpi=100)
        self.ax = self.fig.add_subplot(111)
        self.ax.set_xlabel("Time (s)")
        self.ax.set_ylabel("Value")
        self.ax.set_title("Signal")
        self.ax.grid(True, alpha=0.3)

        self.canvas = FigureCanvasTkAgg(self.fig, master=self.time_frame)
        self.canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)

        toolbar = NavigationToolbar2Tk(self.canvas, self.time_frame)
        toolbar.update()

    # ---------- ACTIONS ----------
    def _load_file(self):
        path = filedialog.askopenfilename(
            title="Select signal CSV",
            filetypes=[("CSV files", "*.csv"), ("All files", "*.*")],
        )
        if not path:
            return

        try:
            self.cleaner = SignalCleaner(path)
            self.cleaner.load()
            self.status_var.set(f"Loaded {len(self.cleaner.raw_df)} rows from {path}")
            self._plot_raw_only()
        except Exception as e:
            messagebox.showerror("Load Error", str(e))

    def _clean_data(self):
        if self.cleaner is None:
            messagebox.showwarning("No data", "Load a CSV first.")
            return

        try:
            df = self.cleaner.clean()
            self.analyzer = SignalAnalyzer(df)
            self.status_var.set(
                f"Cleaned. Rows: {self.cleaner.report['raw_rows']} → {self.cleaner.report['clean_rows']}  |  "
                f"Outliers removed: {self.cleaner.report.get('outliers_removed', 0)}"
            )
            self._update_stats()
            self._refresh()
        except Exception as e:
            messagebox.showerror("Cleaning Error", str(e))

    def _export(self):
        if self.cleaner is None or self.cleaner.clean_df is None:
            messagebox.showwarning("Nothing to export", "Load and clean the data first.")
            return
        path = filedialog.asksaveasfilename(
            defaultextension=".csv",
            filetypes=[("CSV files", "*.csv")],
            title="Save cleaned CSV",
        )
        if path:
            self.cleaner.export(path)
            self.status_var.set(f"Exported cleaned data to {path}")

    # ---------- PLOTTING ----------
    def _plot_raw_only(self):
        self.ax.clear()
        self.ax.grid(True, alpha=0.3)
        self.ax.plot(self.cleaner.raw_df["time"], self.cleaner.raw_df["value"],
                     color="blue", lw=0.8, label="Raw (dirty)")
        self.ax.legend(loc="upper right")
        self.ax.set_xlabel("Time (s)")
        self.ax.set_ylabel("Value")
        self.ax.set_title("Raw Signal")
        self.canvas.draw()

    def _refresh(self):
        if self.cleaner is None or self.cleaner.clean_df is None:
            return

        view = self.view_var.get()
        self.ax.clear()
        self.ax.grid(True, alpha=0.3)

        if "Raw" in view:
            self.ax.plot(self.cleaner.raw_df["time"], self.cleaner.raw_df["value"],
                         color="black", lw=0.7, alpha=0.6, label="Raw (dirty)")

        if "Clean" in view:
            self.ax.plot(self.analyzer.time, self.analyzer.value,
                         color="steelblue", lw=1.0, label="Clean")

        if "Smoothed" in view:
            w = max(1, self.smooth_var.get())
            smoothed = self.analyzer.moving_average(window=w)
            self.ax.plot(self.analyzer.time, smoothed,
                         color="darkorange", lw=1.6, label=f"Smoothed (w={w})")

        self.ax.set_xlabel("Time (s)")
        self.ax.set_ylabel("Value")
        self.ax.set_title("Signal Forge: Python-powered Communication Signal Analyzer")
        self.ax.legend(loc="upper right")
        self.canvas.draw()

    def _show_fft(self):
        if self.analyzer is None:
            messagebox.showwarning("No data", "Clean the data first.")
            return

        freqs, mags = self.analyzer.fft()

        win = tk.Toplevel(self.root)
        win.title("FFT Spectrum")
        win.geometry("900x500")

        fig = Figure(figsize=(9, 5), dpi=100)
        ax = fig.add_subplot(111)
        ax.plot(freqs, mags, color="green", lw=1.0)
        ax.set_xlabel("Frequency (Hz)")
        ax.set_ylabel("Magnitude")
        ax.set_title("Frequency Spectrum (FFT)")
        ax.grid(True, alpha=0.3)

        canvas = FigureCanvasTkAgg(fig, master=win)
        canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)
        canvas.draw()

    # ---------- STATS ----------
    def _update_stats(self):
        if self.analyzer is None:
            return
        stats = self.analyzer.stats()
        self.stats_text.delete("1.0", tk.END)
        self.stats_text.insert(
            tk.END,
            f"=== Cleaning Report ===\n"
            f"Raw rows:              {self.cleaner.report.get('raw_rows')}\n"
            f"Clean rows:            {self.cleaner.report.get('clean_rows')}\n"
            f"Non-numeric dropped:   {self.cleaner.report.get('non_numeric_dropped', 0)}\n"
            f"Duplicates merged:     {self.cleaner.report.get('duplicates_merged', 0)}\n"
            f"Outliers removed:      {self.cleaner.report.get('outliers_removed', 0)}\n\n"
            f"=== Signal Statistics ===\n"
            f"Samples:               {stats['samples']}\n"
            f"Duration (s):          {stats['duration_s']:.3f}\n"
            f"Mean:                  {stats['mean']:.4f}\n"
            f"Std dev:               {stats['std']:.4f}\n"
            f"Min:                   {stats['min']:.4f}\n"
            f"Max:                   {stats['max']:.4f}\n"
            f"RMS:                   {stats['rms']:.4f}\n"
        )

    # ---------- RUN ----------
    def run(self):
        self.root.mainloop()