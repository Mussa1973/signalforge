# 🧪 Python-powered Communication Signal Analyzer

A Python GUI tool that reads **uncleaned** laboratory signal CSV files,
cleans them automatically, and visualises the **pure signal**.

![Python](https://img.shields.io/badge/Python-3.10%2B-blue)
![License](https://img.shields.io/badge/License-MIT-green)

---

## ✨ Features

- 📂 **Load any CSV** — auto-detects time & value columns
- 🧹 **Automatic cleaning**
  - Removes non-numeric rows
  - Merges duplicate timestamps
  - Interpolates small gaps
  - Removes spikes via **rolling MAD** (robust to outliers)
- 📈 **Interactive plots** (zoom, pan, save)
- 🎛️ **Real-time smoothing** (adjustable moving-average window)
- 🔬 **FFT spectrum viewer** for frequency analysis
- 📊 **Statistics tab** with cleaning report
- 💾 **Export cleaned data** to CSV

---

## 🚀 Installation

```bash
git clone https://github.com/yourusername/laboratory-signal-analyzer.git
cd laboratory-signal-analyzer
pip install -r requirements.txt