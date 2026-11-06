"""
Generate a guaranteed-valid dirty signal CSV for testing.
Run:  python make_sample.py
Output: sample_data/dirty_signal.csv
"""
import csv
import os
import random
import math

os.makedirs("sample_data", exist_ok=True)
path = os.path.join("sample_data", "dirty_signal.csv")

random.seed(42)
rows = []
t = 0.0
dt = 0.01

for i in range(200):
    # Pure sine + small noise
    value = 1.0 * math.sin(2 * math.pi * 2.0 * t) + random.uniform(-0.05, 0.05)

    # Inject a spike at samples 40 and 120
    if i == 40:
        value = 99.99
    if i == 120:
        value = -45.00

    # Inject garbage text at sample 60
    if i == 60:
        rows.append([f"{t:.4f}", "abc"])
        t += dt
        continue

    # Inject a missing value at sample 80
    if i == 80:
        rows.append([f"{t:.4f}", ""])
        t += dt
        continue

    rows.append([f"{t:.4f}", f"{value:.4f}"])
    t += dt

# Inject a duplicate timestamp (re-log of sample 100)
dup_time = f"{100*dt:.4f}"
dup_value = rows[100][1]
rows.insert(101, [dup_time, dup_value])

with open(path, "w", newline="") as f:
    w = csv.writer(f)                 # csv.writer handles quoting/escaping correctly
    w.writerow(["time", "value"])
    w.writerows(rows)

print(f"Wrote {len(rows)} data rows to {path}")