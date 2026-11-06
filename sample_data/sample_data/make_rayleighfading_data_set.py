"""
Generates Rayleigh fading datasets for teaching.

Rayleigh fading models a wireless channel with no line-of-sight path
(e.g., mobile phone in a city). The envelope follows a Rayleigh
distribution:

    p(r) = (r / σ²) · exp(-r² / (2σ²)),  r ≥ 0

Two independent Gaussian components (I and Q) are combined:

    envelope = sqrt(I² + Q²)

This creates a signal that is:
  - Always positive (envelope)
  - Heavily skewed (long right tail)
  - Has natural deep fades (near-zero values)
  - Contains legitimate peaks that look like outliers but aren't
"""
import csv
import os
import math
import random

OUT = "sample_data"
os.makedirs(OUT, exist_ok=True)


def write_csv(name, header, rows):
    path = os.path.join(OUT, name)
    with open(path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(header)
        w.writerows(rows)
    print(f"✅ {path}  ({len(rows)} rows)")


def rayleigh_envelope(n, sigma=1.0, fs=1000.0, doppler_hz=50.0, seed=42):
    """
    Generate a correlated Rayleigh fading envelope.

    Uses a simple sum-of-sinusoids (Jakes-like) model for the
    complex channel, so the envelope is correlated in time
    (realistic fading, not white noise).

    n          : number of samples
    sigma      : scale parameter (RMS amplitude of I/Q components)
    fs         : sample rate (Hz)
    doppler_hz : max Doppler frequency (Hz) — controls fade rate
    """
    random.seed(seed)
    dt = 1.0 / fs
    t = [i * dt for i in range(n)]

    # Sum-of-sinusoids for I and Q (Jakes model, simplified)
    N_osc = 8
    alphas = [2 * math.pi * k / N_osc for k in range(N_osc)]
    betas = [random.uniform(0, 2 * math.pi) for _ in range(N_osc)]

    I = [0.0] * n
    Q = [0.0] * n
    for k in range(N_osc):
        omega = 2 * math.pi * doppler_hz * math.cos(alphas[k])
        for i in range(n):
            I[i] += math.cos(omega * t[i] + betas[k])
            Q[i] += math.sin(omega * t[i] + betas[k])

    # Normalize to unit RMS, then scale by sigma
    norm = math.sqrt(N_osc / 2)
    I = [sigma * x / norm for x in I]
    Q = [sigma * x / norm for x in Q]

    # Envelope
    env = [math.sqrt(I[i] ** 2 + Q[i] ** 2) for i in range(n)]
    return t, env


# ─────────────────────────────────────────────────────────────
# A. PURE Rayleigh envelope (reference — NO dirt)
# ─────────────────────────────────────────────────────────────
t, env = rayleigh_envelope(n=1000, sigma=1.0, fs=1000.0, doppler_hz=20.0)
rows = [[f"{ti:.4f}", f"{ei:.6f}"] for ti, ei in zip(t, env)]
write_csv("rayleigh_00_pure.csv", ["time", "value"], rows)


# ─────────────────────────────────────────────────────────────
# B. Rayleigh + SPIKE NOISE (AWGN + impulse noise)
#    This is the classic "wireless channel + interference" case
# ─────────────────────────────────────────────────────────────
random.seed(101)
t, env = rayleigh_envelope(n=1000, sigma=1.0, fs=1000.0, doppler_hz=20.0)
rows = []
for i, (ti, ei) in enumerate(zip(t, env)):
    # Additive white Gaussian noise (sigma_n = 0.15)
    noisy = ei + random.gauss(0, 0.15)
    noisy = max(0.0, noisy)  # envelope stays non-negative

    # Impulse noise: 2% of samples get a huge spike
    if random.random() < 0.02:
        noisy += random.uniform(20, 50)

    rows.append([f"{ti:.4f}", f"{noisy:.6f}"])
write_csv("rayleigh_01_noise_spikes.csv", ["time", "value"], rows)


# ─────────────────────────────────────────────────────────────
# C. Rayleigh + MISSING SAMPLES + DUPLICATES (dropped packets)
# ─────────────────────────────────────────────────────────────
random.seed(202)
t, env = rayleigh_envelope(n=800, sigma=1.0, fs=1000.0, doppler_hz=20.0)
rows = []
for i, (ti, ei) in enumerate(zip(t, env)):
    noisy = ei + random.gauss(0, 0.1)
    noisy = max(0.0, noisy)

    # 3% of packets dropped → empty value
    if random.random() < 0.03:
        rows.append([f"{ti:.4f}", ""])
        continue

    rows.append([f"{ti:.4f}", f"{noisy:.6f}"])

    # 1% of packets duplicated (network retransmit)
    if random.random() < 0.01:
        rows.append([f"{ti:.4f}", f"{noisy + random.gauss(0, 0.05):.6f}"])

write_csv("rayleigh_02_dropped_packets.csv", ["time", "value"], rows)


# ─────────────────────────────────────────────────────────────
# D. Rayleigh + DEEP FADES clipped to zero (receiver saturation)
#    Demonstrates why simple min/max filtering fails
# ─────────────────────────────────────────────────────────────
random.seed(303)
t, env = rayleigh_envelope(n=1000, sigma=1.0, fs=1000.0, doppler_hz=30.0)
rows = []
for ti, ei in zip(t, env):
    noisy = ei + random.gauss(0, 0.1)
    # Simulate receiver floor: any value below 0.02 → 0
    if noisy < 0.02:
        noisy = 0.0
    rows.append([f"{ti:.4f}", f"{noisy:.6f}"])
write_csv("rayleigh_03_receiver_floor.csv", ["time", "value"], rows)


# ─────────────────────────────────────────────────────────────
# E. "KITCHEN SINK" Rayleigh — every problem at once
# ─────────────────────────────────────────────────────────────
random.seed(404)
t, env = rayleigh_envelope(n=1000, sigma=1.0, fs=1000.0, doppler_hz=25.0)
rows = []
for i, (ti, ei) in enumerate(zip(t, env)):
    noisy = ei + random.gauss(0, 0.12)
    noisy = max(0.0, noisy)

    # Garbage text occasionally
    if random.random() < 0.01:
        rows.append([f"{ti:.4f}", random.choice(["ERR", "NaN", "--"])])
        continue

    # Missing packet
    if random.random() < 0.02:
        rows.append([f"{ti:.4f}", ""])
        continue

    # Impulse spike
    if random.random() < 0.02:
        noisy += random.uniform(20, 60)

    # Bad timestamp
    if random.random() < 0.005:
        rows.append(["bad_time", f"{noisy:.6f}"])
        continue

    rows.append([f"{ti:.4f}", f"{noisy:.6f}"])

# Duplicate a few times
for _ in range(10):
    idx = random.randint(0, len(rows) - 1)
    if len(rows[idx]) == 2 and rows[idx][1]:
        rows.insert(idx + 1, [rows[idx][0], f"{float(rows[idx][1]) + random.gauss(0, 0.05):.6f}"])

random.shuffle(rows)   # unsorted
write_csv("rayleigh_04_kitchen_sink.csv", ["time", "value"], rows)


print("\n🎉 Rayleigh teaching set generated in sample_data/")
print("   Load 'rayleigh_00_pure.csv' first as the reference.")