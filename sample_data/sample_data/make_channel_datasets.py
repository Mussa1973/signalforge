"""
Realistic Rayleigh & Rician channel datasets for teaching.

Uses the Sum-of-Sinusoids (SoS) method (Jakes 1974, Zheng & Xiao 2003)
to generate correlated fading with correct Doppler spectrum.

Physical scenarios:
  - Pedestrian   : 5 km/h   →  fd ≈ 8 Hz   (walking)
  - Urban car    : 30 km/h  →  fd ≈ 50 Hz  (city driving)
  - Highway car  : 100 km/h →  fd ≈ 167 Hz (motorway)

Rayleigh : no LoS (NLOS urban canyon)
Rician   : LoS + scatter (rural / open field)

Outputs CSVs compatible with SignalForge.
"""
import csv
import os
import math
import random

OUT = "sample_data"
os.makedirs(OUT, exist_ok=True)

# Physical constants
C = 3e8                  # speed of light (m/s)
FC = 2.0e9               # carrier frequency (2 GHz, typical LTE/5G)
FS = 1000.0              # sample rate (Hz)
N_OSC = 20               # number of sinusoids (Zheng-Xiao: 8+ is enough)


def write_csv(name, header, rows):
    path = os.path.join(OUT, name)
    with open(path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(header)
        w.writerows(rows)
    print(f"✅ {path}  ({len(rows)} rows)")


def doppler_frequency(speed_kmh: float, fc_hz: float = FC) -> float:
    """Max Doppler shift: fd = v·fc/c"""
    v = speed_kmh / 3.6
    return v * fc_hz / C


def sum_of_sinusoids(n: int, fd_hz: float, fs_hz: float,
                     seed: int = 42) -> tuple[list, list]:
    """
    Zheng-Xiao Sum-of-Sinusoids generator for complex Gaussian fading.

    Returns:
        I : in-phase component  (Gaussian, correlated)
        Q : quadrature component (Gaussian, correlated)

    The resulting envelope sqrt(I² + Q²) follows a Rayleigh distribution
    with the correct Doppler spectrum.
    """
    random.seed(seed)
    dt = 1.0 / fs_hz
    t = [i * dt for i in range(n)]

    # Zheng-Xiao parameters
    alpha = [(2 * math.pi * k - math.pi + math.pi / (4 * N_OSC)) / (4 * N_OSC)
             for k in range(1, N_OSC + 1)]
    phi = [random.uniform(-math.pi, math.pi) for _ in range(N_OSC)]
    psi = [random.uniform(-math.pi, math.pi) for _ in range(N_OSC)]

    I = [0.0] * n
    Q = [0.0] * n

    for k in range(N_OSC):
        omega = 2 * math.pi * fd_hz * math.cos(alpha[k])
        for i in range(n):
            I[i] += math.cos(omega * t[i] + phi[k])
            Q[i] += math.sin(omega * t[i] + psi[k])

    # Normalize to unit power (so E[I²] = E[Q²] = 0.5)
    scale = math.sqrt(1.0 / N_OSC)
    I = [x * scale for x in I]
    Q = [x * scale for x in Q]
    return I, Q


def awgn_noise(n: int, snr_db: float, signal_power: float = 1.0,
               seed: int = 123) -> tuple[list, list]:
    """Complex AWGN matched to a given SNR."""
    random.seed(seed)
    noise_power = signal_power / (10 ** (snr_db / 10.0))
    sigma = math.sqrt(noise_power / 2.0)
    nI = [random.gauss(0, sigma) for _ in range(n)]
    nQ = [random.gauss(0, sigma) for _ in range(n)]
    return nI, nQ


# ─────────────────────────────────────────────────────────────
# RAYLEIGH CHANNEL  (no LoS, classic NLOS urban)
# ─────────────────────────────────────────────────────────────
def make_rayleigh(n: int, speed_kmh: float, snr_db: float,
                  seed: int, name: str):
    """Rayleigh fading with AWGN at specified SNR."""
    fd = doppler_frequency(speed_kmh)
    I, Q = sum_of_sinusoids(n, fd, FS, seed=seed)
    nI, nQ = awgn_noise(n, snr_db, signal_power=1.0, seed=seed + 1)

    dt = 1.0 / FS
    rows = []
    for i in range(n):
        t = i * dt
        # Received complex baseband: h·s + n, with |s|=1
        rI = I[i] + nI[i]
        rQ = Q[i] + nQ[i]
        # Envelope (this is what the receiver measures)
        env = math.sqrt(rI ** 2 + rQ ** 2)
        # Phase (in degrees)
        phase = math.degrees(math.atan2(rQ, rI))
        rows.append([f"{t:.6f}", f"{env:.6f}", f"{phase:.4f}"])

    write_csv(name, ["time", "envelope", "phase_deg"], rows)


# ─────────────────────────────────────────────────────────────
# RICIAN CHANNEL  (LoS + scatter, controlled by K-factor)
# ─────────────────────────────────────────────────────────────
def make_rician(n: int, speed_kmh: float, K_db: float, snr_db: float,
                seed: int, name: str):
    """
    Rician fading: LoS + scattered multipath.

    K = LoS power / scattered power
    A = LoS amplitude = sqrt(K / (K+1))
    sigma² = scattered power per component = 1 / (2(K+1))
    """
    K = 10 ** (K_db / 10.0)
    A = math.sqrt(K / (K + 1.0))                 # LoS amplitude
    sigma_sc = math.sqrt(1.0 / (2.0 * (K + 1.0)))

    fd = doppler_frequency(speed_kmh)
    I, Q = sum_of_sinusoids(n, fd, FS, seed=seed)
    I = [x * sigma_sc / math.sqrt(0.5) for x in I]   # scale to sigma_sc
    Q = [x * sigma_sc / math.sqrt(0.5) for x in Q]

    nI, nQ = awgn_noise(n, snr_db, signal_power=1.0, seed=seed + 1)

    dt = 1.0 / FS
    rows = []
    for i in range(n):
        t = i * dt
        rI = A + I[i] + nI[i]
        rQ = Q[i] + nQ[i]
        env = math.sqrt(rI ** 2 + rQ ** 2)
        phase = math.degrees(math.atan2(rQ, rI))
        rows.append([f"{t:.6f}", f"{env:.6f}", f"{phase:.4f}"])

    write_csv(name, ["time", "envelope", "phase_deg"], rows)


# ─────────────────────────────────────────────────────────────
# GENERATE ALL SCENARIOS
# ─────────────────────────────────────────────────────────────
N = 2000   # 2 seconds at 1 kHz

# --- Rayleigh ---
make_rayleigh(N, speed_kmh=5,   snr_db=20, seed=101,
              name="chan_rayleigh_pedestrian_5kmh.csv")
make_rayleigh(N, speed_kmh=30,  snr_db=20, seed=102,
              name="chan_rayleigh_urban_30kmh.csv")
make_rayleigh(N, speed_kmh=100, snr_db=20, seed=103,
              name="chan_rayleigh_highway_100kmh.csv")
make_rayleigh(N, speed_kmh=30,  snr_db=5,  seed=104,
              name="chan_rayleigh_lowsnr_5dB.csv")

# --- Rician (same speed, varying K) ---
make_rician(N, speed_kmh=30, K_db=0,   snr_db=20, seed=201,
            name="chan_rician_K0dB.csv")     # ≈ Rayleigh
make_rician(N, speed_kmh=30, K_db=6,   snr_db=20, seed=202,
            name="chan_rician_K6dB.csv")
make_rician(N, speed_kmh=30, K_db=10,  snr_db=20, seed=203,
            name="chan_rician_K10dB.csv")
make_rician(N, speed_kmh=30, K_db=15,  snr_db=20, seed=204,
            name="chan_rician_K15dB.csv")    # LoS-dominant

# --- "Dirty" versions — add channel-measurement dirt on top ---
def add_dirt(src_name: str, dst_name: str, seed: int = 999):
    """Add spikes, gaps, garbage to a clean channel CSV."""
    random.seed(seed)
    rows = []
    with open(os.path.join(OUT, src_name)) as f:
        reader = csv.reader(f)
        header = next(reader)
        for i, row in enumerate(reader):
            t, env, ph = row

            # 2% impulse spikes
            if random.random() < 0.02:
                env = f"{float(env) + random.uniform(10, 30):.6f}"

            # 3% missing
            if random.random() < 0.03:
                env = ""

            # 1% garbage
            if random.random() < 0.01:
                env = random.choice(["ERR", "NaN", "--"])

            # 0.5% bad timestamps
            if random.random() < 0.005:
                t = "bad_time"

            rows.append([t, env, ph])

    write_csv(dst_name, header, rows)


add_dirt("chan_rayleigh_urban_30kmh.csv", "chan_rayleigh_urban_dirty.csv")
add_dirt("chan_rician_K10dB.csv",         "chan_rician_K10dB_dirty.csv")

print("\n🎉 Realistic Rayleigh & Rician channel datasets generated.")
print("   Clean files : chan_rayleigh_*.csv, chan_rician_*.csv")
print("   Dirty files : chan_*_dirty.csv")