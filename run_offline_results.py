# run_offline_results.py
import numpy as np
import matplotlib.pyplot as plt

import config as cfg
from tx_dsss import generate_dsss
from channel import channel_model
from rx_dsss import despread_and_demod


# ----------------------------
# Core runner (one simulation)
# ----------------------------
def run_one_case(snr_db, jammer_amp, chips_per_bit=None, chip_rate=None, fs=None, rng_seed=None, n_bits=None):
    if chips_per_bit is None:
        chips_per_bit = cfg.CHIPS_PER_BIT
    if chip_rate is None:
        chip_rate = cfg.CHIP_RATE
    if fs is None:
        fs = cfg.FS
    if rng_seed is None:
        rng_seed = cfg.RNG_SEED
    if n_bits is None:
        n_bits = cfg.N_BITS

    rng = np.random.default_rng(rng_seed)

    bits, pn, tx, sps = generate_dsss(n_bits, chips_per_bit, fs, chip_rate, rng)
    rx = channel_model(tx, fs, snr_db, jammer_amp, cfg.JAMMER_FREQ, rng)
    bits_hat, corr_vals = despread_and_demod(rx, pn, sps, chips_per_bit, n_bits)

    ber = float(np.mean(bits != bits_hat))
    return ber, corr_vals, bits, bits_hat, tx, rx, sps


# ----------------------------
# Plot helpers (performance)
# ----------------------------
def plot_corr_vs_bits(corr_vals, title):
    plt.figure()
    plt.plot(corr_vals, linewidth=1.5)
    plt.axhline(0, linestyle="--", linewidth=1)
    plt.xlabel("Bit index")
    plt.ylabel("Correlation output")
    plt.title(title)
    plt.grid(True)


def plot_corr_hist(corr_vals, bits, title):
    corr_0 = corr_vals[bits == 0]
    corr_1 = corr_vals[bits == 1]
    plt.figure()
    plt.hist(corr_0, bins=40, alpha=0.7, label="Bit = 0")
    plt.hist(corr_1, bins=40, alpha=0.7, label="Bit = 1")
    plt.axvline(0, linestyle="--", linewidth=1)
    plt.xlabel("Correlation output")
    plt.ylabel("Count")
    plt.title(title)
    plt.grid(True)
    plt.legend()


def plot_ber_vs_jammer_amp(snr_db, jammer_amps):
    bers = []
    for a in jammer_amps:
        ber, *_ = run_one_case(snr_db=snr_db, jammer_amp=a)
        bers.append(ber)

    plt.figure()
    plt.plot(jammer_amps, bers, marker="o", linewidth=1.5)
    plt.xlabel("Jammer amplitude")
    plt.ylabel("BER")
    plt.title(f"BER vs Jammer Amplitude (SNR={snr_db} dB)")
    plt.grid(True)


def plot_ber_vs_chips_per_bit(snr_db, jammer_amp, chips_list):
    bers = []
    for cpb in chips_list:
        ber, *_ = run_one_case(snr_db=snr_db, jammer_amp=jammer_amp, chips_per_bit=cpb)
        bers.append(ber)

    plt.figure()
    plt.plot(chips_list, bers, marker="o", linewidth=1.5)
    plt.xlabel("Chips per bit")
    plt.ylabel("BER")
    plt.title(f"Processing Gain: BER vs Chips/Bit (SNR={snr_db} dB, JammerAmp={jammer_amp})")
    plt.grid(True)


# ----------------------------
# Existing-style plot: BER vs SNR
# ----------------------------
def plot_ber_vs_snr(snr_list_db, jammer_amp=0.0):
    bers = []
    for snr_db in snr_list_db:
        ber, *_ = run_one_case(snr_db=snr_db, jammer_amp=jammer_amp)
        bers.append(ber)

    plt.figure()
    plt.semilogy(snr_list_db, np.maximum(bers, 1e-6), marker="o", linewidth=1.5)
    plt.xlabel("SNR (dB)")
    plt.ylabel("BER (log scale)")
    plt.title(f"BER vs SNR (JammerAmp={jammer_amp})")
    plt.grid(True, which="both")


# ----------------------------
# Aliasing demo (keep it)
# ----------------------------
def aliasing_demo():
    """
    Simple, clear aliasing demonstration:
    - generate a tone above Nyquist and show how it aliases after sampling.
    This is independent from DSSS and is good for the 'aliasing effect' requirement.
    """
    fs = cfg.FS
    N = 4096
    t = np.arange(N) / fs

    # choose a frequency above Nyquist to force aliasing
    f0 = 0.65 * fs  # above fs/2 => will alias
    x = np.sin(2 * np.pi * f0 * t)

    # spectrum
    X = np.fft.rfft(x * np.hanning(N))
    f = np.fft.rfftfreq(N, 1/fs)
    mag = 20 * np.log10(np.abs(X) + 1e-12)

    plt.figure()
    plt.plot(f, mag, linewidth=1.5)
    plt.xlabel("Frequency (Hz)")
    plt.ylabel("Magnitude (dB)")
    plt.title(f"Aliasing Demo: Input tone f0={f0:.1f} Hz (fs={fs} Hz)")
    plt.grid(True)


# ----------------------------
# MAIN
# ----------------------------
def main():
    # A) Keep your core BER vs SNR plot
    snr_list = np.array([-15, -12, -10, -8, -6, -5, -3, 0, 3, 6, 10], dtype=float)
    plot_ber_vs_snr(snr_list_db=snr_list, jammer_amp=0.0)

    # B) Keep aliasing effect plot
    aliasing_demo()

    # C) NEW: Correlation-based performance plots for your report scenarios
    scenarios = [
        ("Scenario 2 (Noise only)", -5, 0.0),
        ("Scenario 3 (Noise + Jammer)", -8, 1.0),
        ("Scenario 4 (Extreme)", -15, 2.5),
    ]

    for name, snr_db, jammer_amp in scenarios:
        ber, corr_vals, bits, bits_hat, *_ = run_one_case(snr_db, jammer_amp)
        plot_corr_vs_bits(corr_vals, f"{name}: Correlation vs Bit Index | BER={ber:.4f}")
        plot_corr_hist(corr_vals, bits, f"{name}: Correlation Histogram | BER={ber:.4f}")

    # D) NEW: BER vs Jammer amplitude (fixed SNR)
    jammer_amps = np.array([0, 0.5, 1.0, 1.5, 2.0, 2.5], dtype=float)
    plot_ber_vs_jammer_amp(snr_db=-8, jammer_amps=jammer_amps)

    # E) NEW: Processing gain effect (chips/bit)
    plot_ber_vs_chips_per_bit(snr_db=-8, jammer_amp=1.0, chips_list=[4, 8, 16])

    plt.show()


if __name__ == "__main__":
    main()



