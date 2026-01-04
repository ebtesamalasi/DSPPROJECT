# metrics.py
import numpy as np
from tx_dsss import generate_dsss
from channel import channel_model
from rx_dsss import despread_and_demod

def ber(bits: np.ndarray, bits_hat: np.ndarray) -> float:
    n = min(len(bits), len(bits_hat))
    return float(np.mean(bits[:n] != bits_hat[:n]))

def ber_vs_snr(fs, chip_rate, chips_per_bit, n_bits, jammer_amp, jammer_freq, snr_list, rng):
    results = []
    for snr_db in snr_list:
        bits, pn, tx, sps = generate_dsss(n_bits, chips_per_bit, fs, chip_rate, rng)
        rx = channel_model(tx, fs, snr_db, jammer_amp, jammer_freq, rng)
        bits_hat, _ = despread_and_demod(rx, pn, sps, chips_per_bit, n_bits)
        results.append((snr_db, ber(bits, bits_hat)))
    return results
