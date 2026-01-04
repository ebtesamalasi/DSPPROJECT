# rx_dsss.py
import numpy as np

def upsample_repeat(x: np.ndarray, sps: int) -> np.ndarray:
    return np.repeat(x, sps)

def correlation_metric(rx_seg: np.ndarray, pn_samples: np.ndarray) -> float:
    return float(np.sum(rx_seg * pn_samples))

def despread_and_demod(rx: np.ndarray, pn: np.ndarray, sps: int, chips_per_bit: int, n_bits: int):
    bit_len = chips_per_bit * sps
    pn_samples = upsample_repeat(pn, sps).astype(np.float32)

    bits_hat = np.zeros(n_bits, dtype=int)
    corr_vals = np.zeros(n_bits, dtype=float)

    for i in range(n_bits):
        seg = rx[i*bit_len:(i+1)*bit_len]
        if len(seg) < bit_len:
            bits_hat = bits_hat[:i]
            corr_vals = corr_vals[:i]
            break

        corr_vals[i] = correlation_metric(seg, pn_samples)

        # despread then integrate-and-dump
        despread = seg * pn_samples
        metric = float(np.sum(despread))
        bits_hat[i] = 1 if metric > 0 else 0

    return bits_hat, corr_vals
