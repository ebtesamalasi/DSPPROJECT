# tx_dsss.py
import numpy as np

def bits_to_bpsk(bits: np.ndarray) -> np.ndarray:
    return 2 * bits - 1

def upsample_repeat(x: np.ndarray, sps: int) -> np.ndarray:
    return np.repeat(x, sps)

def generate_dsss(n_bits: int, chips_per_bit: int, fs: int, chip_rate: int, rng: np.random.Generator):
    if fs % chip_rate != 0:
        raise ValueError("Choose fs divisible by chip_rate.")
    sps = fs // chip_rate
    if sps < 1:
        raise ValueError("samples_per_chip < 1, increase fs or decrease chip_rate.")

    bits = rng.integers(0, 2, size=n_bits)
    bpsk = bits_to_bpsk(bits)

    pn = rng.choice([-1, 1], size=chips_per_bit)   # +1/-1 PN
    chips = np.concatenate([bpsk[i] * pn for i in range(n_bits)])
    tx = upsample_repeat(chips, sps).astype(np.float32)

    return bits.astype(int), pn.astype(np.float32), tx, sps
