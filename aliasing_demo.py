# aliasing_demo.py
import numpy as np
import matplotlib.pyplot as plt
from tx_dsss import generate_dsss
from channel import channel_model
from rx_dsss import despread_and_demod
from metrics import ber

def run_aliasing_demo(rng):
    # Good vs bad sampling example
    n_bits = 300
    chips_per_bit = 31
    chip_rate = 8000
    snr_db = -6
    jammer_amp = 1.2
    jammer_freq = 3500

    fs_good = 48000  # divisible by 8000
    fs_bad  = 16000  # divisible by 8000, but very low

    # GOOD
    bits_g, pn_g, tx_g, sps_g = generate_dsss(n_bits, chips_per_bit, fs_good, chip_rate, rng)
    rx_g = channel_model(tx_g, fs_good, snr_db, jammer_amp, jammer_freq, rng)
    hat_g, corr_g = despread_and_demod(rx_g, pn_g, sps_g, chips_per_bit, n_bits)
    ber_g = ber(bits_g, hat_g)

    # BAD
    bits_b, pn_b, tx_b, sps_b = generate_dsss(n_bits, chips_per_bit, fs_bad, chip_rate, rng)
    rx_b = channel_model(tx_b, fs_bad, snr_db, jammer_amp, jammer_freq, rng)
    hat_b, corr_b = despread_and_demod(rx_b, pn_b, sps_b, chips_per_bit, n_bits)
    ber_b = ber(bits_b, hat_b)

    plt.figure(figsize=(10,4))
    plt.plot(corr_g[:60], label=f"Good Fs={fs_good} (BER={ber_g:.3f})")
    plt.plot(corr_b[:60], label=f"Bad  Fs={fs_bad}  (BER={ber_b:.3f})")
    plt.title("Aliasing Effect (Correlation Metric)")
    plt.xlabel("Bit window index")
    plt.ylabel("Correlation")
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.show()
