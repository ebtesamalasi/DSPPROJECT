#channel.py
import numpy as np

def awgn(x: np.ndarray, snr_db: float, rng: np.random.Generator) -> np.ndarray:
    p_sig = float(np.mean(x**2))
    snr_lin = 10 ** (snr_db / 10.0)
    p_noise = p_sig / snr_lin
    noise = rng.normal(0.0, np.sqrt(p_noise), size=x.shape)
    return (x + noise).astype(np.float32)

def add_jammer(x: np.ndarray, fs: int, jammer_amp: float, jammer_freq: float) -> np.ndarray:
    t = np.arange(len(x)) / fs
    jammer = jammer_amp * np.sin(2*np.pi*jammer_freq*t)
    return (x + jammer.astype(np.float32)).astype(np.float32)

def channel_model(tx: np.ndarray, fs: int, snr_db: float, jammer_amp: float, jammer_freq: float, rng: np.random.Generator):
    rx = add_jammer(tx, fs, jammer_amp, jammer_freq)
    rx = awgn(rx, snr_db, rng)
    return rx
