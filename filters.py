import numpy as np
from scipy.signal import firwin, lfilter, butter, sosfilt

def anti_alias_fir(x, fs, cutoff_frac=0.45, numtaps=101):
    """
    FIR low-pass anti-alias filter.
    cutoff_frac is fraction of Nyquist (0..1). Example: 0.45 means 0.45*(fs/2).
    """
    cutoff_hz = cutoff_frac * (fs / 2.0)
    taps = firwin(numtaps, cutoff_hz, fs=fs)
    y = lfilter(taps, [1.0], x)
    return y, taps

def anti_alias_iir(x, fs, cutoff_frac=0.45, order=6):
    """
    IIR low-pass anti-alias filter (Butterworth).
    """
    cutoff_hz = cutoff_frac * (fs / 2.0)
    sos = butter(order, cutoff_hz, btype="low", fs=fs, output="sos")
    y = sosfilt(sos, x)
    return y, sos
