# validate_rf_dataset.py
"""
Validate an external recorded RF IQ dataset and demonstrate anti-alias filtering.

What this script does:
1) Loads an IQ WAV file (typically stereo: I on channel 0, Q on channel 1).
2) Converts it to complex IQ (I + jQ) and normalizes to [-1, 1].
3) Plots:
   - Time-domain snippet (I & Q)
   - Spectrum BEFORE filtering
   - Spectrum AFTER FIR anti-alias LPF
   - Spectrum AFTER IIR anti-alias LPF
4) (Optional) Plots FIR/IIR filter frequency responses if the helpers return them.

Requirements:
- numpy, matplotlib, scipy
- filters.py in the same folder, containing:
    anti_alias_fir(x, fs, cutoff_frac=..., numtaps=...) -> (y, h_or_info)
    anti_alias_iir(x, fs, cutoff_frac=..., order=...)   -> (y, sos_or_info)
"""

import os
import numpy as np
import matplotlib.pyplot as plt
from scipy.io import wavfile

# Your project filters (must exist in dsppro folder)
from filters import anti_alias_fir, anti_alias_iir


# ----------------------------
# Helpers
# ----------------------------
def normalize_wav(x: np.ndarray) -> np.ndarray:
    """
    Convert integer WAV to float32 in [-1, 1].
    If already float, just cast to float32.
    """
    x = np.asarray(x)
    if np.issubdtype(x.dtype, np.integer):
        # Use dtype max to normalize (e.g., int16 -> 32767)
        maxv = float(np.iinfo(x.dtype).max)
        return (x.astype(np.float32) / maxv).astype(np.float32)
    return x.astype(np.float32)


def to_complex_iq(wav_data: np.ndarray) -> np.ndarray:
    """
    Convert WAV samples to complex IQ.
    Most IQ WAV files store I in channel 0 and Q in channel 1 (stereo).
    """
    wav_data = np.asarray(wav_data)

    # Stereo (I/Q)
    if wav_data.ndim == 2 and wav_data.shape[1] >= 2:
        i = normalize_wav(wav_data[:, 0])
        q = normalize_wav(wav_data[:, 1])
        return (i + 1j * q).astype(np.complex64)

    # Mono: treat as real signal
    x = normalize_wav(wav_data)
    return x.astype(np.float32)


def plot_time_snippet(iq: np.ndarray, fs: int, title: str, n: int = 5000):
    """
    Plot a short time-domain snippet.
    If complex: plot I and Q separately.
    """
    n = min(len(iq), n)
    t = np.arange(n) / fs

    plt.figure()
    if np.iscomplexobj(iq):
        plt.plot(t, np.real(iq[:n]), linewidth=1.0, label="I (real)")
        plt.plot(t, np.imag(iq[:n]), linewidth=1.0, label="Q (imag)")
        plt.legend()
    else:
        plt.plot(t, iq[:n], linewidth=1.0)
    plt.xlabel("Time (s)")
    plt.ylabel("Amplitude")
    plt.title(title)
    plt.grid(True)


def plot_spectrum(x: np.ndarray, fs: int, title: str, nfft: int = 16384):
    """
    Plot magnitude spectrum in dB.
    Handles both real and complex signals.
    """
    x = np.asarray(x)
    n = min(len(x), nfft)
    x = x[:n]

    # Window
    w = np.hanning(n).astype(np.float32)

    if np.iscomplexobj(x):
        # Complex FFT
        X = np.fft.fft(x * w, n=nfft)
        f = np.fft.fftfreq(nfft, 1 / fs)
        # Keep only positive frequencies
        idx = f >= 0
        f = f[idx]
        X = X[idx]
    else:
        # Real FFT
        x = x.astype(np.float32)
        X = np.fft.rfft(x * w, n=nfft)
        f = np.fft.rfftfreq(nfft, 1 / fs)

    mag_db = 20 * np.log10(np.abs(X) + 1e-12)

    plt.figure()
    plt.plot(f, mag_db, linewidth=1.2)
    plt.xlabel("Frequency (Hz)")
    plt.ylabel("Magnitude (dB)")
    plt.title(title)
    plt.grid(True)


def try_plot_fir_response(fir_info, fs: int, title: str):
    """
    If anti_alias_fir returns taps (h), plot its magnitude response.
    If it returns something else, we skip silently.
    """
    if fir_info is None:
        return
    h = np.asarray(fir_info)
    if h.ndim != 1 or h.size < 4:
        return

    nfft = 16384
    H = np.fft.rfft(h, n=nfft)
    f = np.fft.rfftfreq(nfft, 1 / fs)
    mag_db = 20 * np.log10(np.abs(H) + 1e-12)

    plt.figure()
    plt.plot(f, mag_db, linewidth=1.2)
    plt.xlabel("Frequency (Hz)")
    plt.ylabel("Magnitude (dB)")
    plt.title(title)
    plt.grid(True)


def try_plot_iir_response(iir_info, fs: int, title: str):
    """
    If anti_alias_iir returns SOS array, plot frequency response.
    If it returns something else, skip silently.
    """
    if iir_info is None:
        return

    # Common pattern: return second-order sections (sos)
    sos = np.asarray(iir_info)
    if sos.ndim != 2 or sos.shape[1] != 6:
        return

    # Frequency response by evaluating on unit circle (manual, no scipy.signal needed)
    nfft = 16384
    w = 2 * np.pi * np.linspace(0, 0.5, nfft)  # 0..pi (normalized)
    z = np.exp(1j * w)

    H = np.ones_like(z, dtype=np.complex128)
    # SOS: [b0 b1 b2 a0 a1 a2]
    for сек in sos:
        b0, b1, b2, a0, a1, a2 = сек
        num = b0 + b1 / z + b2 / (z ** 2)
        den = a0 + a1 / z + a2 / (z ** 2)
        H *= num / den

    f = (w / (2 * np.pi)) * fs  # Hz
    mag_db = 20 * np.log10(np.abs(H) + 1e-12)

    plt.figure()
    plt.plot(f, mag_db, linewidth=1.2)
    plt.xlabel("Frequency (Hz)")
    plt.ylabel("Magnitude (dB)")
    plt.title(title)
    plt.grid(True)


# ----------------------------
# Main
# ----------------------------
def main():
    
    wav_path = r"C:\Users\USER\Desktop\dsppro\data\SDRuno_20200907_184926Z_161985kHz.wav"

    if not os.path.exists(wav_path):
        raise FileNotFoundError(
            f"File not found:\n{wav_path}\n\n"
            "Fix: Edit wav_path in validate_rf_dataset.py to point to your downloaded IQ WAV."
        )

    fs, wav_data = wavfile.read(wav_path)

    # Convert to complex IQ (or real if mono)
    iq = to_complex_iq(wav_data)

    # Print quick info
    print("Loaded:", wav_path)
    print("fs:", fs)
    print("wav_data shape:", getattr(wav_data, "shape", None), "dtype:", wav_data.dtype)
    print("iq type:", "complex" if np.iscomplexobj(iq) else "real",
          "shape:", iq.shape, "dtype:", iq.dtype)

    # Plot time snippet
    plot_time_snippet(iq, fs, "Recorded RF IQ Data (Time Domain) - Raw")

    # Spectrum before filtering
    plot_spectrum(iq, fs, "Recorded RF IQ Data (Spectrum) - No Filter")

    # Apply FIR anti-alias filter
    iq_fir, fir_info = anti_alias_fir(iq, fs, cutoff_frac=0.45, numtaps=101)
    plot_spectrum(iq_fir, fs, "After FIR Anti-Alias LPF (firwin, 101 taps)")
    try_plot_fir_response(fir_info, fs, "FIR Anti-Alias Filter Frequency Response")

    # Apply IIR anti-alias filter
    iq_iir, iir_info = anti_alias_iir(iq, fs, cutoff_frac=0.45, order=6)
    plot_spectrum(iq_iir, fs, "After IIR Anti-Alias LPF (Butterworth, order=6)")
    try_plot_iir_response(iir_info, fs, "IIR Anti-Alias Filter Frequency Response")

    plt.show()


if __name__ == "__main__":
    main()
