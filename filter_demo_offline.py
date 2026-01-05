# filter_demo_offline.py
import numpy as np
import matplotlib.pyplot as plt
from scipy import signal
import config as cfg

def plot_spectrum(x, fs, title, filename):
    N = len(x)
    w = np.hanning(N)
    X = np.fft.rfft(x * w)
    f = np.fft.rfftfreq(N, 1/fs)
    mag = 20*np.log10(np.abs(X) + 1e-12)

    plt.figure()
    plt.plot(f, mag, linewidth=1.5)
    plt.xlabel("Frequency (Hz)")
    plt.ylabel("Magnitude (dB)")
    plt.title(title)
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(filename, dpi=200)

def anti_alias_fir(x, fs, cutoff_frac=0.45, numtaps=101):
    nyq = fs/2
    cutoff_hz = cutoff_frac * nyq
    taps = signal.firwin(numtaps, cutoff_hz, fs=fs)
    y = signal.lfilter(taps, [1.0], x)
    return y, taps

def anti_alias_iir(x, fs, cutoff_frac=0.45, order=6):
    nyq = fs/2
    cutoff_hz = cutoff_frac * nyq
    b, a = signal.butter(order, cutoff_hz, btype="low", fs=fs)
    y = signal.lfilter(b, a, x)
    return y, (b, a)

def main():
    fs = cfg.FS
    N = 4096
    t = np.arange(N) / fs

    # Create a tone ABOVE Nyquist to demonstrate aliasing
    f0 = 0.65 * fs   # > fs/2 => aliases
    x = np.sin(2*np.pi*f0*t).astype(np.float32)

    # 1) Spectrum without anti-alias filter
    plot_spectrum(x, fs, f"Aliasing (No Filter): tone f0={f0:.1f} Hz, fs={fs}", "alias_no_filter.png")

    # 2) FIR filter
    y_fir, taps = anti_alias_fir(x, fs, cutoff_frac=0.45, numtaps=101)
    plot_spectrum(y_fir, fs, "After FIR Anti-Alias LPF (firwin, 101 taps)", "alias_after_fir.png")

    # 3) IIR filter
    y_iir, (b, a) = anti_alias_iir(x, fs, cutoff_frac=0.45, order=6)
    plot_spectrum(y_iir, fs, "After IIR Anti-Alias LPF (Butterworth, order=6)", "alias_after_iir.png")

    # 4) Plot FIR magnitude response (for report)
    w, h = signal.freqz(taps, worN=2048, fs=fs)
    plt.figure()
    plt.plot(w, 20*np.log10(np.abs(h)+1e-12))
    plt.xlabel("Frequency (Hz)")
    plt.ylabel("Magnitude (dB)")
    plt.title("FIR Anti-Alias Filter Frequency Response")
    plt.grid(True)
    plt.tight_layout()
    plt.savefig("fir_response.png", dpi=200)

    # 5) Plot IIR magnitude response (for report)
    w2, h2 = signal.freqz(b, a, worN=2048, fs=fs)
    plt.figure()
    plt.plot(w2, 20*np.log10(np.abs(h2)+1e-12))
    plt.xlabel("Frequency (Hz)")
    plt.ylabel("Magnitude (dB)")
    plt.title("IIR Anti-Alias Filter Frequency Response")
    plt.grid(True)
    plt.tight_layout()
    plt.savefig("iir_response.png", dpi=200)

    plt.show()

if __name__ == "__main__":
    main()
