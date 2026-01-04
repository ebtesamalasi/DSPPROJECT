import sys
import numpy as np
import pyqtgraph as pg
from pyqtgraph.Qt import QtCore, QtWidgets

# ---------------- DSP helpers ----------------
def bits_to_bpsk(bits: np.ndarray) -> np.ndarray:
    return 2 * bits - 1

def awgn(x: np.ndarray, snr_db: float, rng: np.random.Generator) -> np.ndarray:
    p_sig = np.mean(x**2)
    snr_lin = 10 ** (snr_db / 10.0)
    p_noise = p_sig / snr_lin
    noise = rng.normal(0.0, np.sqrt(p_noise), size=x.shape)
    return x + noise

def upsample_repeat(x: np.ndarray, sps: int) -> np.ndarray:
    return np.repeat(x, sps)

def generate_dsss_stream(n_bits, chips_per_bit, fs, chip_rate, rng):
    if fs % chip_rate != 0:
        raise ValueError("Choose fs divisible by chip_rate for this simple implementation.")
    sps = fs // chip_rate
    if sps < 1:
        raise ValueError("samples_per_chip < 1. Increase fs or decrease chip_rate.")

    bits = rng.integers(0, 2, size=n_bits)
    bpsk = bits_to_bpsk(bits)
    pn = rng.choice([-1, 1], size=chips_per_bit)

    chips = np.concatenate([bpsk[i] * pn for i in range(n_bits)])
    tx = upsample_repeat(chips, sps).astype(np.float32)
    return bits, pn, tx, sps

def add_channel(tx, fs, snr_db, jammer_amp, jammer_freq, rng):
    t = np.arange(len(tx)) / fs
    jammer = jammer_amp * np.sin(2 * np.pi * jammer_freq * t).astype(np.float32)
    rx = tx + jammer
    rx = awgn(rx, snr_db, rng).astype(np.float32)
    return rx

# ---------------- GUI ----------------
pg.setConfigOptions(antialias=True)

class DSSSRealtime(QtWidgets.QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("DSSS Real-Time: Oscilloscope + Spectrum + Correlation")
        self.resize(1100, 850)

        # --- Parameters ---
        self.rng = np.random.default_rng(0)

        self.fs = 48000
        self.chip_rate = 6000          # must divide fs
        self.chips_per_bit = 31
        self.n_bits = 600

        self.snr_db = -6
        self.jammer_amp = 1.2
        self.jammer_freq = 2000

        # display window
        self.window_sec = 0.01
        self.Nw = int(self.fs * self.window_sec)

        self.sps = self.fs // self.chip_rate
        self.bit_len = self.chips_per_bit * self.sps
        self.nfft = 4096

        # --- Generate stream ---
        self.bits, self.pn, tx, self.sps = generate_dsss_stream(
            self.n_bits, self.chips_per_bit, self.fs, self.chip_rate, self.rng
        )
        self.rx = add_channel(tx, self.fs, self.snr_db, self.jammer_amp, self.jammer_freq, self.rng)
        self.pn_samples = upsample_repeat(self.pn, self.sps).astype(np.float32)

        # --- UI ---
        cw = QtWidgets.QWidget()
        self.setCentralWidget(cw)
        layout = QtWidgets.QVBoxLayout(cw)

        # Use GraphicsLayoutWidget for multiple plots
        self.glw = pg.GraphicsLayoutWidget()
        layout.addWidget(self.glw)

        # Set background HERE (this fixes your error)
        self.bg = (5, 6, 7)
        self.glw.setBackground(self.bg)

        # Colors
        self.trace_color = (0, 255, 110)   # green
        self.spec_color  = (0, 213, 255)   # cyan
        self.corr_color  = (255, 208, 0)   # yellow

        # --- Plot 1: Time (Oscilloscope) ---
        self.p_time = self.glw.addPlot(row=0, col=0, title="OSCILLOSCOPE (Time Domain)")
        self.p_time.showGrid(x=True, y=True, alpha=0.25)
        self.p_time.setLabel("bottom", "Time", units="ms")
        self.p_time.setLabel("left", "Amplitude")
        self.p_time.setYRange(-3, 3)

        self.t_ms = (np.arange(self.Nw) / self.fs) * 1000.0
        self.curve_time = self.p_time.plot(self.t_ms, np.zeros(self.Nw),
                                           pen=pg.mkPen(self.trace_color, width=2))

        # Persistence (store history arrays ourselves, no .yData dependency)
        self.persistence_len = 10
        self.persist_buffers = [np.zeros(self.Nw, dtype=np.float32) for _ in range(self.persistence_len)]
        self.persistence_curves = []
        for i in range(self.persistence_len):
            alpha = int(140 * (1 - i / self.persistence_len))
            pen = pg.mkPen((self.trace_color[0], self.trace_color[1], self.trace_color[2], alpha), width=1)
            self.persistence_curves.append(self.p_time.plot(self.t_ms, self.persist_buffers[i], pen=pen))

        # --- Plot 2: Spectrum ---
        self.p_spec = self.glw.addPlot(row=1, col=0, title="SPECTRUM ANALYZER (FFT of Current Window)")
        self.p_spec.showGrid(x=True, y=True, alpha=0.25)
        self.p_spec.setLabel("bottom", "Frequency", units="Hz")
        self.p_spec.setLabel("left", "Magnitude", units="dB")
        self.p_spec.setXRange(0, self.fs / 2)

        self.f = np.fft.rfftfreq(self.nfft, 1 / self.fs)
        self.curve_spec = self.p_spec.plot(self.f, np.zeros_like(self.f),
                                           pen=pg.mkPen(self.spec_color, width=2))

        # --- Plot 3: Correlation ---
        self.p_corr = self.glw.addPlot(row=2, col=0, title="CORRELATION DETECTOR (per-bit windows)")
        self.p_corr.showGrid(x=True, y=True, alpha=0.25)
        self.p_corr.setLabel("bottom", "Bit Window Index")
        self.p_corr.setLabel("left", "Correlation")

        self.corr_hist = []
        self.corr_x = []
        self.curve_corr = self.p_corr.plot([], [],
                                           pen=pg.mkPen(self.corr_color, width=2),
                                           symbol='o', symbolSize=5)

        # Status text
        self.status = QtWidgets.QLabel()
        self.status.setStyleSheet("color: white; font-size: 12px;")
        layout.addWidget(self.status)

        # Streaming pointers
        self.ptr = 0
        self.bit_ptr = 0

        # Timer
        self.timer = QtCore.QTimer()
        self.timer.timeout.connect(self.update_live)
        self.timer.start(30)

    def update_live(self):
        # Loop when end reached
        if self.ptr + self.Nw >= len(self.rx) or (self.bit_ptr + 1) * self.bit_len >= len(self.rx):
            self.ptr = 0
            self.bit_ptr = 0
            self.corr_hist.clear()
            self.corr_x.clear()

        # Current window
        win = self.rx[self.ptr:self.ptr + self.Nw]

        # Update persistence buffers (shift)
        self.persist_buffers = [win.astype(np.float32)] + self.persist_buffers[:-1]
        for i in range(self.persistence_len):
            self.persistence_curves[i].setData(self.t_ms, self.persist_buffers[i])

        # Main trace
        self.curve_time.setData(self.t_ms, win)

        # FFT spectrum
        n = min(len(win), self.nfft)
        x = win[:n]
        w = np.hanning(n)
        X = np.fft.rfft(x * w, n=self.nfft)
        mag_db = 20 * np.log10(np.abs(X) + 1e-12)
        self.curve_spec.setData(self.f, mag_db)

        # Correlation per bit window
        seg = self.rx[self.bit_ptr * self.bit_len:(self.bit_ptr + 1) * self.bit_len]
        corr = float(np.sum(seg * self.pn_samples))
        self.corr_hist.append(corr)
        self.corr_x.append(self.bit_ptr)

        # Plot last 80 correlation values
        last = 80
        x_plot = self.corr_x[-last:]
        y_plot = self.corr_hist[-last:]
        self.curve_corr.setData(x_plot, y_plot)

        # Detection flag (simple dynamic threshold)
        recent_max = max(1.0, np.max(np.abs(y_plot)))
        detected = abs(corr) > 0.6 * recent_max

        self.status.setText(
            f"Fs={self.fs}Hz | chip_rate={self.chip_rate} | sps={self.sps} | "
            f"SNR={self.snr_db}dB | Jammer={self.jammer_freq}Hz | "
            f"Corr={corr:.1f} | Detected={'YES' if detected else 'NO'}"
        )

        # Advance
        self.bit_ptr += 1
        self.ptr += self.Nw


if __name__ == "__main__":
    app = QtWidgets.QApplication(sys.argv)
    w = DSSSRealtime()
    w.show()
    sys.exit(app.exec())
