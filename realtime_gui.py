# realtime_gui.py
import sys
import numpy as np
import pyqtgraph as pg
from pyqtgraph.Qt import QtCore, QtWidgets

from tx_dsss import generate_dsss
from channel import channel_model
from rx_dsss import despread_and_demod
from io_bits_plot import IOBitsPlot   # NEW

pg.setConfigOptions(antialias=True)


def format_bits_with_errors(
    tx_bits: str,
    rx_bits: str,
    tx_color: str = "#00D5FF",     # cyan
    rx_ok_color: str = "#00FF6A",  # green
    err_color: str = "#FF3B30"     # red
):
    """
    Returns (tx_html, rx_html, n_errors).
    RX bits are colored green if correct, red if wrong.
    """
    n = min(len(tx_bits), len(rx_bits))
    errors = 0
    rx_spans = []

    for i in range(n):
        if tx_bits[i] == rx_bits[i]:
            rx_spans.append(f"<span style='color:{rx_ok_color}'>{rx_bits[i]}</span>")
        else:
            rx_spans.append(f"<span style='color:{err_color}; font-weight:800'>{rx_bits[i]}</span>")
            errors += 1

    tx_html = f"<span style='color:{tx_color}; font-weight:800'>{tx_bits[:n]}</span>"
    rx_html = "".join(rx_spans)
    return tx_html, rx_html, errors


class RealtimeDSSSGUI(QtWidgets.QMainWindow):
    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg
        self.setWindowTitle("DSSS Real-Time Instrument View")
        self.resize(1150, 1000)

        self.rng = np.random.default_rng(cfg.RNG_SEED)

        # ---------------- Generate stream (TX -> Channel -> RX) ----------------
        self.bits, self.pn, tx, self.sps = generate_dsss(
            cfg.N_BITS, cfg.CHIPS_PER_BIT, cfg.FS, cfg.CHIP_RATE, self.rng
        )
        self.rx = channel_model(tx, cfg.FS, cfg.SNR_DB, cfg.JAMMER_AMP, cfg.JAMMER_FREQ, self.rng)

        # Receiver outputs (computed once then displayed live)
        self.bits_hat, self.corr_vals = despread_and_demod(
            self.rx, self.pn, self.sps, cfg.CHIPS_PER_BIT, cfg.N_BITS
        )

        self.Nw = int(cfg.FS * cfg.WINDOW_SEC)
        self.nfft = cfg.NFFT

        # ---------------- UI Layout ----------------
        cw = QtWidgets.QWidget()
        self.setCentralWidget(cw)
        layout = QtWidgets.QVBoxLayout(cw)

        self.glw = pg.GraphicsLayoutWidget()
        layout.addWidget(self.glw)

        self.glw.setBackground((5, 6, 7))

        # ---------------- Plot 1: Oscilloscope ----------------
        self.p_time = self.glw.addPlot(row=0, col=0, title="OSCILLOSCOPE (RX Time Domain)")
        self.p_time.showGrid(x=True, y=True, alpha=0.25)
        self.p_time.setLabel("bottom", "Time", units="ms")
        self.p_time.setLabel("left", "Amplitude")
        self.p_time.setYRange(-3, 3)

        self.t_ms = (np.arange(self.Nw) / cfg.FS) * 1000.0
        self.curve_time = self.p_time.plot(
            self.t_ms, np.zeros(self.Nw),
            pen=pg.mkPen((0, 255, 110), width=2)
        )

        # Persistence
        self.persist_len = cfg.PERSIST_LEN
        self.persist_buffers = [np.zeros(self.Nw, dtype=np.float32) for _ in range(self.persist_len)]
        self.persist_curves = []
        for i in range(self.persist_len):
            alpha = int(140 * (1 - i / self.persist_len))
            pen = pg.mkPen((0, 255, 110, alpha), width=1)
            self.persist_curves.append(self.p_time.plot(self.t_ms, self.persist_buffers[i], pen=pen))

        # ---------------- Plot 2: Spectrum Analyzer ----------------
        self.p_spec = self.glw.addPlot(row=1, col=0, title="SPECTRUM ANALYZER (FFT of RX Window)")
        self.p_spec.showGrid(x=True, y=True, alpha=0.25)
        self.p_spec.setLabel("bottom", "Frequency", units="Hz")
        self.p_spec.setLabel("left", "Magnitude", units="dB")
        self.p_spec.setXRange(0, cfg.FS / 2)

        self.f = np.fft.rfftfreq(self.nfft, 1 / cfg.FS)
        self.curve_spec = self.p_spec.plot(
            self.f, np.zeros_like(self.f),
            pen=pg.mkPen((0, 213, 255), width=2)
        )

        # ---------------- Plot 3: Correlation Detector ----------------
        self.p_corr = self.glw.addPlot(row=2, col=0, title="CORRELATION DETECTOR")
        self.p_corr.showGrid(x=True, y=True, alpha=0.25)
        self.p_corr.setLabel("bottom", "Bit window index")
        self.p_corr.setLabel("left", "Correlation")

        self.curve_corr = self.p_corr.plot(
            [], [],
            pen=pg.mkPen((255, 208, 0), width=2),
            symbol='o', symbolSize=5
        )

        # ---------------- Plot 4 (separate file): Digital IO ----------------
        self.io_plot = IOBitsPlot().add(self.glw, row=3, col=0)

        # ---------------- Status + Bits output ----------------
        self.status = QtWidgets.QLabel()
        self.status.setStyleSheet(
            "color:#EAEAEA; font-size:12px; font-family:Consolas; "
            "background-color:#050607; padding:6px; border:1px solid #333;"
        )
        layout.addWidget(self.status)

        self.bits_label = QtWidgets.QLabel()
        self.bits_label.setTextFormat(QtCore.Qt.RichText)
        self.bits_label.setStyleSheet("color:white;")
        layout.addWidget(self.bits_label)

        # ---------------- Streaming pointers ----------------
        self.ptr = 0
        self.bit_ptr = 0

        self.timer = QtCore.QTimer()
        self.timer.timeout.connect(self.update_live)
        self.timer.start(30)

    def update_live(self):
        cfg = self.cfg

        if self.ptr + self.Nw >= len(self.rx) or self.bit_ptr >= len(self.bits_hat):
            self.ptr = 0
            self.bit_ptr = 0

        win = self.rx[self.ptr:self.ptr + self.Nw]

        # Persistence update
        self.persist_buffers = [win.astype(np.float32)] + self.persist_buffers[:-1]
        for i in range(self.persist_len):
            self.persist_curves[i].setData(self.t_ms, self.persist_buffers[i])

        self.curve_time.setData(self.t_ms, win)

        # FFT
        n = min(len(win), self.nfft)
        x = win[:n]
        w = np.hanning(n)
        X = np.fft.rfft(x * w, n=self.nfft)
        mag_db = 20 * np.log10(np.abs(X) + 1e-12)
        self.curve_spec.setData(self.f, mag_db)

        # Correlation
        last = cfg.CORR_HISTORY
        x_corr = np.arange(max(0, self.bit_ptr - last), self.bit_ptr + 1)
        y_corr = self.corr_vals[max(0, self.bit_ptr - last):self.bit_ptr + 1]
        self.curve_corr.setData(x_corr, y_corr)

        # Bits window
        view = cfg.BITS_VIEW
        start = max(0, self.bit_ptr - view + 1)
        end = self.bit_ptr

        tx_bits_str = "".join(str(b) for b in self.bits[start:end + 1])
        rx_bits_str = "".join(str(b) for b in self.bits_hat[start:end + 1])

        tx_html, rx_html, n_err = format_bits_with_errors(tx_bits_str, rx_bits_str)
        shown_n = min(len(tx_bits_str), len(rx_bits_str))

        # RX black background + visible error count
        self.bits_label.setText(
            "<div style='background-color:#050607; padding:10px; border:1px solid #444;'>"
            "<div style='font-family:Consolas; font-size:16px; "
            "background-color:#0B0C0D; padding:6px; border-radius:4px;'>"
            f"<span style='color:#FFFFFF; font-weight:800'>TX:</span> {tx_html}"
            "</div>"
            "<div style='font-family:Consolas; font-size:16px; margin-top:8px; "
            "background-color:#000000; padding:6px; border-radius:4px;'>"
            f"<span style='color:#FFFFFF; font-weight:800'>RX:</span> {rx_html}"
            "</div>"
            f"<div style='font-family:Consolas; font-size:13px; margin-top:10px; color:#000000;'>"
            f"Errors (shown window): {n_err} / {shown_n}"
            "</div>"
            "</div>"
        )

        # Update digital IO plot (separate module)
        self.io_plot.update(start, end, self.bits, self.bits_hat)

        # BER
        ncmp = self.bit_ptr + 1
        ber_now = float(np.mean(self.bits[:ncmp] != self.bits_hat[:ncmp]))
        ber_total = float(np.mean(self.bits != self.bits_hat))

        self.status.setText(
            f"Fs={cfg.FS}Hz | chip_rate={cfg.CHIP_RATE} | sps={self.sps} | "
            f"SNR={cfg.SNR_DB}dB | Jammer={cfg.JAMMER_FREQ}Hz | "
            f"BER(now)={ber_now:.3f} | BER(total)={ber_total:.3f}"
        )

        self.ptr += self.Nw
        self.bit_ptr += 1


def run_gui(cfg):
    app = QtWidgets.QApplication(sys.argv)
    w = RealtimeDSSSGUI(cfg)
    w.show()
    sys.exit(app.exec())
