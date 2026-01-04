# io_bits_plot.py
import numpy as np
import pyqtgraph as pg

class IOBitsPlot:
    """
    Digital TX vs RX step-plot (logic-analyzer style).
    Compatible with older pyqtgraph stepMode=True requirement:
      len(x) must be len(y)+1
    """
    def __init__(self, title="INPUT vs OUTPUT (Digital Bits)"):
        self.title = title
        self.p_bits = None
        self.curve_tx = None
        self.curve_rx = None

    def add(self, glw, row=3, col=0):
        self.p_bits = glw.addPlot(row=row, col=col, title=self.title)
        self.p_bits.showGrid(x=True, y=True, alpha=0.25)
        self.p_bits.setLabel("bottom", "Bit index")
        self.p_bits.setLabel("left", "Value")
        self.p_bits.setYRange(-0.5, 1.5)

        # Keep stepMode=True for maximum compatibility
        self.curve_tx = self.p_bits.plot(
            [], [], stepMode=True,
            pen=pg.mkPen("#00D5FF", width=2)
        )
        self.curve_rx = self.p_bits.plot(
            [], [], stepMode=True,
            pen=pg.mkPen("#00FF6A", width=2)
        )
        return self

    def update(self, start_idx, end_idx, bits_tx, bits_rx, rx_offset=-0.15):
        if self.curve_tx is None or self.curve_rx is None:
            return
        if end_idx < start_idx:
            return

        # y length = N
        tx = bits_tx[start_idx:end_idx + 1]
        rx = bits_rx[start_idx:end_idx + 1] + rx_offset

        # ✅ x length must be N+1 for stepMode=True
        # Example: for 1 bit -> y has len 1, x must have len 2
        x = np.arange(start_idx, end_idx + 2)

        self.curve_tx.setData(x, tx)
        self.curve_rx.setData(x, rx)
