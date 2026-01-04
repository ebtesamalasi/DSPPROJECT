import sys
import numpy as np
import pyqtgraph as pg
from pyqtgraph.Qt import QtCore, QtWidgets

# ---- Oscilloscope look ----
pg.setConfigOptions(antialias=True)

class Scope(QtWidgets.QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Oscilloscope (Real-Time)")
        self.resize(1000, 450)

        # Central widget
        cw = QtWidgets.QWidget()
        self.setCentralWidget(cw)
        layout = QtWidgets.QVBoxLayout(cw)

        # Plot widget
        self.plot = pg.PlotWidget()
        layout.addWidget(self.plot)

        # Scope styling
        self.plot.setBackground((5, 6, 7))
        self.plot.showGrid(x=True, y=True, alpha=0.25)
        self.plot.setLabel("bottom", "Time", units="s")
        self.plot.setLabel("left", "Amplitude")
        self.plot.setYRange(-2.5, 2.5)

        # Data settings
        self.fs = 48000
        self.window_sec = 0.01
        self.N = int(self.fs * self.window_sec)
        self.t = np.arange(self.N) / self.fs

        # Main trace (neon green)
        self.curve = self.plot.plot(self.t, np.zeros(self.N), pen=pg.mkPen((0, 255, 110), width=2))

        # Persistence traces (fading history)
        self.persistence = []
        self.persistence_len = 12
        for i in range(self.persistence_len):
            alpha = int(140 * (1 - i / self.persistence_len))  # fade
            pen = pg.mkPen((0, 255, 110, alpha), width=1)
            self.persistence.append(self.plot.plot(self.t, np.zeros(self.N), pen=pen))

        # Timer for real-time updates
        self.k = 0
        self.timer = QtCore.QTimer()
        self.timer.timeout.connect(self.update_scope)
        self.timer.start(30)  # ~33 FPS

    def update_scope(self):
        # Example live signal: sine + noise + occasional burst
        f0 = 1200
        x = 1.2*np.sin(2*np.pi*f0*self.t + 0.15*self.k) + 0.15*np.random.randn(self.N)
        if (self.k % 30) < 5:
            x += 0.8*np.sin(2*np.pi*3500*self.t)

        # Update persistence: shift older traces down
        for i in range(self.persistence_len-1, 0, -1):
            self.persistence[i].setData(self.t, self.persistence[i-1].yData)

        self.persistence[0].setData(self.t, x)
        self.curve.setData(self.t, x)

        self.k += 1

if __name__ == "__main__":
    app = QtWidgets.QApplication(sys.argv)
    w = Scope()
    w.show()
    sys.exit(app.exec())
