# config.py

FS = 48000
CHIP_RATE = 6000          # must divide FS for our simple upsample method
CHIPS_PER_BIT = 15
N_BITS = 800

SNR_DB = -15               # realtime demo SNR
JAMMER_AMP = 2.5
JAMMER_FREQ = 2000        # Hz

# Realtime display
WINDOW_SEC = 0.01         # 10 ms scope window
NFFT = 4096
PERSIST_LEN = 10
CORR_HISTORY = 80
BITS_VIEW = 80            # show last N recovered bits in GUI

# Offline evaluation
SNR_SWEEP = list(range(-25, 6, 3))  # -25 to +5 dB
RNG_SEED = 0
