# DSSS BPSK Communication System

This repository contains a complete implementation of a Direct Sequence Spread Spectrum (DSSS)
communication system using Python.

## Features
- DSSS transmitter and receiver using BPSK
- AWGN and sinusoidal jammer channel model
- Real-time oscilloscope, spectrum analyzer, and correlation detector
- Offline performance evaluation (BER vs SNR, jammer impact, processing gain)
- Aliasing demonstrations

## Main Files
- run_realtime.py – real-time visualization GUI
- run_offline_results.py – offline performance analysis
- tx_dsss.py – DSSS transmitter
- rx_dsss.py – DSSS receiver
- channel.py – noise and jammer model
- metrics.py – BER and performance metrics
