"""Whitening and band-passing of strain data."""
from __future__ import annotations

import numpy as np
from scipy.signal import butter, filtfilt


def whiten(freq_series, psd):
    """Whiten a PyCBC FrequencySeries with ``psd``; returns a TimeSeries."""
    dt = 1.0 / freq_series.sample_rate
    norm = 1.0 / np.sqrt(1.0 / (dt * 2))
    white_hf = freq_series / np.sqrt(psd.data) * norm
    return white_hf.to_timeseries()


def bandpass_array(x, fband, fs, order=4):
    """Zero-phase Butterworth band-pass of a 1-D array, with the original
    notebooks' normalisation ``sqrt((f_hi - f_lo) / (fs / 2))``."""
    b, a = butter(order, [fband[0] * 2.0 / fs, fband[1] * 2.0 / fs], btype="band")
    norm = np.sqrt((fband[1] - fband[0]) / (fs / 2))
    return filtfilt(b, a, np.asarray(x)) / norm


def bandpass(strain, fband, fs):
    """Band-pass a PyCBC TimeSeries, keeping its epoch and sampling interval."""
    from pycbc.types import TimeSeries
    y = bandpass_array(strain, fband, fs)
    return TimeSeries(y, strain.delta_t, epoch=strain.start_time, copy=False)


def preprocess_network(sim, fband=None):
    """Whiten + band-pass signal-only and signal+noise data for every detector.

    Returns ``(signal_whitenbp, data_whitenbp)``, dicts of TimeSeries keyed by ifo.
    """
    fband = fband or sim.cfg.fband
    sig, dat = {}, {}
    for ifo in sim.cfg.ifos:
        s = whiten(sim.signal_fd[ifo], sim.psd[ifo])
        d = whiten(sim.data_fd[ifo], sim.psd[ifo])
        fs = d.get_sample_rate()
        sig[ifo] = bandpass(s, fband, fs)
        dat[ifo] = bandpass(d, fband, fs)
    return sig, dat
