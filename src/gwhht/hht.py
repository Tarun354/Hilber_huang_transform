"""Empirical mode decomposition (EMD) and the Hilbert-Huang transform (HHT).

Everything here operates on plain arrays (``numpy``); PyCBC objects are accepted
because they are ``ndarray`` subclasses.
"""
from __future__ import annotations

import numpy as np
from scipy.signal import find_peaks, hilbert


def sift(x, **kwargs) -> np.ndarray:
    """EMD sifting. Returns IMFs with shape ``(n_samples, n_imfs)``; the last
    column is the residual, so ``imfs.sum(axis=1)`` reconstructs ``x``."""
    import emd
    return emd.sift.sift(np.asarray(x, dtype=float), **kwargs)


def analytic_signal(x, fs):
    """Plain Hilbert transform of ``x``: amplitude envelope, unwrapped phase and
    instantaneous frequency (Hz; length ``len(x) - 1``)."""
    z = hilbert(np.asarray(x, dtype=float))
    amp = np.abs(z)
    phase = np.unwrap(np.angle(z))
    freq = np.diff(phase) / (2.0 * np.pi) * fs
    return amp, phase, freq


def frequency_transform(imfs, fs, method="nht"):
    """Instantaneous phase, frequency and amplitude of each IMF
    (``method``: 'hilbert', 'nht' (normalised HT), 'quad', 'zc', ...)."""
    import emd
    return emd.spectra.frequency_transform(imfs, fs, method)


def hht_spectrum(imfs, fs, freq_range=(0, 500, 100), method="nht"):
    """Hilbert-Huang spectrum ``H(t, f)`` summed over IMFs.

    Returns ``(IP, IF, IA, f, hht)``; ``hht`` has shape ``(n_freq_bins, n_samples)``.
    """
    import emd
    IP, IF, IA = frequency_transform(imfs, fs, method)
    f, hht = emd.spectra.hilberthuang(IF, IA, freq_range, sum_time=False)
    return IP, IF, IA, f, hht


def reconstruct(imfs, n_imfs):
    """Sum of the first ``n_imfs`` IMFs."""
    return np.sum(imfs[:, :n_imfs], axis=1)


def reconstruction_error(x, imfs, n_imfs):
    """Mean-squared error between ``x`` and the sum of its first ``n_imfs`` IMFs."""
    return float(np.mean((np.asarray(x) - reconstruct(imfs, n_imfs)) ** 2))


def peak_time(times, x, t_lo=None, t_hi=None, use_abs=False):
    """Time of the largest local maximum of ``x`` within ``[t_lo, t_hi]``.

    This is the crude merger-time estimator used in the original notebooks
    (``use_abs=True`` also accepts large negative peaks).
    """
    times = np.asarray(times, dtype=float)
    x = np.asarray(x, dtype=float)
    mask = np.ones_like(x, dtype=bool)
    if t_lo is not None:
        mask &= times >= t_lo
    if t_hi is not None:
        mask &= times <= t_hi
    idx = np.flatnonzero(mask)
    y = np.abs(x[idx]) if use_abs else x[idx]
    peaks, _ = find_peaks(y)
    k = peaks[np.argmax(y[peaks])] if len(peaks) else int(np.argmax(y))
    return float(times[idx[k]])
