"""Cross-correlation based time-lag estimation between detector pairs."""
from __future__ import annotations

import numpy as np
from scipy import signal


def xcorr(x, y, dt, normalize=True):
    """Cross-correlate two equal-rate arrays; returns ``(lags_in_s, corr)``.

    The median of the correlation is subtracted (as in the original notebooks)
    and, if ``normalize``, the result is scaled so that ``max|corr| = 1``.
    Sign convention (verified in ``tests/test_timelag.py``): if ``y`` is ``x``
    delayed by ``d``, the correlation peaks at lag ``-d``.
    """
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    c = signal.correlate(x, y)
    c = c - np.median(c)
    if normalize:
        m = np.max(np.abs(c))
        if m > 0:
            c = c / m
    lags = signal.correlation_lags(len(x), len(y)) * dt
    return lags, c


def calc_xcorr(X, Y, t_ref, interval, normalize=True):
    """Cross-correlate two PyCBC TimeSeries on ``[t_ref + interval[0], t_ref + interval[1]]``."""
    x = X.time_slice(t_ref + interval[0], t_ref + interval[1])
    y = Y.time_slice(t_ref + interval[0], t_ref + interval[1])
    return xcorr(x, y, float(x.delta_t), normalize=normalize)


def lag_within_window(lags, corr, max_lag):
    """Lag of the largest |correlation| restricted to ``|lag| <= max_lag``
    (the physical light-travel time between the two detectors)."""
    idx = np.flatnonzero(np.abs(lags) <= max_lag)
    best = idx[np.argmax(np.abs(corr[idx]))]
    return float(lags[best])


def pairwise_xcorr(series, pairs, light_time, t_ref, interval, normalize=True):
    """Cross-correlation curve and lag for every detector pair.

    Parameters
    ----------
    series : dict ifo -> PyCBC TimeSeries
    pairs : list of (ifoA, ifoB)
    light_time : callable ``(a, b) -> seconds`` or dict like ``{'L1H1': 0.01}``

    Returns dict ``'L1H1' -> {'lags', 'corr', 'lag', 'max_lag'}``.
    """
    lt = light_time if callable(light_time) else (
        lambda a, b: light_time.get(f"{a}{b}", light_time.get(f"{b}{a}")))
    out = {}
    for a, b in pairs:
        lags, corr = calc_xcorr(series[a], series[b], t_ref, interval, normalize)
        max_lag = lt(a, b)
        out[f"{a}{b}"] = dict(lags=lags, corr=corr, max_lag=max_lag,
                              lag=lag_within_window(lags, corr, max_lag))
    return out
