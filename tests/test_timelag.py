import numpy as np
from gwhht.timelag import lag_within_window, xcorr


def _chirp(t):
    return np.sin(2 * np.pi * (20 * t + 40 * t ** 2)) * np.exp(-((t - 0.8) ** 2) / 0.1)


def test_recovers_known_shift_magnitude():
    fs, shift = 4096, 0.006
    t = np.arange(0, 2, 1 / fs)
    x, y = _chirp(t), _chirp(t - shift)          # y is x delayed by `shift`
    lags, c = xcorr(x, y, 1 / fs)
    lag = lag_within_window(lags, c, 0.01)
    assert abs(abs(lag) - shift) <= 1.5 / fs


def test_sign_convention_documented():
    """scipy.correlate(x, y): if y = x delayed by d, the peak sits at lag = -d."""
    fs, shift = 4096, 0.004
    t = np.arange(0, 2, 1 / fs)
    x, y = _chirp(t), _chirp(t - shift)
    lags, c = xcorr(x, y, 1 / fs)
    assert lag_within_window(lags, c, 0.01) < 0


def test_window_restricts_search():
    lags = np.linspace(-0.1, 0.1, 201)
    corr = np.zeros_like(lags)
    corr[0] = 1.0        # strong peak at -0.1 s, outside the window
    corr[110] = 0.5      # weaker peak inside window
    assert lag_within_window(lags, corr, 0.02) == lags[110]


def test_normalisation():
    x = np.random.default_rng(0).normal(size=512)
    _, c = xcorr(x, x, 1.0, normalize=True)
    assert np.isclose(np.max(np.abs(c)), 1.0)
