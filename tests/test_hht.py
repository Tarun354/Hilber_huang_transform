import numpy as np
import pytest

pytest.importorskip("emd")
from gwhht.hht import analytic_signal, peak_time, reconstruction_error, sift


@pytest.fixture
def two_tone():
    fs = 1024
    t = np.arange(0, 4, 1 / fs)
    return fs, t, np.sin(2 * np.pi * 8 * t) + 0.7 * np.sin(2 * np.pi * 60 * t)


def test_emd_perfect_reconstruction(two_tone):
    _, _, x = two_tone
    imfs = sift(x)
    assert reconstruction_error(x, imfs, imfs.shape[1]) < 1e-20


def test_first_imf_is_fastest(two_tone):
    fs, _, x = two_tone
    imfs = sift(x)
    _, _, f = analytic_signal(imfs[:, 0], fs)
    assert 40 < np.median(f) < 80          # ~60 Hz component


def test_hilbert_frequency():
    fs = 1024
    t = np.arange(0, 2, 1 / fs)
    _, _, f = analytic_signal(np.sin(2 * np.pi * 50 * t), fs)
    assert abs(np.median(f) - 50) < 0.5


def test_peak_time_window():
    t = np.linspace(0, 10, 1001)
    x = np.exp(-((t - 3) ** 2)) + 2 * np.exp(-((t - 8) ** 2))
    assert abs(peak_time(t, x) - 8) < 0.02
    assert abs(peak_time(t, x, 0, 5) - 3) < 0.02
