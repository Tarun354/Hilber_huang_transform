import numpy as np
from gwhht.preprocess import bandpass_array


def test_bandpass_removes_out_of_band_tone():
    fs = 2048
    t = np.arange(0, 8, 1 / fs)
    x = np.sin(2 * np.pi * 100 * t) + np.sin(2 * np.pi * 600 * t)
    y = bandpass_array(x, [15, 250], fs)
    spec = np.abs(np.fft.rfft(y))
    f = np.fft.rfftfreq(len(y), 1 / fs)
    assert spec[np.argmin(abs(f - 100))] > 50 * spec[np.argmin(abs(f - 600))]
