"""Matched-filter SNR with the injected waveform as template."""
from __future__ import annotations

import numpy as np


def matched_filter_snr(sim, fband=None):
    """|matched-filter SNR| time series per detector (template = injection).

    Returns ``(snr_series, peak_snr, network_snr)`` where ``network_snr`` is
    ``sqrt(sum_i peak_i^2)`` (an approximation: per-detector peaks are taken
    independently, not at a common time).
    """
    from pycbc.waveform import get_fd_waveform
    from pycbc.filter.matchedfilter import matched_filter

    fband = fband or sim.cfg.fband
    hp, _ = get_fd_waveform(**sim.frozen_params)
    hp.resize(sim.n_freq)
    series, peak = {}, {}
    for ifo in sim.cfg.ifos:
        snr = matched_filter(template=hp, data=sim.data_fd[ifo], psd=sim.psd[ifo],
                             low_frequency_cutoff=fband[0], high_frequency_cutoff=fband[1])
        series[ifo] = abs(snr)
        peak[ifo] = float(series[ifo].max())
    network = float(np.sqrt(sum(v ** 2 for v in peak.values())))
    return series, peak, network
