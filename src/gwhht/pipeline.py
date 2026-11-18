"""End-to-end run: simulate -> whiten/band-pass -> SNR -> HHT -> time lags + merger time."""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from .config import Config
from .hht import peak_time, sift
from .preprocess import preprocess_network
from .simulate import simulate_network
from .snr import matched_filter_snr
from .timelag import pairwise_xcorr


@dataclass
class RunResult:
    metrics: dict                                  # flat dict, one row of the benchmark table
    artifacts: dict = field(default_factory=dict)  # time series, IMFs, curves (for plotting)


def _crop(ts, t0, t1):
    t0 = max(t0, float(ts.start_time))
    t1 = min(t1, float(ts.end_time))
    return ts.time_slice(t0, t1)


def run_single(cfg: Config, seed: int | None = None, distance: float | None = None) -> RunResult:
    """Run the full analysis once. ``metrics`` holds (per detector pair ``AB``):

    * ``lag_{data,imf1,signal}_AB``  time lag (ms) from whitened data / IMF1 of the data /
      noise-free whitened signal (the latter is the reference),
    * ``merger_err_{peak,imf1,signal}`` |t_peak - t_trigger| (s) in the first detector.
    """
    from pycbc.types import TimeSeries

    sim = simulate_network(cfg, seed=seed, distance=distance)
    cfg = sim.cfg
    sig_all, dat_all = preprocess_network(sim)
    snr_series, snr_peak, snr_net = matched_filter_snr(sim)

    tt = sim.t_trigger
    sig = {i: _crop(sig_all[i], tt + cfg.window[0], tt + cfg.window[1]) for i in cfg.ifos}
    dat = {i: _crop(dat_all[i], tt + cfg.window[0], tt + cfg.window[1]) for i in cfg.ifos}

    imfs, imf1 = {}, {}
    for i in cfg.ifos:
        imfs[i] = sift(dat[i])
        imf1[i] = TimeSeries(imfs[i][:, 0], dat[i].delta_t, epoch=dat[i].start_time, copy=False)

    pairs = cfg.pairs()
    lt = cfg.light_time_for
    curves = {
        "data": pairwise_xcorr(dat, pairs, lt, tt, cfg.xcorr_interval),
        "imf1": pairwise_xcorr(imf1, pairs, lt, tt, cfg.xcorr_interval),
        "signal": pairwise_xcorr(sig, pairs, lt, tt, cfg.xcorr_interval),
    }

    m = dict(seed=cfg.seed if seed is None else seed, distance=cfg.distance,
             snr_network=snr_net, **{f"snr_{k}": v for k, v in snr_peak.items()})
    for method, res in curves.items():
        for pair, r in res.items():
            m[f"lag_{method}_{pair}"] = r["lag"] * 1e3  # ms

    ref = cfg.ifos[0]
    t = np.asarray(dat[ref].sample_times, dtype=float)
    lo, hi = tt + cfg.merger_search[0], tt + cfg.merger_search[1]
    m["merger_err_peak"] = abs(peak_time(t, dat[ref], lo, hi) - tt)
    m["merger_err_imf1"] = abs(peak_time(t, imfs[ref][:, 0], lo, hi) - tt)
    m["merger_err_signal"] = abs(peak_time(t, sig[ref], lo, hi) - tt)

    arts = dict(sim=sim, signal_wbp=sig_all, data_wbp=dat_all, signal=sig, data=dat,
                imfs=imfs, imf1=imf1, snr_series=snr_series, curves=curves)
    return RunResult(metrics=m, artifacts=arts)
