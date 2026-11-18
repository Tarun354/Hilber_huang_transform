"""Monte-Carlo benchmark: estimation error vs network SNR."""
from __future__ import annotations

import csv
from pathlib import Path

import numpy as np

from .config import Config
from .pipeline import run_single

LAG_METHODS = ("data", "imf1")


def run_benchmark(cfg: Config, distances, n_trials=10, base_seed=0, verbose=True):
    """Run ``n_trials`` noise realisations at each distance; returns a list of metric dicts."""
    rows = []
    for d in distances:
        for k in range(n_trials):
            seed = base_seed + 1000 * k   # well-separated seeds (each ifo uses seed + index)
            rows.append(run_single(cfg, seed=seed, distance=d).metrics)
            if verbose:
                print(f"distance={d:g} Mpc trial {k + 1}/{n_trials} "
                      f"SNR={rows[-1]['snr_network']:.1f}", flush=True)
    return rows


def save_csv(rows, path):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    keys = list(rows[0].keys())
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=keys)
        w.writeheader()
        w.writerows(rows)


def load_csv(path):
    with open(path) as f:
        return [{k: float(v) for k, v in r.items()} for r in csv.DictReader(f)]


def summarize(rows, pairs):
    """Aggregate per distance. Lag error = |lag_method - lag_signal| averaged over detector
    pairs (ms); merger error in ms. Returns a list of dicts with median and 16/84 percentiles."""
    out = []
    for d in sorted({r["distance"] for r in rows}):
        sel = [r for r in rows if r["distance"] == d]
        rec = {"distance": d, "n": len(sel),
               "snr": float(np.median([r["snr_network"] for r in sel]))}
        series = {}
        for m in LAG_METHODS:
            series[f"lag_{m}"] = [np.mean([abs(r[f"lag_{m}_{p}"] - r[f"lag_signal_{p}"])
                                           for p in pairs]) for r in sel]
        for m in ("peak", "imf1"):
            series[f"merger_{m}"] = [1e3 * r[f"merger_err_{m}"] for r in sel]
        for name, v in series.items():
            rec[name] = tuple(float(x) for x in np.percentile(v, [50, 16, 84]))  # (median, lo, hi)
        out.append(rec)
    return out
