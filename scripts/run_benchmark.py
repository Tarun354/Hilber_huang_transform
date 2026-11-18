#!/usr/bin/env python
"""Monte-Carlo benchmark of lag / merger-time error vs SNR.

    python scripts/run_benchmark.py --distances 100 200 400 800 --n-trials 20
"""
import argparse
from pathlib import Path

import matplotlib
matplotlib.use("Agg")          # draw plots into files, do not open windows
import matplotlib.pyplot as plt

from gwhht.benchmark import run_benchmark, save_csv, summarize
from gwhht.config import load_config
from gwhht.plotting import plot_error_vs_snr


def main():
    # ---- read the options typed in the terminal ----
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/default.yaml")
    parser.add_argument("--distances", type=float, nargs="+", default=[100, 200, 400, 800])
    parser.add_argument("--n-trials", type=int, default=20)
    parser.add_argument("--out", default="results")
    args = parser.parse_args()

    cfg = load_config(args.config)

    # ---- run many noise realisations at each distance ----
    rows = run_benchmark(cfg, args.distances, args.n_trials)

    out = Path(args.out)
    save_csv(rows, out / "benchmark.csv")
    (out / "figures").mkdir(parents=True, exist_ok=True)

    # detector pairs as strings, e.g. "L1H1", "L1V1", "H1V1"
    pair_names = []
    for a, b in cfg.pairs():
        pair_names.append(a + b)
    summary = summarize(rows, pair_names)

    # ---- plot 1: time-lag error vs SNR ----
    plot_error_vs_snr(summary,
                      [("lag_data", "whitened data"), ("lag_imf1", "IMF1 of data")],
                      "Time-lag error vs SNR",
                      "mean |lag - lag$_{signal}$| over pairs (ms)")
    plt.savefig(out / "figures/lag_error_vs_snr.png", dpi=150)
    plt.close()

    # ---- plot 2: merger-time error vs SNR ----
    plot_error_vs_snr(summary,
                      [("merger_peak", "peak of data"), ("merger_imf1", "peak of IMF1")],
                      "Merger-time error vs SNR",
                      "|t$_{peak}$ - t$_c$| (ms)")
    plt.savefig(out / "figures/merger_error_vs_snr.png", dpi=150)
    plt.close()

    # ---- print a short table ----
    for s in summary:
        line = {}
        for key in s:
            if isinstance(s[key], tuple):
                line[key] = round(s[key][0], 3)     # median
            else:
                line[key] = s[key]
        print(line)


if __name__ == "__main__":
    main()
