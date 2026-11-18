#!/usr/bin/env python
"""Run the full pipeline once and save figures + a JSON summary.

    python scripts/run_pipeline.py --config configs/default.yaml --out results
"""
import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")          # draw plots into files, do not open windows
import matplotlib.pyplot as plt
import numpy as np

from gwhht.config import load_config
from gwhht.hht import hht_spectrum
from gwhht import plotting as P
from gwhht.pipeline import run_single


def main():
    # ---- read the options typed in the terminal ----
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/default.yaml")
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--distance", type=float, default=None, help="Mpc (overrides config)")
    parser.add_argument("--out", default="results")
    args = parser.parse_args()

    # ---- run the analysis once ----
    cfg = load_config(args.config)
    result = run_single(cfg, seed=args.seed, distance=args.distance)

    arts = result.artifacts
    out = Path(args.out)
    (out / "figures").mkdir(parents=True, exist_ok=True)

    t_trigger = arts["sim"].t_trigger
    ifos = cfg.ifos

    # ---- plot 1: whitened strain ----
    P.plot_strain(arts["data_wbp"], arts["signal_wbp"], ifos, t_trigger, cfg.fband)
    plt.savefig(out / "figures/strain_whitened.png", dpi=150, bbox_inches="tight")
    plt.close()

    # ---- plot 2: matched-filter SNR ----
    P.plot_snr(arts["snr_series"], ifos, t_trigger, cfg.fband)
    plt.savefig(out / "figures/snr_timeseries.png", dpi=150, bbox_inches="tight")
    plt.close()

    # ---- plot 3: cross-correlation vs time lag ----
    labels = {"data": "whitened data", "imf1": "IMF1 of data", "signal": "noise-free signal"}
    P.plot_xcorr_pairs(arts["curves"], labels=labels)
    plt.savefig(out / "figures/xcorr_time_lag.png", dpi=150, bbox_inches="tight")
    plt.close()

    # ---- plot 4: IMFs of the first detector ----
    first = ifos[0]
    t = np.asarray(arts["data"][first].sample_times) - t_trigger
    P.plot_imfs(arts["imfs"][first], t)
    plt.savefig(out / "figures/imfs.png", dpi=150, bbox_inches="tight")
    plt.close()

    # ---- plot 5: Hilbert-Huang spectrum of the first detector ----
    IP, IF, IA, f, hht = hht_spectrum(arts["imfs"][first], cfg.fsamp, freq_range=(0, 500, 100))
    P.plot_hht(hht, t, f)
    plt.savefig(out / "figures/hht_spectrum.png", dpi=150, bbox_inches="tight")
    plt.close()

    # ---- save the numbers ----
    (out / "summary.json").write_text(json.dumps(result.metrics, indent=2))
    print(json.dumps(result.metrics, indent=2))


if __name__ == "__main__":
    main()
