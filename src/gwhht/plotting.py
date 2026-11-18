"""Plot helpers.

All plots are made with simple matplotlib.pyplot (plt) commands.
Each function draws the plot and returns the current figure (plt.gcf()),
so you can still call plt.show() or plt.savefig(...) afterwards.
"""
import matplotlib.pyplot as plt
import numpy as np

COLORS = {"L1": "tab:red", "H1": "tab:green", "V1": "tab:blue"}


def plot_strain(data_wbp, signal_wbp, ifos, t_trigger, fband, xlim=(-2.0, 0.1)):
    """Whitened + band-passed data and signal, one panel per detector."""
    n = len(ifos)
    plt.figure(figsize=(10, 6))

    for i in range(n):
        ifo = ifos[i]
        plt.subplot(n, 1, i + 1)

        # time axis relative to the merger time
        t_data = np.asarray(data_wbp[ifo].sample_times) - t_trigger
        t_signal = np.asarray(signal_wbp[ifo].sample_times) - t_trigger

        plt.plot(t_data, data_wbp[ifo], color="darkgray", label="data")
        plt.plot(t_signal, signal_wbp[ifo], color=COLORS.get(ifo, "k"), alpha=0.6, label="signal")
        plt.axvline(0, color="k", linestyle="--")
        plt.ylabel(ifo)
        plt.xlim(xlim[0], xlim[1])

        # only show the x tick labels on the last panel
        if i < n - 1:
            plt.tick_params(labelbottom=False)

    plt.xlabel("time - t_c (s)")
    plt.suptitle("Whitened, band-passed strain (" + str(fband[0]) + "-" + str(fband[1]) + " Hz)")
    plt.subplots_adjust(hspace=0, top=0.92)
    return plt.gcf()


def plot_snr(snr_series, ifos, t_trigger, fband, xlim=(-0.5, 0.1)):
    """Matched-filter SNR time series of all detectors."""
    plt.figure(figsize=(10, 3))

    for ifo in ifos:
        t = np.asarray(snr_series[ifo].sample_times) - t_trigger
        plt.plot(t, snr_series[ifo], color=COLORS.get(ifo), label=ifo)

    plt.axvline(0, color="k", linestyle="--", alpha=0.5)
    plt.xlim(xlim[0], xlim[1])
    plt.xlabel("time - t_c (s)")
    plt.ylabel("|SNR|")
    plt.title("Matched-filter SNR (" + str(fband[0]) + "-" + str(fband[1]) + " Hz)")
    plt.legend()
    return plt.gcf()


def plot_xcorr_pairs(curves, labels=None, xlim=(-0.1, 0.1)):
    """Cross-correlation vs time lag, one panel per detector pair.

    curves: dict  method -> {pair: {lags, corr, lag, max_lag}}   (see pairwise_xcorr)
    """
    if labels is None:
        labels = {}
    methods = list(curves)
    pairs = list(curves[methods[0]])
    n = len(pairs)

    plt.figure(figsize=(9, 2.6 * n))

    for i in range(n):
        pair = pairs[i]
        plt.subplot(n, 1, i + 1)

        # green band = allowed lags (light travel time between the two detectors)
        max_lag = curves[methods[0]][pair]["max_lag"]
        plt.axvspan(-max_lag, max_lag, color="g", alpha=0.15)

        for m in methods:
            r = curves[m][pair]
            name = labels.get(m, m)
            plt.plot(r["lags"], r["corr"], alpha=0.65,
                     label=name + ": " + format(r["lag"] * 1e3, "+.2f") + " ms")

        plt.xlim(xlim[0], xlim[1])
        plt.ylim(-1.05, 1.05)
        plt.ylabel(pair)
        plt.legend(fontsize=8, loc="lower left")

    plt.xlabel("time lag (s)")
    plt.suptitle("Cross-correlation vs time lag (green band = light-travel-time window)")
    plt.tight_layout()
    return plt.gcf()


def plot_imfs(imfs, t, max_imfs=8):
    """The first few IMFs (the last one is the residual), one panel each."""
    n = min(imfs.shape[1], max_imfs)
    plt.figure(figsize=(10, 1.4 * n))

    for k in range(n):
        plt.subplot(n, 1, k + 1)
        plt.plot(t, imfs[:, k], linewidth=0.7)
        if k < imfs.shape[1] - 1:
            plt.ylabel("IMF" + str(k + 1))
        else:
            plt.ylabel("res.")
        if k < n - 1:
            plt.tick_params(labelbottom=False)

    plt.xlabel("time - t_c (s)")
    return plt.gcf()


def plot_hht(hht, t, f, freq_lims=(0, 500), fig=None):
    """Hilbert-Huang spectrum (time-frequency map)."""
    import emd
    if fig is None:
        plt.figure(figsize=(10, 4))
        fig = plt.gcf()
    emd.plotting.plot_hilberthuang(hht, t, f, freq_lims=freq_lims, fig=fig, cmap="cividis")
    return fig


def plot_error_vs_snr(summary, key_pairs, title, ylabel):
    """Error vs SNR with 16-84 percentile error bars.

    key_pairs: list of (summary_key, label), e.g. [("lag_data", "whitened data")]
    """
    plt.figure(figsize=(6, 4))
    snr = np.array([s["snr"] for s in summary])

    for key, label in key_pairs:
        med = np.array([s[key][0] for s in summary])
        low = np.array([s[key][1] for s in summary])
        high = np.array([s[key][2] for s in summary])
        plt.errorbar(snr, med, yerr=[med - low, high - med], marker="o", capsize=3, label=label)

    plt.xscale("log")
    plt.xlabel("network matched-filter SNR")
    plt.ylabel(ylabel)
    plt.title(title)
    plt.grid(alpha=0.3)
    plt.legend()
    plt.tight_layout()
    return plt.gcf()
