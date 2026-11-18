# Method

## Data
A coherent `IMRPhenomD` signal (m1 = 9, m2 = 8 M☉, non-spinning, 200 Mpc by default, iota = pi/4) is generated in
L1, H1 and V1 with PyCBC's `FDomainDetFrameGenerator`, for a fixed sky position. Coloured Gaussian noise is drawn from the
aLIGO zero-detuned-high-power PSD (L1, H1) and the AdV design PSD (V1). Noise seeds are passed explicitly (`seed + k`
for the k-th detector).

## Pre-processing
Frequency-domain whitening by the PSD, then a 4th-order zero-phase Butterworth band-pass (15-250 Hz).

## SNR
Matched filter with the injected waveform as template (optimal SNR). Network SNR = sqrt(sum of squared per-detector peaks).

## Hilbert-Huang transform
Empirical mode decomposition splits the data into intrinsic mode functions,
x(t) = sum_i IMF_i(t) + r(t). Each IMF has an analytic signal z_i = IMF_i + j H[IMF_i] = A_i e^{j theta_i} and
instantaneous frequency f_i = (1/2pi) d theta_i / dt (normalised Hilbert transform, `emd` package). The Hilbert-Huang
spectrum is HHS(t, f) = sum_i A_i(t) delta(f - f_i(t)).

## Time lag
For each detector pair the data (or IMF1) in [t_c - 3.5 s, t_c + 1.5 s] are cross-correlated; the lag is the position of
max |corr| restricted to |lag| <= light-travel time of the pair. The reference lag is obtained the same way from the
**noise-free** whitened signal, so the lag error isolates the effect of noise. Lags are quantised to 1/fs (0.49 ms at 2048 Hz).
Sign convention: if y is x delayed by d, the peak is at -d.

## Merger time
Time of the highest local maximum of the whitened data (or of IMF1) in [t_c - 10 s, t_c + 0.5 s], compared with the injected
t_c. Note the noise-free whitened/band-passed signal itself peaks ~1 ms away from t_c, which sets a floor.

## Benchmark
`scripts/run_benchmark.py`: the same injection at several distances (hence SNRs), 30 noise realisations each. Seeds are
shared across distances (paired design).
