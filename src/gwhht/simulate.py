"""Simulate a coherent compact-binary signal + Gaussian noise in a detector network."""
from __future__ import annotations

from dataclasses import dataclass, replace

import numpy as np

from .config import Config


@dataclass
class SimulatedData:
    cfg: Config
    t_trigger: float          # GPS time of coalescence (tc)
    signal_duration: float    # in-band duration of the signal (s)
    seglen: int               # segment length (s), power of two
    delta_f: float
    n_freq: int
    frozen_params: dict
    signal_fd: dict           # noise-free signal, frequency domain, per ifo
    data_fd: dict             # signal + noise, frequency domain, per ifo
    psd: dict                 # PSD per ifo


def simulate_network(cfg: Config, seed: int | None = None,
                     distance: float | None = None) -> SimulatedData:
    """Generate a coherent signal in ``cfg.ifos`` and add coloured Gaussian noise.

    Parameters
    ----------
    seed : noise seed (defaults to ``cfg.seed``). Each detector gets ``seed + k``
        passed *explicitly* to PyCBC, so a given seed always gives the same data.
    distance : optional override of ``cfg.distance`` (Mpc) - used to scan SNR.
    """
    from pycbc.waveform import get_waveform_filter_length_in_time as chirplen
    from pycbc.waveform.generator import FDomainDetFrameGenerator, FDomainCBCGenerator
    from pycbc.psd import aLIGOZeroDetHighPower, AdVDesignSensitivityP1200087
    from pycbc.noise.gaussian import frequency_noise_from_psd

    if distance is not None:
        cfg = replace(cfg, distance=distance)
    seed = cfg.seed if seed is None else seed

    duration = chirplen(approximant=cfg.approximant, mass1=cfg.m1, mass2=cfg.m2,
                        spin1z=cfg.s1z, spin2z=cfg.s2z, f_lower=cfg.flow)
    seglen = 2 ** (int(np.log2(duration)) + 1)
    delta_f = 1.0 / seglen
    n_time = cfg.fsamp * seglen
    n_freq = int(n_time // 2 + 1)
    t_trigger = cfg.t_start + duration

    frozen = dict(approximant=cfg.approximant, mass1=cfg.m1, mass2=cfg.m2,
                  spin1z=cfg.s1z, spin2z=cfg.s2z, delta_f=delta_f, f_lower=cfg.flow,
                  f_final=cfg.fhigh, inclination=cfg.iota, distance=cfg.distance,
                  mode_array=cfg.mode_array, coa_phase=cfg.phic)

    generator = FDomainDetFrameGenerator(
        FDomainCBCGenerator, detectors=cfg.ifos, epoch=cfg.t_start,
        variable_args=["ra", "dec", "tc", "polarization"], **frozen)
    sig = generator.generate(ra=cfg.ra, dec=cfg.dec, tc=t_trigger, polarization=cfg.psi)
    for ifo in cfg.ifos:
        if len(sig[ifo]) != n_freq:
            sig[ifo].resize(n_freq)

    psd_v1 = AdVDesignSensitivityP1200087(n_freq, delta_f=delta_f, low_freq_cutoff=cfg.flow)
    psd_ligo = aLIGOZeroDetHighPower(n_freq, delta_f=delta_f, low_freq_cutoff=cfg.flow)
    psd = {ifo: (psd_v1 if ifo == "V1" else psd_ligo) for ifo in cfg.ifos}

    data = {}
    for k, ifo in enumerate(cfg.ifos):
        noise = frequency_noise_from_psd(psd[ifo], seed=seed + k)
        data[ifo] = sig[ifo] + noise
    # avoid division by zero when whitening (PSD is 0 below the low-frequency cutoff)
    for p in {id(p): p for p in psd.values()}.values():
        p.data[np.where(p.data == 0)[0]] = 1e-50

    return SimulatedData(cfg=replace(cfg), t_trigger=t_trigger, signal_duration=duration,
                         seglen=seglen, delta_f=delta_f, n_freq=n_freq,
                         frozen_params=frozen, signal_fd=sig, data_fd=data, psd=psd)
