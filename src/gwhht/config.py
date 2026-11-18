"""Configuration for the injection / analysis pipeline.

All numbers that used to be hard-coded in the notebooks live here, and can be
overridden from a YAML file (see ``configs/default.yaml``).
"""
from __future__ import annotations

from dataclasses import dataclass, field, fields
from pathlib import Path

import yaml

# Light-travel times (s) between detector pairs, copied from the original notebooks.
DEFAULT_LIGHT_TIME = {
    "L1H1": 0.010012846152267725,
    "L1V1": 0.02644834101635671,
    "H1V1": 0.027287979933397113,
}


@dataclass
class Config:
    # --- binary ---
    approximant: str = "IMRPhenomD"
    m1: float = 9.0
    m2: float = 8.0
    s1z: float = 0.0
    s2z: float = 0.0
    distance: float = 200.0          # Mpc
    iota: float = 0.7853981633974483  # pi/4
    phic: float = 1.0471975511965976  # pi/3
    mode_array: list = field(default_factory=lambda: [[2, 2]])

    # --- sky location / polarisation ---
    ra: float = 1.2566370614359172    # pi/2.5
    dec: float = -0.5235987755982988  # -pi/6
    psi: float = 0.0

    # --- data ---
    ifos: list = field(default_factory=lambda: ["L1", "H1", "V1"])
    fsamp: int = 2048
    flow: float = 10.0
    fhigh: float = 1024.0
    t_start: float = 1367469555.0    # 7 May 2023, 04:38:57 UTC
    seed: int = 0

    # --- analysis ---
    fband: list = field(default_factory=lambda: [15.0, 250.0])
    window: list = field(default_factory=lambda: [-10.0, 2.0])         # data segment, rel. to trigger (s)
    merger_search: list = field(default_factory=lambda: [-10.0, 0.5])  # merger-time search range (s)
    xcorr_interval: list = field(default_factory=lambda: [-3.5, 1.5])  # cross-corr segment (s)
    light_time: dict = field(default_factory=lambda: dict(DEFAULT_LIGHT_TIME))

    def pairs(self):
        """Detector pairs in the order they are processed, e.g. L1H1, L1V1, H1V1."""
        out = []
        for i, a in enumerate(self.ifos):
            for b in self.ifos[i + 1:]:
                out.append((a, b))
        return out

    def light_time_for(self, a: str, b: str) -> float:
        return self.light_time.get(f"{a}{b}", self.light_time.get(f"{b}{a}"))


def load_config(path: str | Path | None = None, **overrides) -> Config:
    """Load a Config from YAML (optional) and apply keyword overrides."""
    data = {}
    if path is not None:
        with open(path) as f:
            data = yaml.safe_load(f) or {}
    data.update(overrides)
    valid = {f.name for f in fields(Config)}
    unknown = set(data) - valid
    if unknown:
        raise ValueError(f"Unknown config keys: {sorted(unknown)}")
    return Config(**data)
