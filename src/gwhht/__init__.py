"""gwhht: Hilbert-Huang transform and cross-correlation time-lag estimation
for simulated compact-binary gravitational-wave signals in a detector network.

Heavy dependencies (PyCBC / LAL) are imported lazily inside the functions that
need them, so the array-level routines (`hht`, `timelag`) work without them.
"""

__version__ = "0.1.0"
