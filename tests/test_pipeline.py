"""Integration test (skipped when PyCBC is not installed)."""
import pytest

pytest.importorskip("pycbc")
pytest.importorskip("emd")
from gwhht.config import Config
from gwhht.pipeline import run_single


def test_run_single_is_reproducible_and_physical():
    cfg = Config()
    a = run_single(cfg, seed=3).metrics
    b = run_single(cfg, seed=3).metrics
    assert a == b                                   # same seed -> identical data
    assert a["snr_network"] > 8                     # loud injection
    # noise-free lags must lie inside each pair's light-travel-time window
    for p, lt in cfg.light_time.items():
        assert abs(a[f"lag_signal_{p}"]) <= lt * 1e3 + 1e-9
