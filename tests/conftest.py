"""Shared pytest fixtures for the MPM cloth tests.

Tests run on CPU at a reduced grid (16x16 cloth, 32-cell grid) so the suite
completes in well under a minute.
"""

from copy import deepcopy
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
CONFIG_PATH = ROOT / "configs" / "mpm.yaml"


@pytest.fixture(scope="session")
def base_cfg():
    from src.mpm_cloth import load_mpm_config
    return load_mpm_config(CONFIG_PATH)


@pytest.fixture
def small_cfg(base_cfg):
    """Reduced-resolution config for fast CPU tests."""
    cfg = deepcopy(base_cfg)
    cfg["backend"]["arch"] = "cpu"
    cfg["cloth"]["grid"] = [16, 16]
    cfg["mpm"]["grid_resolution"] = 32
    cfg["mpm"]["dx_m"] = cfg["mpm"]["domain_size_m"] / cfg["mpm"]["grid_resolution"]
    cfg["mpm"]["inv_dx"] = 1.0 / cfg["mpm"]["dx_m"]
    # Drop close enough that contact happens in <0.5 s of sim time
    cfg["cloth"]["initial_height_m"] = 0.7
    return cfg
