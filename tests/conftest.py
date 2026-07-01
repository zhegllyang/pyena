"""Shared pytest fixtures for the pyena test suite.

All test modules import these via the standard pytest fixture-discovery
mechanism — no explicit imports are needed.
"""

from __future__ import annotations

import pickle
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

DATA_DIR = Path(__file__).resolve().parent / "data"


# ===== Path / loader fixtures =====

@pytest.fixture(scope="session")
def data_dir():
    """Absolute path to the bundled test-data directory."""
    return DATA_DIR


@pytest.fixture(scope="session")
def toy_data():
    """The 90-unit toy DataFrame used throughout validation."""
    p = DATA_DIR / "toy_data_n90.csv"
    if not p.exists():
        pytest.fail(f"toy_data_n90.csv missing at {p}")
    return pd.read_csv(p)


@pytest.fixture(scope="session")
def codes():
    """The five code names used by the toy dataset."""
    return ["Data", "Theory", "Question", "Example", "Critique"]


# ===== Reference-pickle fixtures =====

@pytest.fixture(scope="session")
def pyena_reference():
    """pyena_reference_output.pkl — AV, normalized, centered, SVD coords."""
    p = DATA_DIR / "pyena_reference_output.pkl"
    if not p.exists():
        pytest.fail(f"pyena_reference_output.pkl missing at {p}")
    with p.open("rb") as f:
        return pickle.load(f)


@pytest.fixture(scope="session")
def rena_reference():
    """rena_validation_results.pkl — rENA-exact AV + SVD comparison."""
    p = DATA_DIR / "rena_validation_results.pkl"
    if not p.exists():
        pytest.fail(f"rena_validation_results.pkl missing at {p}")
    with p.open("rb") as f:
        return pickle.load(f)


@pytest.fixture(scope="session")
def mr_reference():
    """mr_validation_results.pkl — rENA means-rotation comparison."""
    p = DATA_DIR / "mr_validation_results.pkl"
    if not p.exists():
        pytest.fail(f"mr_validation_results.pkl missing at {p}")
    with p.open("rb") as f:
        return pickle.load(f)


@pytest.fixture(scope="session")
def compare_reference():
    """compare_reference.pkl — ena_group_comparison reference outputs."""
    p = DATA_DIR / "compare_reference.pkl"
    if not p.exists():
        pytest.fail(f"compare_reference.pkl missing at {p}")
    with p.open("rb") as f:
        return pickle.load(f)


@pytest.fixture(scope="session")
def reproducibility_reference():
    """reproducibility_reference.pkl — reproducibility_metrics reference."""
    p = DATA_DIR / "reproducibility_reference.pkl"
    if not p.exists():
        pytest.fail(f"reproducibility_reference.pkl missing at {p}")
    with p.open("rb") as f:
        return pickle.load(f)


# ===== Helpers =====

@pytest.fixture(scope="session")
def align_sign():
    """Return a function that sign-aligns the columns of `a` to those of `b`.

    SVD-based outputs have an arbitrary sign per component; alignment is the
    standard way to compare them without spurious mismatches.
    """
    def _align_sign(a, b):
        out = a.copy()
        for k in range(a.shape[1]):
            if np.dot(a[:, k], b[:, k]) < 0:
                out[:, k] = -a[:, k]
        return out
    return _align_sign

# ===== RS.data cross-speaker fixtures =====
@pytest.fixture(scope="session")
def rsdata_raw():
    """RS.data — multi-speaker conversations, the cross-speaker regression input."""
    p = DATA_DIR / "rs_data.csv"
    if not p.exists():
        pytest.fail(f"rs_data.csv missing at {p}")
    return pd.read_csv(p)


@pytest.fixture(scope="session")
def rsdata_codes():
    """The six RS.data code names, in rENA column order."""
    return ["Data", "Technical.Constraints", "Performance.Parameters",
            "Client.and.Consultant.Requests", "Design.Reasoning", "Collaboration"]


@pytest.fixture(scope="session")
def rsdata_rena_adjacency():
    """rENA-exact adjacency for RS.data (48 units x 15 pairs), window.size.back=4.

    Ground truth exported from rENA 0.3.1; columns are in rENA's native
    (column-major upper-triangle) pair order.
    """
    p = DATA_DIR / "rena_adjacency_rsdata.csv"
    if not p.exists():
        pytest.fail(f"rena_adjacency_rsdata.csv missing at {p}")
    return pd.read_csv(p).set_index("ENA_UNIT")
    
@pytest.fixture(scope="session")
def rsdata_rena_svd():
    """rENA SVD coords for RS.data (48 units), window.size.back=4.
    Columns: SVD1..SVD15; only the first two are compared (pyena is 2D)."""
    p = DATA_DIR / "rena_svd_rsdata.csv"
    if not p.exists():
        pytest.fail(f"rena_svd_rsdata.csv missing at {p}")
    return pd.read_csv(p).set_index("ENA_UNIT")


@pytest.fixture(scope="session")
def rsdata_rena_mr():
    """rENA means-rotation coords for RS.data (FirstGame vs SecondGame).
    Columns: MR1, SVD2..SVD15; only the first two are compared."""
    p = DATA_DIR / "rena_mr_rsdata.csv"
    if not p.exists():
        pytest.fail(f"rena_mr_rsdata.csv missing at {p}")
    return pd.read_csv(p).set_index("ENA_UNIT")
    
@pytest.fixture(scope="session")
def rsdata_rena_nodes():
    """rENA network node positions for RS.data (6 codes), SVD projection.
    Columns: code, SVD1..SVD15; only the first two dims are compared."""
    p = DATA_DIR / "rena_nodes_rsdata.csv"
    if not p.exists():
        pytest.fail(f"rena_nodes_rsdata.csv missing at {p}")
    return pd.read_csv(p)