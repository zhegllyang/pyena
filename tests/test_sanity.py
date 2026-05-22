"""Sanity check: confirm fixtures and src layout are wired correctly."""

import pyena


def test_pyena_importable():
    """Package must be importable from src/ layout."""
    assert hasattr(pyena, "ENA")
    assert hasattr(pyena, "__version__")


def test_toy_data_fixture(toy_data):
    assert toy_data.shape == (1800, 9)
    assert toy_data["unit"].nunique() == 90


def test_codes_fixture(codes):
    assert codes == ["Data", "Theory", "Question", "Example", "Critique"]


def test_reference_pickles_load(
    pyena_reference, rena_reference, mr_reference,
    compare_reference, reproducibility_reference,
):
    """All five reference pickles must load and have expected top-level keys."""
    assert "av_matrix" in pyena_reference
    assert "pyena_avs" in rena_reference
    assert "rena_mr_points" in mr_reference
    assert "result_x" in compare_reference
    assert "result_grouped" in reproducibility_reference


def test_align_sign_helper(align_sign):
    import numpy as np
    a = np.array([[1.0, 2.0], [3.0, 4.0]])
    b = -a
    aligned = align_sign(a, b)
    assert np.allclose(aligned, b)
