"""Tests for pyena.core.normalize."""

import numpy as np

from pyena.core.normalize import sphere_normalize


# ===== Reference regression =====

def test_sphere_normalize_matches_reference(pyena_reference):
    """Same AV matrix must yield the same sphere-normalized matrix as notebook 01."""
    normalized = sphere_normalize(pyena_reference["av_matrix"])
    diff = np.abs(normalized - pyena_reference["normalized_matrix"])
    assert diff.max() == 0.0


# ===== Mathematical properties =====

class TestSphereNormalize:
    def test_2d_each_row_has_unit_norm(self):
        rng = np.random.default_rng(0)
        av = rng.random((50, 10))
        out = sphere_normalize(av)
        norms = np.linalg.norm(out, axis=1)
        np.testing.assert_allclose(norms, np.ones(50), atol=1e-14)

    def test_1d_input_returns_unit_vector(self):
        v = np.array([3.0, 4.0])  # ||v|| = 5
        out = sphere_normalize(v)
        np.testing.assert_allclose(out, [0.6, 0.8])
        assert np.isclose(np.linalg.norm(out), 1.0)

    def test_zero_row_returned_unchanged(self):
        """Zero-norm rows must be passed through (rENA convention)."""
        av = np.array([[3.0, 4.0], [0.0, 0.0], [1.0, 0.0]])
        out = sphere_normalize(av)
        np.testing.assert_array_equal(out[1], [0.0, 0.0])  # unchanged
        np.testing.assert_allclose(out[0], [0.6, 0.8])
        np.testing.assert_allclose(out[2], [1.0, 0.0])

    def test_zero_vector_1d_returned_unchanged(self):
        out = sphere_normalize(np.zeros(5))
        np.testing.assert_array_equal(out, np.zeros(5))

    def test_idempotent(self):
        """Normalizing an already-normalized vector is a no-op."""
        rng = np.random.default_rng(1)
        av = rng.random((20, 8))
        once = sphere_normalize(av)
        twice = sphere_normalize(once)
        np.testing.assert_allclose(once, twice, atol=1e-14)

    def test_preserves_direction(self):
        """sphere_normalize(α·v) must equal sphere_normalize(v) for α > 0."""
        rng = np.random.default_rng(2)
        v = rng.random(10)
        out1 = sphere_normalize(v)
        out2 = sphere_normalize(5.0 * v)
        np.testing.assert_allclose(out1, out2, atol=1e-14)

    def test_shape_preserved(self):
        for shape in [(10,), (5, 3), (90, 10), (1, 5)]:
            av = np.random.default_rng(3).random(shape)
            out = sphere_normalize(av)
            assert out.shape == shape

    def test_dtype_is_float(self):
        av = np.array([[1, 2], [3, 4]], dtype=int)
        out = sphere_normalize(av)
        assert out.dtype == np.float64
