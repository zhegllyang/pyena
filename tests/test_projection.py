"""Tests for pyena.core.projection."""

import numpy as np
import pytest

from pyena.core.projection import (
    MeansRotationResult,
    ProjectionResult,
    means_rotation,
    orthogonal_svd,
    svd_project,
)


# ===== Reference regression =====

class TestSvdProjectReference:
    def test_matches_pyena_reference(self, pyena_reference, align_sign):
        result = svd_project(pyena_reference["normalized_matrix"], n_components=2)

        # centered_matrix and singular_values must match exactly
        np.testing.assert_array_equal(
            result.centered, pyena_reference["centered_matrix"]
        )
        np.testing.assert_allclose(
            result.singular_values, pyena_reference["singular_values"][:2],
            atol=1e-12, rtol=1e-12,
        )

        # coords and components must match up to sign
        coords_aligned = align_sign(result.coords, pyena_reference["points_2d"])
        np.testing.assert_allclose(
            coords_aligned, pyena_reference["points_2d"], atol=1e-14
        )

        ref_Vt = pyena_reference["Vt"][:2]
        components = result.components.copy()
        for k in range(2):
            if np.dot(components[k], ref_Vt[k]) < 0:
                components[k] = -components[k]
        np.testing.assert_allclose(components, ref_Vt, atol=1e-14)

    def test_returns_projection_result(self, pyena_reference):
        result = svd_project(pyena_reference["normalized_matrix"], n_components=2)
        assert isinstance(result, ProjectionResult)


class TestMeansRotationReference:
    def test_matches_rena_means_rotation(self, mr_reference, align_sign):
        result = means_rotation(
            mr_reference["centered_input"],
            mr_reference["group_labels"],
            group_a="A_tight", group_b="B_tight",
        )
        # First two columns are MR1, SVD2
        coords_aligned = align_sign(result.points[:, :2], mr_reference["rena_mr_points"])
        diff = np.abs(coords_aligned - mr_reference["rena_mr_points"])
        # rENA-exact verified at machine epsilon
        assert diff.max() < 1e-10, f"max abs diff = {diff.max():.3e}"

    def test_returns_means_rotation_result(self, mr_reference):
        result = means_rotation(
            mr_reference["centered_input"],
            mr_reference["group_labels"],
            group_a="A_tight", group_b="B_tight",
        )
        assert isinstance(result, MeansRotationResult)
        assert result.groups_used == ("A_tight", "B_tight")


# ===== svd_project properties =====

class TestSvdProject:
    def test_singular_values_are_sorted_descending(self):
        rng = np.random.default_rng(0)
        M = rng.random((50, 10))
        result = svd_project(M, n_components=5)
        sv = result.singular_values
        assert all(sv[i] >= sv[i + 1] for i in range(len(sv) - 1))

    def test_components_are_orthonormal(self):
        rng = np.random.default_rng(1)
        M = rng.random((50, 10))
        result = svd_project(M, n_components=5)
        # components are rows of Vt[:k], should be orthonormal
        gram = result.components @ result.components.T
        np.testing.assert_allclose(gram, np.eye(5), atol=1e-12)

    def test_centered_has_zero_mean(self):
        rng = np.random.default_rng(2)
        M = rng.random((50, 10))
        result = svd_project(M, n_components=2)
        np.testing.assert_allclose(result.centered.mean(axis=0), 0, atol=1e-14)

    def test_coords_equal_centered_times_components_T(self):
        rng = np.random.default_rng(3)
        M = rng.random((50, 10))
        result = svd_project(M, n_components=3)
        expected = result.centered @ result.components.T
        np.testing.assert_allclose(result.coords, expected, atol=1e-14)

    def test_n_components_parameter_respected(self):
        rng = np.random.default_rng(4)
        M = rng.random((30, 10))
        for k in (1, 2, 5, 10):
            result = svd_project(M, n_components=k)
            assert result.coords.shape == (30, k)
            assert result.components.shape == (k, 10)


# ===== orthogonal_svd properties =====

class TestOrthogonalSvd:
    def test_first_columns_equal_weights(self):
        """The first k columns of the output must coincide with weights (up to QR's sign)."""
        rng = np.random.default_rng(0)
        data = rng.random((20, 5))
        weights = rng.random((5, 1))
        weights /= np.linalg.norm(weights, axis=0)
        rotation = orthogonal_svd(data, weights)
        # QR may flip sign; compare absolute alignment
        align = abs(np.dot(rotation[:, 0], weights[:, 0]))
        assert align == pytest.approx(1.0, abs=1e-12)

    def test_rotation_is_orthonormal(self):
        rng = np.random.default_rng(1)
        data = rng.random((20, 5))
        weights = rng.random((5, 1))
        weights /= np.linalg.norm(weights, axis=0)
        rotation = orthogonal_svd(data, weights)
        np.testing.assert_allclose(rotation @ rotation.T, np.eye(5), atol=1e-12)


# ===== means_rotation properties =====

class TestMeansRotation:
    def test_mr1_axis_separates_groups(self, mr_reference):
        """The first axis must separate the two groups used for rotation."""
        result = means_rotation(
            mr_reference["centered_input"],
            mr_reference["group_labels"],
            group_a="A_tight", group_b="B_tight",
        )
        labels = mr_reference["group_labels"]
        mr1 = result.points[:, 0]
        # The two group centroids on MR1 must be far apart
        mean_a = mr1[labels == "A_tight"].mean()
        mean_b = mr1[labels == "B_tight"].mean()
        assert abs(mean_a - mean_b) > 0.5  # toy data is well-separated

    def test_rotation_is_orthonormal(self, mr_reference):
        result = means_rotation(
            mr_reference["centered_input"],
            mr_reference["group_labels"],
            group_a="A_tight", group_b="B_tight",
        )
        R = result.rotation
        np.testing.assert_allclose(R @ R.T, np.eye(R.shape[0]), atol=1e-12)

    def test_raises_when_group_missing(self, mr_reference):
        with pytest.raises(ValueError, match="must both be present"):
            means_rotation(
                mr_reference["centered_input"],
                mr_reference["group_labels"],
                group_a="A_tight", group_b="NONEXISTENT",
            )

    def test_raises_when_group_means_identical(self):
        """If both groups have identical centroids, MR1 is undefined."""
        M = np.array([[1.0, 0.0], [1.0, 0.0], [1.0, 0.0], [1.0, 0.0]])
        labels = np.array(["A", "A", "B", "B"])
        with pytest.raises(ValueError, match="undefined"):
            means_rotation(M, labels, group_a="A", group_b="B")
