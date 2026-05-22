"""Tests for pyena.viz.plots."""

import matplotlib
matplotlib.use("Agg")  # headless backend for CI
import matplotlib.pyplot as plt
import numpy as np
import pytest
from matplotlib.patches import Ellipse
from scipy.stats import chi2

from pyena.viz.plots import plot_reproducibility


def _extract_structure(ax):
    """Pull scatter coords, centroid coords, and ellipse params from an Axes."""
    scatters, centroids, ellipses = [], [], []
    for coll in ax.collections:
        offsets = coll.get_offsets()
        if len(offsets) == 1:
            centroids.append(np.array(offsets[0]))
        else:
            scatters.append(np.array(offsets))
    for patch in ax.patches:
        if isinstance(patch, Ellipse):
            ellipses.append({
                "center": np.array(patch.get_center()),
                "width": patch.get_width(),
                "height": patch.get_height(),
                "angle": patch.angle,
            })
    return scatters, centroids, ellipses


# ===== Figure structure =====

class TestStructure:
    def test_returns_axes(self, mr_reference):
        fig, ax = plt.subplots()
        out = plot_reproducibility(
            mr_reference["pyena_mr_points"], mr_reference["group_labels"], ax=ax,
        )
        assert out is ax
        plt.close(fig)

    def test_one_scatter_per_group(self, mr_reference):
        fig, ax = plt.subplots()
        plot_reproducibility(
            mr_reference["pyena_mr_points"], mr_reference["group_labels"], ax=ax,
        )
        scatters, centroids, ellipses = _extract_structure(ax)
        n_groups = len(np.unique(mr_reference["group_labels"]))
        assert len(scatters) == n_groups
        assert len(centroids) == n_groups
        assert len(ellipses) == n_groups
        plt.close(fig)

    def test_scatter_count_per_group_correct(self, mr_reference):
        fig, ax = plt.subplots()
        plot_reproducibility(
            mr_reference["pyena_mr_points"], mr_reference["group_labels"], ax=ax,
        )
        scatters, _, _ = _extract_structure(ax)
        labels = mr_reference["group_labels"]
        for g, sc in zip(np.unique(labels), scatters):
            expected_n = (labels == g).sum()
            assert len(sc) == expected_n
        plt.close(fig)

    def test_ellipse_not_drawn_when_show_ellipse_false(self, mr_reference):
        fig, ax = plt.subplots()
        plot_reproducibility(
            mr_reference["pyena_mr_points"], mr_reference["group_labels"],
            ax=ax, show_ellipse=False,
        )
        _, _, ellipses = _extract_structure(ax)
        assert len(ellipses) == 0
        plt.close(fig)

    def test_creates_new_figure_when_ax_is_none(self, mr_reference):
        ax = plot_reproducibility(
            mr_reference["pyena_mr_points"], mr_reference["group_labels"],
        )
        assert ax is not None
        plt.close(ax.figure)


# ===== Ellipse math =====

class TestEllipseMath:
    def test_ellipse_center_equals_group_centroid(self, mr_reference):
        fig, ax = plt.subplots()
        plot_reproducibility(
            mr_reference["pyena_mr_points"], mr_reference["group_labels"], ax=ax,
        )
        _, centroids, ellipses = _extract_structure(ax)
        for c, el in zip(centroids, ellipses):
            np.testing.assert_array_equal(el["center"], c)
        plt.close(fig)

    def test_ellipse_dimensions_match_formula(self, mr_reference):
        fig, ax = plt.subplots()
        plot_reproducibility(
            mr_reference["pyena_mr_points"], mr_reference["group_labels"], ax=ax,
        )
        _, _, ellipses = _extract_structure(ax)
        chi2_val = chi2.ppf(0.95, df=2)

        labels = mr_reference["group_labels"]
        for g, el in zip(np.unique(labels), ellipses):
            pts = mr_reference["pyena_mr_points"][labels == g]
            cov = np.cov(pts.T)
            eigvals, eigvecs = np.linalg.eigh(cov)
            order = eigvals.argsort()[::-1]
            eigvals = eigvals[order]
            eigvecs = eigvecs[:, order]

            expected_width = 2 * np.sqrt(chi2_val * eigvals[0])
            expected_height = 2 * np.sqrt(chi2_val * eigvals[1])
            expected_angle = np.degrees(np.arctan2(eigvecs[1, 0], eigvecs[0, 0]))

            assert el["width"] == pytest.approx(expected_width, rel=1e-12)
            assert el["height"] == pytest.approx(expected_height, rel=1e-12)
            assert el["angle"] == pytest.approx(expected_angle, rel=1e-12)
        plt.close(fig)

    def test_ellipse_skipped_for_groups_with_fewer_than_3_points(self):
        pts = np.array([[0.0, 0.0], [1.0, 1.0]])
        labels = np.array(["A", "A"])
        fig, ax = plt.subplots()
        plot_reproducibility(pts, labels, ax=ax)
        _, _, ellipses = _extract_structure(ax)
        assert len(ellipses) == 0
        plt.close(fig)


# ===== Labels and aesthetics =====

class TestLabels:
    def test_default_axis_labels(self, mr_reference):
        fig, ax = plt.subplots()
        plot_reproducibility(
            mr_reference["pyena_mr_points"], mr_reference["group_labels"], ax=ax,
        )
        assert ax.get_xlabel() == "PC1"
        assert ax.get_ylabel() == "PC2"
        plt.close(fig)

    def test_custom_axis_labels(self, mr_reference):
        fig, ax = plt.subplots()
        plot_reproducibility(
            mr_reference["pyena_mr_points"], mr_reference["group_labels"], ax=ax,
            xlabel="MR1", ylabel="SVD2",
        )
        assert ax.get_xlabel() == "MR1"
        assert ax.get_ylabel() == "SVD2"
        plt.close(fig)

    def test_legend_has_one_entry_per_group(self, mr_reference):
        fig, ax = plt.subplots()
        plot_reproducibility(
            mr_reference["pyena_mr_points"], mr_reference["group_labels"], ax=ax,
        )
        legend_texts = [t.get_text() for t in ax.get_legend().get_texts()]
        assert len(legend_texts) == len(np.unique(mr_reference["group_labels"]))
        # Each label should include the group name and "n=" count
        for t in legend_texts:
            assert "n=" in t
        plt.close(fig)
