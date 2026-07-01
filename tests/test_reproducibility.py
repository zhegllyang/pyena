"""Tests for pyena.stats.reproducibility."""

import numpy as np
import pytest
from scipy.spatial import ConvexHull
from scipy.spatial.distance import pdist
from scipy.stats import chi2

from pyena.stats.reproducibility import reproducibility_metrics


# ===== Reference regression =====

class TestReproducibilityReference:
    def test_grouped_matches_notebook_reference(self, reproducibility_reference):
        pkg = reproducibility_metrics(
            reproducibility_reference["input_points"],
            reproducibility_reference["input_labels"],
        )
        ref = reproducibility_reference["result_grouped"]
        assert list(pkg.index) == list(ref.index)
        assert list(pkg.columns) == list(ref.columns)
        pkg_num = pkg.astype(float).to_numpy()
        ref_num = ref.astype(float).to_numpy()
        diff = np.abs(pkg_num - ref_num)
        diff[np.isnan(pkg_num) & np.isnan(ref_num)] = 0
        assert diff.max() < 1e-9

    def test_all_matches_notebook_reference(self, reproducibility_reference):
        pkg = reproducibility_metrics(reproducibility_reference["input_points"], None)
        ref = reproducibility_reference["result_all"]
        assert list(pkg.index) == ["all"]
        diff = np.abs(pkg.astype(float).to_numpy() - ref.astype(float).to_numpy())
        assert diff.max() < 1e-9


# ===== Individual metric cross-checks against standard implementations =====

class TestMetricFormulas:
    """Each metric must match a direct implementation of its definition."""

    @pytest.fixture
    def sample(self, reproducibility_reference):
        labels = reproducibility_reference["input_labels"]
        pts = reproducibility_reference["input_points"][labels == "A_tight"]
        return pts

    def test_centroid_dispersion_formula(self, sample):
        df = reproducibility_metrics(sample, None)
        centroid = sample.mean(axis=0)
        d = np.linalg.norm(sample - centroid, axis=1)
        assert df.loc["all", "centroid_dispersion_mean"] == d.mean()
        assert df.loc["all", "centroid_dispersion_std"] == d.std()

    def test_pairwise_distance_formula(self, sample):
        df = reproducibility_metrics(sample, None)
        pw = pdist(sample)
        assert df.loc["all", "pairwise_distance_mean"] == pw.mean()
        assert df.loc["all", "pairwise_distance_max"] == pw.max()

    def test_ellipse_area_formula(self, sample):
        df = reproducibility_metrics(sample, None)
        cov = np.cov(sample.T)
        eig = np.linalg.eigvalsh(cov)
        expected = np.pi * chi2.ppf(0.95, df=2) * np.sqrt(eig[0] * eig[1])
        assert df.loc["all", "ellipse_95_area"] == expected

    def test_convex_hull_formula(self, sample):
        df = reproducibility_metrics(sample, None)
        hull = ConvexHull(sample)
        assert df.loc["all", "convex_hull_area"] == hull.volume


# ===== Properties / edge cases =====

class TestProperties:
    def test_zero_dispersion_for_identical_points(self):
        pts = np.tile([1.0, 2.0], (10, 1))
        df = reproducibility_metrics(pts, None)
        assert df.loc["all", "centroid_dispersion_mean"] == 0.0
        assert df.loc["all", "pairwise_distance_mean"] == 0.0
        assert df.loc["all", "pairwise_distance_max"] == 0.0
        # cov is zero → ellipse area is 0
        assert df.loc["all", "ellipse_95_area"] == 0.0

    def test_dispersion_scales_with_spread(self):
        rng = np.random.default_rng(0)
        base = rng.standard_normal((30, 2))
        tight = reproducibility_metrics(base, None).loc["all", "centroid_dispersion_mean"]
        loose = reproducibility_metrics(5 * base, None).loc["all", "centroid_dispersion_mean"]
        # 5x spread should give exactly 5x dispersion
        assert loose == pytest.approx(5 * tight, rel=1e-12)

    def test_single_point_returns_nan_for_size_dependent_metrics(self):
        pts = np.array([[1.0, 2.0]])
        df = reproducibility_metrics(pts, None)
        assert df.loc["all", "n"] == 1
        assert df.loc["all", "centroid_dispersion_mean"] == 0.0
        assert np.isnan(df.loc["all", "pairwise_distance_mean"])
        assert np.isnan(df.loc["all", "ellipse_95_area"])
        assert np.isnan(df.loc["all", "convex_hull_area"])

    def test_two_points_have_pairwise_but_no_ellipse(self):
        pts = np.array([[0.0, 0.0], [3.0, 4.0]])
        df = reproducibility_metrics(pts, None)
        assert df.loc["all", "pairwise_distance_mean"] == 5.0
        assert df.loc["all", "pairwise_distance_max"] == 5.0
        assert np.isnan(df.loc["all", "ellipse_95_area"])
        assert np.isnan(df.loc["all", "convex_hull_area"])

    def test_grouped_returns_one_row_per_group(self, reproducibility_reference):
        df = reproducibility_metrics(
            reproducibility_reference["input_points"],
            reproducibility_reference["input_labels"],
        )
        assert set(df.index) == {"A_loose", "A_tight", "B_tight"}
        assert len(df) == 3

    def test_columns_are_consistent(self, reproducibility_reference):
        df_g = reproducibility_metrics(
            reproducibility_reference["input_points"],
            reproducibility_reference["input_labels"],
        )
        df_a = reproducibility_metrics(reproducibility_reference["input_points"], None)
        assert list(df_g.columns) == list(df_a.columns)

    def test_toy_data_loose_is_more_dispersed_than_tight(self, reproducibility_reference):
        """Sanity check: A_loose has 0.20 jitter, A_tight has 0.03 → looser."""
        df = reproducibility_metrics(
            reproducibility_reference["input_points"],
            reproducibility_reference["input_labels"],
        )
        for metric in ["centroid_dispersion_mean", "pairwise_distance_mean",
                       "ellipse_95_area", "convex_hull_area"]:
            assert df.loc["A_loose", metric] > df.loc["A_tight", metric], (
                f"A_loose should be more dispersed than A_tight in {metric}"
            )
