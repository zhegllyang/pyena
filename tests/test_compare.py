"""Tests for pyena.stats.compare."""

import numpy as np
import pytest
from scipy import stats as scipy_stats

from pyena.stats.compare import ena_group_comparison


# ===== Reference regression =====

class TestCompareReference:
    def test_x_axis_matches_notebook_reference(self, compare_reference):
        pkg = ena_group_comparison(
            compare_reference["input_points"],
            compare_reference["input_labels"],
            axis="x",
            n_permutations=compare_reference["n_permutations"],
            random_state=compare_reference["random_state"],
        )
        ref = compare_reference["result_x"]
        for k, v in ref.items():
            if isinstance(v, (int, float, np.integer, np.floating)):
                assert pkg[k] == pytest.approx(v, rel=1e-9, abs=1e-12), f"mismatch in {k}"
            else:
                assert pkg[k] == v, f"mismatch in {k}"

    def test_y_axis_matches_notebook_reference(self, compare_reference):
        pkg = ena_group_comparison(
            compare_reference["input_points"],
            compare_reference["input_labels"],
            axis="y",
            n_permutations=compare_reference["n_permutations"],
            random_state=compare_reference["random_state"],
        )
        ref = compare_reference["result_y"]
        for k, v in ref.items():
            if isinstance(v, (int, float, np.integer, np.floating)):
                assert pkg[k] == pytest.approx(v, abs=0, rel=0), f"mismatch in {k}"
            else:
                assert pkg[k] == v


# ===== scipy external cross-check =====

class TestScipyCrossCheck:
    """The Welch and Mann-Whitney values must match scipy directly."""

    def test_welch_matches_scipy(self, compare_reference):
        pts = compare_reference["input_points"]
        labels = compare_reference["input_labels"]
        groups = np.unique(labels)
        for axis, axis_idx in [("x", 0), ("y", 1)]:
            g1 = pts[labels == groups[0], axis_idx]
            g2 = pts[labels == groups[1], axis_idx]
            t_scipy, p_scipy = scipy_stats.ttest_ind(g1, g2, equal_var=False)
            pkg = ena_group_comparison(pts, labels, axis=axis, random_state=0,
                                       n_permutations=1)  # perms don't matter here
            assert pkg["welch_t"] == t_scipy
            assert pkg["welch_p"] == p_scipy

    def test_mannwhitney_matches_scipy(self, compare_reference):
        pts = compare_reference["input_points"]
        labels = compare_reference["input_labels"]
        groups = np.unique(labels)
        for axis, axis_idx in [("x", 0), ("y", 1)]:
            g1 = pts[labels == groups[0], axis_idx]
            g2 = pts[labels == groups[1], axis_idx]
            u_scipy, p_scipy = scipy_stats.mannwhitneyu(g1, g2, alternative="two-sided")
            pkg = ena_group_comparison(pts, labels, axis=axis, random_state=0,
                                       n_permutations=1)
            assert pkg["mannwhitney_U"] == u_scipy
            assert pkg["mannwhitney_p"] == p_scipy


# ===== Algorithmic properties =====

class TestProperties:
    def test_raises_on_more_than_two_groups(self):
        pts = np.random.default_rng(0).random((30, 2))
        labels = np.array(["A"] * 10 + ["B"] * 10 + ["C"] * 10)
        with pytest.raises(ValueError, match="Exactly two groups"):
            ena_group_comparison(pts, labels)

    def test_raises_on_single_group(self):
        pts = np.random.default_rng(0).random((10, 2))
        labels = np.array(["A"] * 10)
        with pytest.raises(ValueError, match="Exactly two groups"):
            ena_group_comparison(pts, labels)

    def test_small_sample_emits_warning(self):
        pts = np.array([[1.0, 0], [2.0, 0], [3.0, 0],
                        [10.0, 0], [11.0, 0], [12.0, 0]])
        labels = np.array(["A", "A", "A", "B", "B", "B"])
        with pytest.warns(UserWarning, match="Small sample warning"):
            ena_group_comparison(pts, labels, n_permutations=100, random_state=0)

    def test_permutation_p_value_in_unit_interval(self):
        rng = np.random.default_rng(0)
        pts = rng.random((40, 2))
        labels = np.array(["A"] * 20 + ["B"] * 20)
        result = ena_group_comparison(pts, labels, n_permutations=500, random_state=0)
        assert 0.0 <= result["permutation_p"] <= 1.0

    def test_permutation_p_reproducible_with_seed(self):
        rng = np.random.default_rng(0)
        pts = rng.random((40, 2))
        labels = np.array(["A"] * 20 + ["B"] * 20)
        r1 = ena_group_comparison(pts, labels, n_permutations=500, random_state=42)
        r2 = ena_group_comparison(pts, labels, n_permutations=500, random_state=42)
        assert r1["permutation_p"] == r2["permutation_p"]

    def test_identical_distributions_give_large_p_values(self):
        rng = np.random.default_rng(0)
        pts = rng.standard_normal((100, 2))
        labels = np.array(["A"] * 50 + ["B"] * 50)
        result = ena_group_comparison(pts, labels, n_permutations=500, random_state=0)
        # Under H0, all three p-values should be reasonably large
        assert result["welch_p"] > 0.1
        assert result["mannwhitney_p"] > 0.1
        assert result["permutation_p"] > 0.1

    def test_cohens_d_sign_matches_observed_diff_sign(self):
        rng = np.random.default_rng(0)
        pts = rng.random((20, 2))
        labels = np.array(["A"] * 10 + ["B"] * 10)
        result = ena_group_comparison(pts, labels, n_permutations=100, random_state=0)
        assert np.sign(result["cohens_d"]) == np.sign(result["observed_diff"])

    def test_axis_x_and_y_use_different_columns(self):
        pts = np.column_stack([
            np.concatenate([np.zeros(10), np.ones(10)]),       # huge x-axis diff
            np.random.default_rng(0).standard_normal(20),      # noise on y
        ])
        labels = np.array(["A"] * 10 + ["B"] * 10)
        res_x = ena_group_comparison(pts, labels, axis="x",
                                     n_permutations=100, random_state=0)
        res_y = ena_group_comparison(pts, labels, axis="y",
                                     n_permutations=100, random_state=0)
        assert res_x["welch_p"] < res_y["welch_p"]
