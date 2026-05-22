"""Tests for the ENA estimator (pyena.api)."""

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pytest

from pyena import ENA


# ===== End-to-end regression =====

class TestSvdScenario:
    """ENA(rotation='svd') must reproduce every validated reference."""

    @pytest.fixture(scope="class")
    def ena(self, toy_data, codes):
        return ENA(codes=codes, window_size=4, rotation="svd").fit(toy_data)

    def test_av_matches_rena_exact(self, ena, rena_reference):
        diff = np.abs(ena.av_ - rena_reference["pyena_avs"]).max()
        assert diff == 0.0

    def test_unit_ids_match_reference(self, ena, rena_reference):
        assert list(ena.unit_ids_) == list(rena_reference["unit_ids"])

    def test_fitted_attributes_present(self, ena):
        for attr in ["av_", "av_normalized_", "av_centered_", "mean_",
                     "components_", "singular_values_", "coords_",
                     "unit_ids_", "group_ids_", "codes_", "n_features_in_"]:
            assert hasattr(ena, attr), f"missing fitted attribute: {attr}"
        assert ena.rotation_matrix_ is None
        assert ena.mr_groups_used_ is None

    def test_shapes_consistent(self, ena):
        n_units = len(ena.unit_ids_)
        n_pairs = ena.n_features_in_
        assert ena.av_.shape == (n_units, n_pairs)
        assert ena.av_normalized_.shape == (n_units, n_pairs)
        assert ena.av_centered_.shape == (n_units, n_pairs)
        assert ena.mean_.shape == (n_pairs,)
        assert ena.coords_.shape == (n_units, 2)
        assert ena.components_.shape == (2, n_pairs)
        assert ena.singular_values_.shape == (2,)


class TestMeansScenario:
    """ENA(rotation='means') must reproduce the rENA MR validation."""

    @pytest.fixture(scope="class")
    def ena(self, toy_data, codes):
        return ENA(
            codes=codes, window_size=4,
            rotation="means", mr_groups=("A_tight", "B_tight"),
        ).fit(toy_data)

    def test_mr_coords_match_rena(self, ena, mr_reference, align_sign):
        coords_aligned = align_sign(ena.coords_, mr_reference["rena_mr_points"])
        diff = np.abs(coords_aligned - mr_reference["rena_mr_points"]).max()
        assert diff < 1e-10

    def test_mr_specific_attributes(self, ena):
        assert ena.singular_values_ is None
        assert ena.rotation_matrix_ is not None
        assert ena.rotation_matrix_.shape == (ena.n_features_in_, ena.n_features_in_)
        assert ena.mr_groups_used_ == ("A_tight", "B_tight")


# ===== sklearn-style API contract =====

class TestSklearnApi:
    def test_fit_returns_self(self, toy_data, codes):
        ena = ENA(codes=codes, window_size=4)
        assert ena.fit(toy_data) is ena

    def test_transform_requires_fit(self, toy_data, codes):
        ena = ENA(codes=codes, window_size=4)
        with pytest.raises(RuntimeError, match="fit"):
            ena.transform(toy_data)

    def test_transform_on_training_data_matches_coords(self, toy_data, codes):
        ena = ENA(codes=codes, window_size=4).fit(toy_data)
        coords_via_transform = ena.transform(toy_data)
        diff = np.abs(coords_via_transform - ena.coords_).max()
        assert diff < 1e-10

    def test_fit_transform_equals_fit_then_coords(self, toy_data, codes):
        ena1 = ENA(codes=codes, window_size=4).fit(toy_data)
        ena2 = ENA(codes=codes, window_size=4)
        coords2 = ena2.fit_transform(toy_data)
        np.testing.assert_array_equal(coords2, ena1.coords_)

    def test_get_params_round_trip(self, codes):
        ena = ENA(codes=codes, window_size=7, n_components=3,
                  rotation="means", mr_groups=("A", "B"))
        params = ena.get_params()
        assert params["window_size"] == 7
        assert params["n_components"] == 3
        assert params["rotation"] == "means"
        assert params["mr_groups"] == ("A", "B")

    def test_set_params(self, codes):
        ena = ENA(codes=codes, window_size=4)
        ena.set_params(window_size=8, rotation="means", mr_groups=("X", "Y"))
        assert ena.window_size == 8
        assert ena.rotation == "means"


# ===== Parameter validation =====

class TestParameterValidation:
    def test_invalid_rotation_raises(self, toy_data, codes):
        ena = ENA(codes=codes, rotation="invalid")
        with pytest.raises(ValueError, match="rotation must be"):
            ena.fit(toy_data)

    def test_means_without_mr_groups_raises(self, toy_data, codes):
        ena = ENA(codes=codes, rotation="means", mr_groups=None)
        with pytest.raises(ValueError, match="mr_groups"):
            ena.fit(toy_data)

    def test_means_without_group_column_raises(self, toy_data, codes):
        df_no_group = toy_data.drop(columns=["group"])
        ena = ENA(codes=codes, rotation="means", mr_groups=("A_tight", "B_tight"))
        with pytest.raises(ValueError, match="group"):
            ena.fit(df_no_group)


# ===== Convenience methods =====

class TestConvenienceMethods:
    @pytest.fixture(scope="class")
    def ena_mr(self, toy_data, codes):
        return ENA(
            codes=codes, window_size=4,
            rotation="means", mr_groups=("A_tight", "B_tight"),
        ).fit(toy_data)

    def test_compare_uses_mr_groups_by_default(self, ena_mr, compare_reference):
        result = ena_mr.compare(axis="x", random_state=42, n_permutations=10000)
        ref = compare_reference["result_x"]
        for k, v in ref.items():
            if isinstance(v, (int, float, np.integer, np.floating)):
                assert result[k] == v, f"mismatch in {k}"

    def test_compare_with_explicit_groups(self, ena_mr):
        result = ena_mr.compare(
            axis="x", groups=("A_tight", "A_loose"),
            n_permutations=100, random_state=0,
        )
        assert result["n1"] == 30
        assert result["n2"] == 30

    def test_compare_raises_when_three_groups_and_no_mr(self, toy_data, codes):
        ena = ENA(codes=codes, rotation="svd").fit(toy_data)
        with pytest.raises(ValueError, match="More than two groups"):
            ena.compare(axis="x")

    def test_reproducibility_returns_one_row_per_group(self, ena_mr):
        df = ena_mr.reproducibility()
        assert set(df.index) == {"A_tight", "A_loose", "B_tight"}

    def test_plot_returns_axes(self, ena_mr):
        ax = ena_mr.plot()
        assert ax is not None
        plt.close(ax.figure)


# ===== Custom column names =====

class TestColumnRenaming:
    def test_custom_group_col_works(self, toy_data, codes):
        df_renamed = toy_data.rename(columns={"group": "condition"})
        ena = ENA(
            codes=codes, group_col="condition",
            rotation="means", mr_groups=("A_tight", "B_tight"),
        ).fit(df_renamed)
        assert ena.mr_groups_used_ == ("A_tight", "B_tight")
        assert set(ena.group_ids_) == {"A_tight", "A_loose", "B_tight"}
