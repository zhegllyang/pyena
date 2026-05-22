"""Tests for pyena.core.adjacency.

The reference target is rENA 0.3.1: the same toy dataset must yield bit-for-bit
identical adjacency vectors. Additional tests check mathematical properties of
``vector_to_ut`` and ``ref_window_df`` independently of the reference fixture.
"""

import numpy as np
import pytest

from pyena.core.adjacency import (
    compute_all_avs,
    get_rena_pair_order,
    ref_window_df,
    vector_to_ut,
)


# ===== Reference-fixture regression (the headline guarantee) =====

class TestRenaParity:
    """Adjacency vectors must match rENA 0.3.1 bit-for-bit."""

    def test_compute_all_avs_matches_rena_exact(self, toy_data, codes, rena_reference):
        avs, unit_ids, _ = compute_all_avs(
            toy_data, codes,
            unit_col="unit", conversation_col="conversation",
            window_size=4, window_forward=0, binary=True,
        )
        assert avs.shape == rena_reference["pyena_avs"].shape
        assert list(unit_ids) == list(rena_reference["unit_ids"])
        diff = np.abs(avs - rena_reference["pyena_avs"])
        assert diff.max() == 0.0, f"max abs diff = {diff.max():.3e}"

    def test_compute_all_avs_matches_rena_output(self, toy_data, codes, rena_reference):
        """Same call must also match rENA's own output (not just our pyena baseline)."""
        avs, _, _ = compute_all_avs(
            toy_data, codes, window_size=4, binary=True,
        )
        diff = np.abs(avs - rena_reference["rena_avs"])
        assert diff.max() == 0.0


# ===== vector_to_ut properties =====

class TestVectorToUt:
    def test_length(self):
        for n in (2, 3, 5, 8):
            v = np.arange(1, n + 1, dtype=float)
            out = vector_to_ut(v)
            assert out.shape == (n * (n - 1) // 2,)

    def test_known_value_n3(self):
        # For v = [a, b, c], output is [a*b, a*c, b*c]
        v = np.array([2.0, 3.0, 5.0])
        out = vector_to_ut(v)
        np.testing.assert_array_equal(out, [6.0, 10.0, 15.0])

    def test_known_value_n5(self):
        # Manually compute strict UT of [1,2,3,4,5] in rENA's column-major order
        v = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        out = vector_to_ut(v)
        expected = np.array([
            1*2,                # (0,1)
            1*3, 2*3,           # (0,2), (1,2)
            1*4, 2*4, 3*4,      # (0,3), (1,3), (2,3)
            1*5, 2*5, 3*5, 4*5, # (0,4), (1,4), (2,4), (3,4)
        ], dtype=float)
        np.testing.assert_array_equal(out, expected)

    def test_zeros_yield_zeros(self):
        out = vector_to_ut(np.zeros(5))
        np.testing.assert_array_equal(out, np.zeros(10))

    def test_pair_order_matches_codes(self):
        codes = ["A", "B", "C", "D"]
        pairs = get_rena_pair_order(codes)
        assert pairs == [
            ("A", "B"),
            ("A", "C"), ("B", "C"),
            ("A", "D"), ("B", "D"), ("C", "D"),
        ]


# ===== ref_window_df properties =====

class TestRefWindowDf:
    def test_output_shape(self):
        cm = np.random.RandomState(0).randint(0, 2, size=(20, 5))
        out = ref_window_df(cm, window_size=4)
        assert out.shape == (20, 10)  # 5*(5-1)/2 = 10

    def test_binary_output_is_zero_or_one(self):
        cm = np.random.RandomState(0).randint(0, 2, size=(30, 5))
        out = ref_window_df(cm, window_size=7, binary=True)
        assert set(np.unique(out)).issubset({0.0, 1.0})

    def test_all_codes_always_present_gives_full_cooccurrence(self):
        """If every utterance activates every code, every pair co-occurs in every window.

        Note: rENA counts within-utterance co-occurrence — two codes appearing
        in the *same* row of the window already form a pair. So even a
        single-row window with multiple codes produces nonzero output.
        """
        cm = np.ones((10, 4), dtype=int)
        out = ref_window_df(cm, window_size=4, binary=True)
        np.testing.assert_array_equal(out, np.ones((10, 6)))

    def test_single_code_only_gives_zero_pairs(self):
        """If only one code is ever active, no pair can co-occur."""
        cm = np.zeros((10, 4), dtype=int)
        cm[:, 0] = 1  # only code 0 is ever active
        out = ref_window_df(cm, window_size=4, binary=True)
        np.testing.assert_array_equal(out, np.zeros((10, 6)))

    def test_window_does_not_cross_array_start(self):
        """The first row uses a truncated window (no rows before it)."""
        cm = np.zeros((10, 3), dtype=int)
        # Activate codes 0 and 1 only on row 5; codes 1 and 2 only on row 6.
        cm[5, [0, 1]] = 1
        cm[6, [1, 2]] = 1
        out = ref_window_df(cm, window_size=4, binary=True)
        # Rows 0-4 are all zero (nothing activated yet)
        np.testing.assert_array_equal(out[:5], np.zeros((5, 3)))
        # Row 5: only (code0, code1) co-occurs
        # rENA pair order for 3 codes: (0,1), (0,2), (1,2)
        np.testing.assert_array_equal(out[5], [1, 0, 0])


# ===== compute_all_avs invariants =====

class TestComputeAllAvs:
    def test_group_ids_populated_when_group_col_present(self, toy_data, codes):
        _, unit_ids, group_ids = compute_all_avs(toy_data, codes)
        assert len(group_ids) == len(unit_ids)
        assert set(group_ids) == {"A_tight", "A_loose", "B_tight"}

    def test_group_ids_empty_when_no_group_col(self, toy_data, codes):
        df_no_group = toy_data.drop(columns=["group"])
        _, _, group_ids = compute_all_avs(df_no_group, codes)
        assert group_ids == []

    def test_binary_output_is_nonnegative_integers_when_binary(self, toy_data, codes):
        avs, _, _ = compute_all_avs(toy_data, codes, binary=True)
        assert (avs >= 0).all()
        # binary=True에 unit-level summation이라 정수값 (각 row가 0/1, 합산하면 정수)
        assert np.all(avs == avs.astype(int))
