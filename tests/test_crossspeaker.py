"""Cross-speaker moving-stanza window regression tests.

These tests pin the behaviour that distinguishes the correct rENA-exact
cross-speaker stanza window from the earlier within-speaker bug. They use
RS.data (multi-speaker conversations), where the two algorithms diverge —
unlike the single-speaker toy dataset, on which they coincide. If
compute_all_avs is ever reverted to within-speaker windowing, the adjacency
test below will fail.
"""

import numpy as np
import pandas as pd
import pytest

from pyena.core.adjacency import (
    compute_all_avs,
    get_rena_pair_order,
    ref_window_df,
)


def _prepare_rsdata(rsdata_raw):
    """Build pyena input keys from raw RS.data.

    Mirrors rENA's unit/conversation definitions:
      unit         = Condition :: UserName
      conversation = Condition :: GroupName
      turn         = row order within each conversation (rENA uses input order)
    """
    df = rsdata_raw.copy()
    df["unit_id"] = df["Condition"].astype(str) + "::" + df["UserName"].astype(str)
    df["conv_id"] = df["Condition"].astype(str) + "::" + df["GroupName"].astype(str)
    df["turn"] = df.groupby("conv_id").cumcount() + 1
    return df


class TestCrossSpeakerAdjacency:
    def test_av_matches_rena_exact(
        self, rsdata_raw, rsdata_codes, rsdata_rena_adjacency
    ):
        """48x15 adjacency must match rENA exactly (integer arithmetic)."""
        df = _prepare_rsdata(rsdata_raw)
        pair_cols = [f"{a} & {b}" for a, b in get_rena_pair_order(rsdata_codes)]

        avs, unit_ids, _ = compute_all_avs(
            df, rsdata_codes,
            unit_col="unit_id", conversation_col="conv_id",
            window_size=4, window_forward=0, binary=True,
        )
        pkg = pd.DataFrame(avs, index=unit_ids, columns=pair_cols)

        ref = rsdata_rena_adjacency[pair_cols]
        pkg = pkg.reindex(ref.index)

        # unit sets and ordering must align
        assert list(pkg.index) == list(ref.index)
        # integer co-occurrence counts → exact equality
        diff = np.abs(pkg.to_numpy() - ref.to_numpy())
        assert diff.max() == 0, (
            f"max abs diff = {diff.max()}; "
            f"{int((diff > 0).sum())} of {diff.size} cells differ"
        )

    def test_steven_z_is_sixteen(
        self, rsdata_raw, rsdata_codes, rsdata_rena_adjacency
    ):
        """Hand-traced anchor: FirstGame::steven z, Data & TC = 16 (cross),
        which would be 17 under the within-speaker bug."""
        df = _prepare_rsdata(rsdata_raw)
        pair_cols = [f"{a} & {b}" for a, b in get_rena_pair_order(rsdata_codes)]
        avs, unit_ids, _ = compute_all_avs(
            df, rsdata_codes,
            unit_col="unit_id", conversation_col="conv_id",
            window_size=4, window_forward=0, binary=True,
        )
        pkg = pd.DataFrame(avs, index=unit_ids, columns=pair_cols)
        val = int(pkg.loc["FirstGame::steven z", "Data & Technical.Constraints"])
        assert val == 16


class TestPairOrder:
    def test_pair_order_is_column_major_not_combinations(self, rsdata_codes):
        """rENA serialises pairs column-major (vector_to_ut), NOT in
        itertools.combinations (row-major) order. Labelling av_ columns with
        combinations silently mislabels every off-diagonal pair."""
        from itertools import combinations

        rena_order = get_rena_pair_order(rsdata_codes)
        combos = list(combinations(rsdata_codes, 2))

        assert rena_order != combos, (
            "get_rena_pair_order must differ from combinations order"
        )
        assert set(rena_order) == set(combos), "same pairs, different order"
        # spot-check the known divergence point (index 2)
        assert rena_order[2] == ("Technical.Constraints", "Performance.Parameters")
        assert combos[2] == ("Data", "Client.and.Consultant.Requests")


class TestForwardWindowGuard:
    def test_forward_window_raises(self, rsdata_codes):
        """window_forward > 0 is not yet validated against rENA and must be
        guarded rather than silently producing unverified output."""
        cm = np.array([[1, 0, 0, 0, 0, 0],
                       [0, 1, 0, 0, 0, 0],
                       [0, 0, 1, 0, 0, 0]], dtype=float)
        with pytest.raises(NotImplementedError):
            ref_window_df(cm, window_size=4, window_forward=2)
            
from pyena.api import ENA

class TestCrossSpeakerSvdE2E:
    """End-to-end: RS.data adjacency -> normalize -> SVD must match rENA
    coordinates. This exercises the full pipeline on multi-speaker data,
    which the toy-data E2E tests cannot (toy data is single-speaker)."""

    def test_svd_coords_match_rena(
        self, rsdata_raw, rsdata_codes, rsdata_rena_svd, align_sign
    ):
        df = _prepare_rsdata(rsdata_raw)
        ena = ENA(
            codes=rsdata_codes,
            unit_col="unit_id",
            conversation_col="conv_id",
            window_size=4,
            rotation="svd",
        ).fit(df)

        # rENA ground-truth coords, first 2 dims, aligned to pyena unit order
        ref = rsdata_rena_svd.loc[list(ena.unit_ids_), ["SVD1", "SVD2"]].to_numpy()
        coords_aligned = align_sign(ena.coords_, ref)
        diff = np.abs(coords_aligned - ref).max()
        assert diff < 1e-9, f"max coord diff = {diff}"
        
class TestCrossSpeakerMrE2E:
    """End-to-end means rotation on RS.data (FirstGame vs SecondGame).
    The MR1 axis is defined by the group-mean difference, so the 'group'
    column and mr_groups must match what rENA rotated by."""

    def test_mr_coords_match_rena(
        self, rsdata_raw, rsdata_codes, rsdata_rena_mr, align_sign
    ):
        df = _prepare_rsdata(rsdata_raw)
        df["group"] = df["Condition"]   # MR groups keyed on Condition

        ena = ENA(
            codes=rsdata_codes,
            unit_col="unit_id",
            conversation_col="conv_id",
            window_size=4,
            rotation="means",
            mr_groups=("FirstGame", "SecondGame"),
        ).fit(df)

        # rENA MR coords: first 2 dims are MR1, SVD2
        ref = rsdata_rena_mr.loc[list(ena.unit_ids_), ["MR1", "SVD2"]].to_numpy()
        coords_aligned = align_sign(ena.coords_, ref)
        diff = np.abs(coords_aligned - ref).max()
        assert diff < 1e-9, f"max coord diff = {diff}"
        
class TestNodePlacement:
    """rENA-exact least-squares node placement (lws_lsq_positions)."""

    def test_node_positions_match_rena(
        self, rsdata_raw, rsdata_codes, rsdata_rena_nodes
    ):
        df = _prepare_rsdata(rsdata_raw)
        ena = ENA(
            codes=rsdata_codes,
            unit_col="unit_id",
            conversation_col="conv_id",
            window_size=4,
            rotation="svd",
        ).fit(df)

        # rENA node positions, first 2 dims, in code order
        ref = rsdata_rena_nodes[["SVD1", "SVD2"]].to_numpy()
        assert ena.nodes_.shape == ref.shape
        diff = np.abs(ena.nodes_ - ref).max()
        assert diff < 1e-9, f"max node position diff = {diff}"