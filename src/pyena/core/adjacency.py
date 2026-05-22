"""Adjacency vector construction, rENA-exact.

This module mirrors the C++ source of rENA 0.3.1 (``vector_to_ut`` and
``ref_window_df`` in ``rENA/src/ena.cpp``). The implementation has been
verified bit-for-bit against rENA on a 90-unit toy dataset
(900/900 cells match, max absolute difference = 0).

References
----------
.. [1] Marquart, C. L., Swiecki, Z., Collier, W., Eagan, B., Woodward, R.,
       & Shaffer, D. W. (2021). rENA: Epistemic Network Analysis (R package
       version 0.3.1).
"""

from __future__ import annotations

import numpy as np
import pandas as pd

__all__ = [
    "compute_all_avs",
    "get_rena_pair_order",
    "ref_window_df",
    "vector_to_ut",
]


def vector_to_ut(v):
    """Vectorize the strict upper triangle of an outer product.

    Reimplementation of rENA's ``vector_to_ut`` (``ena.cpp`` lines 142-155).
    Given a code-count vector ``v`` of length ``n``, returns the strict upper
    triangle of the outer product ``v v^T`` in column-major order, which is
    the canonical ordering used by rENA throughout the pipeline.

    Parameters
    ----------
    v : array-like of shape (n,)
        Code-count vector (typically the column-sum of a stanza window).

    Returns
    -------
    np.ndarray of shape (n * (n - 1) / 2,)
        Strict upper-triangular entries of ``v v^T`` in rENA's column-major
        ordering.

    Notes
    -----
    The traversal order is::

        for i in range(2, n + 1):
            for j in range(0, i - 1):
                yield v[j] * v[i - 1]

    This matches rENA's C++ implementation exactly. Changing the order would
    break parity with rENA.
    """
    n = len(v)
    out = []
    for i in range(2, n + 1):
        for j in range(0, i - 1):
            out.append(v[j] * v[i - 1])
    return np.array(out, dtype=float)


def get_rena_pair_order(codes):
    """Return the code-pair ordering used by ``vector_to_ut``.

    Parameters
    ----------
    codes : list of str
        Code names in their original order.

    Returns
    -------
    list of tuple of (str, str)
        Code pairs in the same order as the entries returned by
        ``vector_to_ut``. Useful for labelling adjacency-vector columns.
    """
    pairs = []
    n = len(codes)
    for i in range(2, n + 1):
        for j in range(0, i - 1):
            pairs.append((codes[j], codes[i - 1]))
    return pairs


def ref_window_df(code_matrix, window_size=4, window_forward=0, binary=True):
    """Compute per-row adjacency vectors using rENA's moving-stanza window.

    Faithful reimplementation of rENA's ``ref_window_df`` (``ena.cpp`` lines
    205-311), including the head/tail subtraction trick that yields each
    focal row's stanza window co-occurrence without recomputing from scratch.

    Parameters
    ----------
    code_matrix : np.ndarray of shape (n_rows, n_codes)
        Binary code-presence matrix. Each row is one utterance, each column
        one code.
    window_size : int, default 4
        Size of the moving stanza window (focal row + lookback).
    window_forward : int, default 0
        Forward lookahead beyond the focal row. rENA's default is 0.
    binary : bool, default True
        If True, co-occurrence counts are clipped to 0/1 (rENA default).
        If False, raw counts are returned.

    Returns
    -------
    np.ndarray of shape (n_rows, n_pairs)
        Per-row adjacency vectors in rENA pair order, where
        ``n_pairs = n_codes * (n_codes - 1) / 2``.
    """
    df_rows, df_cols = code_matrix.shape
    n_pairs = (df_cols * (df_cols + 1)) // 2 - df_cols
    df_co_occurred = np.zeros((df_rows, n_pairs))

    for row in range(df_rows):
        earliest_row = max(0, row - (window_size - 1))
        last_row = min(df_rows - 1, row + window_forward)

        curr_rows = code_matrix[earliest_row:last_row + 1]
        curr_rows_summed = curr_rows.sum(axis=0)
        to_ut = vector_to_ut(curr_rows_summed)

        # Head subtraction: remove contributions from rows before the focal
        curr_rows_n = curr_rows.shape[0]
        if curr_rows_n > 0 and window_size > 1 and row - 1 >= 0:
            head_rows_count = curr_rows_n - 1 - window_forward
            if head_rows_count > 0:
                curr_rows_refs = curr_rows[:head_rows_count]
                curr_row_refs_summed = curr_rows_refs.sum(axis=0)
                to_ut_refs = vector_to_ut(curr_row_refs_summed)
                to_ut = to_ut - to_ut_refs

        # Tail subtraction: remove forward-lookahead contributions
        if curr_rows_n > 0 and window_forward > 0 and last_row <= df_rows - 1:
            tail_rows_to_use = last_row - row
            if tail_rows_to_use > 0:
                curr_rows_refs = curr_rows[-tail_rows_to_use:]
                curr_row_refs_summed = curr_rows_refs.sum(axis=0)
                to_ut_refs = vector_to_ut(curr_row_refs_summed)
                to_ut = to_ut - to_ut_refs

        df_co_occurred[row] = to_ut

    if binary:
        df_co_occurred = (df_co_occurred > 0).astype(float)

    return df_co_occurred


def compute_all_avs(
    df,
    codes,
    unit_col='unit',
    conversation_col='conversation',
    window_size=4,
    window_forward=0,
    binary=True,
):
    """Compute per-unit adjacency vectors over all stanza windows.

    For each unit, the function iterates over its conversations, applies the
    rENA-exact moving-stanza window within each conversation (windows do not
    cross conversation boundaries), and sums per-row adjacency vectors into a
    single per-unit vector.

    Parameters
    ----------
    df : pd.DataFrame
        Long-format utterance table. Must be sortable by a ``'turn'`` column
        within each unit, and must contain the columns listed in ``codes``,
        plus ``unit_col`` and ``conversation_col``.
    codes : list of str
        Code column names. Their order is preserved in the output AV columns.
    unit_col : str, default 'unit'
        Column name defining ENA units.
    conversation_col : str, default 'conversation'
        Column name defining conversation boundaries.
    window_size : int, default 4
        Stanza window size, passed to ``ref_window_df``.
    window_forward : int, default 0
        Forward lookahead, passed to ``ref_window_df``.
    binary : bool, default True
        Whether to clip co-occurrence counts to 0/1, passed to ``ref_window_df``.

    Returns
    -------
    avs : np.ndarray of shape (n_units, n_pairs)
        Per-unit adjacency vectors in rENA pair order.
    unit_ids : list
        Unit identifiers in the same order as ``avs`` rows.
    group_ids : list
        Group labels per unit if ``'group'`` is a column in ``df``; otherwise
        an empty list.
    """
    all_avs = []
    unit_ids = []
    group_ids = []
    has_group = 'group' in df.columns

    for unit_id in df[unit_col].unique():
        unit_data = (
            df[df[unit_col] == unit_id].sort_values('turn').reset_index(drop=True)
        )
        n_codes = len(codes)
        unit_av = np.zeros((n_codes * (n_codes + 1)) // 2 - n_codes)

        for conv in unit_data[conversation_col].unique():
            conv_data = (
                unit_data[unit_data[conversation_col] == conv]
                .sort_values('turn')
                .reset_index(drop=True)
            )
            cm = conv_data[codes].values
            co_occurred = ref_window_df(
                cm,
                window_size=window_size,
                window_forward=window_forward,
                binary=binary,
            )
            unit_av += co_occurred.sum(axis=0)

        all_avs.append(unit_av)
        unit_ids.append(unit_id)
        if has_group:
            group_ids.append(unit_data['group'].iloc[0])

    return np.array(all_avs), unit_ids, group_ids
