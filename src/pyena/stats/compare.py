"""Group comparison on ENA coordinates.

Reports three complementary tests so that the user can choose the most
appropriate one for their sample-size and distributional regime:

  - Welch's t-test    : parametric, no equal-variance assumption.
  - Mann-Whitney U    : non-parametric, distribution-free.
  - Permutation test  : exact under the null, robust at small ``n``.

In addition the function reports Cohen's d (standardised effect size) and the
smallest achievable two-sided Mann-Whitney p-value at the given sample sizes,
which is useful for diagnosing under-powered designs.
"""

from __future__ import annotations

import warnings
from math import comb

import numpy as np
from scipy import stats

__all__ = ["ena_group_comparison"]


def ena_group_comparison(
    points,
    group_labels,
    axis='x',
    n_permutations=10000,
    random_state=42,
):
    """Compare two groups along one ENA axis using three statistical tests.

    Parameters
    ----------
    points : np.ndarray of shape (n_units, >= 2)
        ENA coordinates. ``axis='x'`` uses column 0, ``axis='y'`` uses column 1.
    group_labels : array-like of shape (n_units,)
        Group label per unit. Exactly two distinct labels are required.
    axis : {'x', 'y'}, default 'x'
        Which axis to test.
    n_permutations : int, default 10000
        Number of permutations used by the permutation test.
    random_state : int, default 42
        Seed for the permutation test's random number generator. Fixing it
        guarantees byte-for-byte reproducibility.

    Returns
    -------
    dict
        Results dictionary with the following keys:

        - ``axis``, ``n1``, ``n2``
        - ``g1_mean``, ``g1_std``, ``g2_mean``, ``g2_std``, ``observed_diff``
        - ``mannwhitney_U``, ``mannwhitney_p``, ``mannwhitney_min_possible_p``
        - ``welch_t``, ``welch_p``
        - ``permutation_p``, ``permutation_n``
        - ``cohens_d``

    Raises
    ------
    ValueError
        If ``group_labels`` does not contain exactly two distinct values.

    Warns
    -----
    UserWarning
        Emitted when either group has fewer than four units, since the
        Mann-Whitney U test cannot achieve p < 0.05 under those conditions.
    """
    axis_idx = 0 if axis == 'x' else 1
    groups = np.unique(group_labels)
    if len(groups) != 2:
        raise ValueError(
            f"Exactly two groups are required. Got: {list(groups)}"
        )

    group_labels = np.array(group_labels)
    g1_vals = points[group_labels == groups[0], axis_idx]
    g2_vals = points[group_labels == groups[1], axis_idx]
    n1, n2 = len(g1_vals), len(g2_vals)

    # Small-sample warning
    if min(n1, n2) < 4:
        warnings.warn(
            f"Small sample warning: n1={n1}, n2={n2}. "
            f"Mann-Whitney U cannot achieve p < 0.05 with fewer than four "
            f"units per group. Consider the permutation test or larger samples.",
            UserWarning,
        )

    # Standard tests
    u_stat, u_pval = stats.mannwhitneyu(g1_vals, g2_vals, alternative='two-sided')
    t_stat, t_pval = stats.ttest_ind(g1_vals, g2_vals, equal_var=False)

    # Permutation test
    observed_diff = g1_vals.mean() - g2_vals.mean()
    combined = np.concatenate([g1_vals, g2_vals])
    rng = np.random.default_rng(random_state)
    perm_diffs = np.zeros(n_permutations)
    for i in range(n_permutations):
        shuffled = rng.permutation(combined)
        perm_diffs[i] = shuffled[:n1].mean() - shuffled[n1:].mean()
    perm_pval = (np.abs(perm_diffs) >= np.abs(observed_diff)).mean()

    # Cohen's d (pooled standard deviation)
    pooled_std = np.sqrt(
        ((n1 - 1) * g1_vals.var(ddof=1) + (n2 - 1) * g2_vals.var(ddof=1))
        / (n1 + n2 - 2)
    )
    cohens_d = observed_diff / pooled_std if pooled_std > 0 else np.inf

    # Smallest achievable two-sided Mann-Whitney p-value at these sample sizes
    min_possible_p = 2 / comb(n1 + n2, n1)

    return {
        'axis': axis,
        'n1': n1, 'n2': n2,
        'g1_mean': g1_vals.mean(), 'g1_std': g1_vals.std(ddof=1),
        'g2_mean': g2_vals.mean(), 'g2_std': g2_vals.std(ddof=1),
        'observed_diff': observed_diff,
        'mannwhitney_U': u_stat, 'mannwhitney_p': u_pval,
        'mannwhitney_min_possible_p': min_possible_p,
        'welch_t': t_stat, 'welch_p': t_pval,
        'permutation_p': perm_pval,
        'permutation_n': n_permutations,
        'cohens_d': cohens_d,
    }
