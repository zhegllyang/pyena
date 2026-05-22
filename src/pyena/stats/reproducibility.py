"""Reproducibility metrics in ENA space.

Quantifies how tightly a set of units clusters in the projected ENA space.
For each group (or for the data as a whole), four indicators are computed:

  1. **Centroid dispersion** — mean (and standard deviation) of the distance
     from each unit to the group centroid. Low values indicate convergence.
  2. **Pairwise distance** — mean and maximum pairwise Euclidean distance
     between units.
  3. **95% confidence ellipse area** — area of the bivariate-normal
     confidence ellipse, computed as ``π · χ²(0.95, df=2) · √(λ₁ λ₂)`` where
     ``λ₁, λ₂`` are the eigenvalues of the sample covariance matrix.
  4. **Convex hull area** — area of the actual polygon spanned by the units
     (via ``scipy.spatial.ConvexHull``).

Lower values across all four indicators indicate higher reproducibility.
These metrics are designed for, but not limited to, LLM dialogue-path
analyses where the same prompt is run multiple times.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.spatial import ConvexHull
from scipy.spatial.distance import pdist
from scipy.stats import chi2

__all__ = ["reproducibility_metrics"]


def reproducibility_metrics(points, group_labels=None):
    """Compute four reproducibility indicators, per group or overall.

    Parameters
    ----------
    points : np.ndarray of shape (n_units, 2)
        ENA two-dimensional coordinates.
    group_labels : array-like of shape (n_units,), optional
        If provided, metrics are computed per group. If ``None``, all rows
        are treated as a single group with index ``'all'``.

    Returns
    -------
    pd.DataFrame
        One row per group (or one row labelled ``'all'``), with nine columns:

        ``n``, ``centroid_x``, ``centroid_y``,
        ``centroid_dispersion_mean``, ``centroid_dispersion_std``,
        ``pairwise_distance_mean``, ``pairwise_distance_max``,
        ``ellipse_95_area``, ``convex_hull_area``.

        Columns that require ``n >= 2`` (pairwise) or ``n >= 3`` (ellipse,
        hull) become ``NaN`` for under-sized groups.
    """
    def _compute_one_group(pts):
        n = len(pts)
        centroid = pts.mean(axis=0)

        # (1) Centroid dispersion
        distances_to_centroid = np.linalg.norm(pts - centroid, axis=1)
        centroid_dispersion = distances_to_centroid.mean()
        centroid_dispersion_std = distances_to_centroid.std()

        # (2) Pairwise distance
        if n >= 2:
            pairwise = pdist(pts)
            pairwise_mean = pairwise.mean()
            pairwise_max = pairwise.max()
        else:
            pairwise_mean = np.nan
            pairwise_max = np.nan

        # (3) 95% confidence ellipse area
        # Area = π · χ²(0.95, df=2) · √(λ₁ λ₂)
        if n >= 3:
            cov = np.cov(pts.T)
            eigenvals = np.linalg.eigvalsh(cov)
            chi2_val = chi2.ppf(0.95, df=2)
            ellipse_area = np.pi * chi2_val * np.sqrt(np.prod(eigenvals))
        else:
            ellipse_area = np.nan

        # (4) Convex hull area (in 2D, ConvexHull.volume == area)
        if n >= 3:
            try:
                hull = ConvexHull(pts)
                hull_area = hull.volume
            except Exception:
                hull_area = np.nan
        else:
            hull_area = np.nan

        return {
            'n': n,
            'centroid_x': centroid[0], 'centroid_y': centroid[1],
            'centroid_dispersion_mean': centroid_dispersion,
            'centroid_dispersion_std': centroid_dispersion_std,
            'pairwise_distance_mean': pairwise_mean,
            'pairwise_distance_max': pairwise_max,
            'ellipse_95_area': ellipse_area,
            'convex_hull_area': hull_area,
        }

    if group_labels is None:
        return pd.DataFrame([_compute_one_group(points)], index=['all'])

    group_labels = np.array(group_labels)
    results = {}
    for g in np.unique(group_labels):
        pts_g = points[group_labels == g]
        results[g] = _compute_one_group(pts_g)

    return pd.DataFrame(results).T
