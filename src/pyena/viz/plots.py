"""Visualization of ENA coordinates.

Provides a single high-level function, :func:`plot_reproducibility`, which
draws a 2D scatter of ENA coordinates with per-group centroids (``X`` markers)
and 95% confidence ellipses.
"""

from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Ellipse
from scipy.stats import chi2

__all__ = ["plot_reproducibility"]


def plot_reproducibility(
    points,
    group_labels,
    ax=None,
    colors=None,
    show_ellipse=True,
    title='ENA Space: Reproducibility Visualization',
    xlabel='PC1',
    ylabel='PC2',
):
    """Plot ENA coordinates coloured by group with 95% confidence ellipses.

    Each group is rendered as scattered units, a centroid marker, and a
    dashed-edge 95% confidence ellipse (drawn from the eigendecomposition
    of the group's sample covariance matrix). The ellipse is drawn only for
    groups with at least three units.

    Parameters
    ----------
    points : np.ndarray of shape (n_units, 2)
        ENA two-dimensional coordinates.
    group_labels : array-like of shape (n_units,)
        Group label per unit.
    ax : matplotlib.axes.Axes, optional
        Existing axes to draw into. If ``None``, a new figure and axes are
        created.
    colors : sequence of color, optional
        One colour per group, in the order returned by ``np.unique``. If
        ``None``, the ``tab10`` colormap is sampled.
    show_ellipse : bool, default True
        Whether to draw the 95% confidence ellipse for each group.
    title, xlabel, ylabel : str
        Figure title and axis labels.

    Returns
    -------
    matplotlib.axes.Axes
        The axes on which the plot was drawn.
    """
    if ax is None:
        _, ax = plt.subplots(figsize=(10, 8))

    group_labels = np.array(group_labels)
    groups = np.unique(group_labels)

    if colors is None:
        colors = plt.cm.tab10(np.linspace(0, 1, len(groups)))

    for g, color in zip(groups, colors):
        mask = group_labels == g
        pts = points[mask]

        # Individual points
        ax.scatter(
            pts[:, 0], pts[:, 1],
            s=40, alpha=0.5, color=color,
            label=f'{g} (n={mask.sum()})',
            edgecolor='white', linewidth=0.5,
        )

        # Group centroid (X marker)
        centroid = pts.mean(axis=0)
        ax.scatter(
            centroid[0], centroid[1],
            s=300, marker='X', color=color,
            edgecolor='black', linewidth=2, zorder=5,
        )

        # 95% confidence ellipse
        if show_ellipse and len(pts) >= 3:
            cov = np.cov(pts.T)
            eigenvals, eigenvecs = np.linalg.eigh(cov)
            order = eigenvals.argsort()[::-1]
            eigenvals = eigenvals[order]
            eigenvecs = eigenvecs[:, order]

            angle = np.degrees(np.arctan2(eigenvecs[1, 0], eigenvecs[0, 0]))
            chi2_val = chi2.ppf(0.95, df=2)
            width = 2 * np.sqrt(chi2_val * eigenvals[0])
            height = 2 * np.sqrt(chi2_val * eigenvals[1])

            ellipse = Ellipse(
                centroid, width, height, angle=angle,
                facecolor=color, alpha=0.15, edgecolor=color,
                linewidth=2, linestyle='--',
            )
            ax.add_patch(ellipse)

    ax.axhline(0, color='gray', linewidth=0.5, alpha=0.5)
    ax.axvline(0, color='gray', linewidth=0.5, alpha=0.5)
    ax.set_xlabel(xlabel, fontsize=12)
    ax.set_ylabel(ylabel, fontsize=12)
    ax.set_title(
        f'{title}\n'
        '(dots = individual runs, ellipses = 95% confidence area, '
        'X = centroid)',
        fontsize=12,
    )
    ax.legend(loc='best', fontsize=11)
    ax.set_aspect('equal')
    ax.grid(True, alpha=0.2)

    return ax
