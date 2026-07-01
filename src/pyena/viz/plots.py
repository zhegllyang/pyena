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
from itertools import combinations

__all__ = ["plot_reproducibility", "plot_network"]


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

def plot_network(
    coords,
    node_positions,
    line_weights,
    group_labels=None,
    code_labels=None,
    ax=None,
    xlabel="MR1",
    ylabel="SVD2",
    title="ENA network",
    colors=("#c0392b", "#2471a3"),
    node_scale=0.85,
):
    """Draw a standard ENA network visualization.

    Renders code nodes at their least-squares positions, edges whose
    thickness reflects mean co-occurrence weight, and unit points with
    per-group centroids and 95% confidence ellipses.

    When exactly two groups are present, edges show the sphere-normalized
    mean-network difference between them (``colors[0]`` where the first group
    is stronger, ``colors[1]`` where the second is), which removes the
    total-connectivity effect and reveals pattern differences. With a single
    group (or no group labels), edges show that group's mean network.

    Parameters
    ----------
    coords : np.ndarray of shape (n_units, 2)
        Rotated ENA coordinates (e.g. ``ENA.coords_``).
    node_positions : np.ndarray of shape (n_codes, 2)
        Node positions (e.g. ``ENA.nodes_``).
    line_weights : np.ndarray of shape (n_units, n_pairs)
        Sphere-normalized per-unit edge weights (e.g.
        ``ENA.av_normalized_``).
    group_labels : array-like of shape (n_units,), optional
        Group label per unit. If two distinct groups are present, a
        subtracted network is drawn; otherwise a single mean network.
    code_labels : list of str, optional
        Node labels; defaults to ``["C1", "C2", ...]``.
    ax : matplotlib.axes.Axes, optional
        Axis to draw on; a new figure/axis is created if omitted.
    xlabel, ylabel, title : str
        Axis labels and title.
    colors : tuple of (str, str)
        Colors for the first and second group.
    node_scale : float
        Fraction of the coordinate extent used to scale node positions
        into the points space.

    Returns
    -------
    matplotlib.axes.Axes
        The axis containing the plot.
    """
    coords = np.asarray(coords, dtype=float)
    nodes = np.asarray(node_positions, dtype=float)
    lw = np.asarray(line_weights, dtype=float)
    n_codes = nodes.shape[0]
    pairs = list(combinations(range(n_codes), 2))

    if code_labels is None:
        code_labels = [f"C{i + 1}" for i in range(n_codes)]

    # Determine the network to draw.
    groups = None if group_labels is None else np.asarray(group_labels)
    uniq = [] if groups is None else list(dict.fromkeys(groups.tolist()))
    if len(uniq) == 2:
        m1 = groups == uniq[0]
        m2 = groups == uniq[1]
        net = lw[m1].mean(axis=0) - lw[m2].mean(axis=0)
        group_masks = [(m1, colors[0], str(uniq[0])),
                       (m2, colors[1], str(uniq[1]))]
    else:
        net = lw.mean(axis=0)
        group_masks = [(np.ones(len(coords), dtype=bool), colors[0], "all")]

    if ax is None:
        _, ax = plt.subplots(figsize=(7, 6))

    # Scale node positions into the coordinate space.
    denom = np.abs(nodes).max()
    scale = (np.abs(coords).max() / denom * node_scale) if denom > 0 else 1.0
    nodes_s = nodes * scale

    # Edges (thick drawn first so thin lines sit on top).
    tmax = np.abs(net).max() or 1.0
    for idx in np.argsort(np.abs(net)):
        i, j = pairs[idx]
        w = net[idx]
        color = colors[0] if w >= 0 else colors[1]
        ax.plot(
            [nodes_s[i, 0], nodes_s[j, 0]],
            [nodes_s[i, 1], nodes_s[j, 1]],
            color=color, lw=0.3 + 5.0 * (abs(w) / tmax),
            alpha=0.6, zorder=1, solid_capstyle="round",
        )

    # Nodes (size reflects summed incident edge magnitude).
    node_w = np.zeros(n_codes)
    for idx, (i, j) in enumerate(pairs):
        node_w[i] += abs(net[idx])
        node_w[j] += abs(net[idx])
    nmax = node_w.max() or 1.0
    ax.scatter(nodes_s[:, 0], nodes_s[:, 1],
               s=40 + 150 * (node_w / nmax),
               c="#333333", zorder=3, edgecolor="white", lw=1)
    for k, lab in enumerate(code_labels):
        ax.annotate(lab, nodes_s[k], fontsize=9, fontweight="bold",
                    xytext=(6, 4), textcoords="offset points", zorder=4)

    # Unit points, centroids, and 95% confidence ellipses.
    for mask, color, name in group_masks:
        pts = coords[mask]
        ax.scatter(pts[:, 0], pts[:, 1], s=20, c=color, alpha=0.55,
                   zorder=2, label=f"{name} (n={int(mask.sum())})")
        centroid = pts.mean(axis=0)
        ax.scatter(*centroid, s=240, c=color, marker="X",
                   edgecolor="black", lw=1.5, zorder=5)
        if len(pts) >= 3:
            cov = np.cov(pts.T)
            vals, vecs = np.linalg.eigh(cov)
            order = vals.argsort()[::-1]
            vals, vecs = vals[order], vecs[:, order]
            angle = np.degrees(np.arctan2(*vecs[:, 0][::-1]))
            width, height = 2.0 * np.sqrt(vals * chi2.ppf(0.95, 2))
            ax.add_patch(Ellipse(centroid, width, height, angle=angle,
                                 facecolor=color, alpha=0.10,
                                 edgecolor=color, lw=1))

    ax.axhline(0, color="#dddddd", lw=0.7, zorder=0)
    ax.axvline(0, color="#dddddd", lw=0.7, zorder=0)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    if any(name != "all" for _, _, name in group_masks):
        ax.legend(loc="lower right", frameon=True, fontsize=9)
    ax.set_aspect("equal", adjustable="datalim")
    return ax