"""SVD projection and rENA-exact means rotation.

Two dimensionality-reduction routines used downstream of sphere normalization:

  - ``svd_project``    : centering + thin SVD, returns coords, components,
                         singular values, and the centering vector.
  - ``means_rotation`` : faithful reimplementation of rENA's
                         ``ena.rotate.by.mean``, including the auxiliary
                         ``orthogonal_svd`` routine. Verified against rENA
                         0.3.1 at machine epsilon (max |Δ| ≈ 2e-16).
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

__all__ = [
    "MeansRotationResult",
    "ProjectionResult",
    "means_rotation",
    "orthogonal_svd",
    "svd_project",
    "lws_lsq_positions",
]


@dataclass
class ProjectionResult:
    """Container for the output of :func:`svd_project`.

    Attributes
    ----------
    coords : np.ndarray of shape (n_units, n_components)
        Projected coordinates.
    components : np.ndarray of shape (n_components, n_features)
        The first ``n_components`` right-singular vectors (``Vt[:n_components]``).
    singular_values : np.ndarray of shape (n_components,)
        Top ``n_components`` singular values.
    mean : np.ndarray of shape (n_features,)
        Column-mean used for centering.
    centered : np.ndarray of shape (n_units, n_features)
        The centered input matrix, retained for downstream use and verification.
    """

    coords: np.ndarray
    components: np.ndarray
    singular_values: np.ndarray
    mean: np.ndarray
    centered: np.ndarray


@dataclass
class MeansRotationResult:
    """Container for the output of :func:`means_rotation`.

    Attributes
    ----------
    points : np.ndarray of shape (n_units, n_features)
        Rotated coordinates over all retained dimensions. The first column is
        ``MR1`` (the means-rotation axis), columns 2 onward are ``SVD2``,
        ``SVD3``, ... within the subspace orthogonal to ``MR1``.
    rotation : np.ndarray of shape (n_features, n_features)
        Full rotation matrix. ``points == centered @ rotation``.
    weights : np.ndarray of shape (n_features, 1)
        Unit vector defining the ``MR1`` axis (the normalized group-mean
        difference).
    groups_used : tuple
        ``(group_a, group_b)`` labels used to define the rotation axis.
    """

    points: np.ndarray
    rotation: np.ndarray
    weights: np.ndarray
    groups_used: tuple


def svd_project(normalized_matrix, n_components=2):
    """Center the input and project it onto its top singular components.

    Parameters
    ----------
    normalized_matrix : np.ndarray of shape (n_units, n_features)
        Output of :func:`pyena.core.normalize.sphere_normalize`.
    n_components : int, default 2
        Number of components to retain.

    Returns
    -------
    ProjectionResult
    """
    M = np.asarray(normalized_matrix, dtype=float)
    mean = M.mean(axis=0)
    centered = M - mean

    U, S, Vt = np.linalg.svd(centered, full_matrices=False)

    components = Vt[:n_components]
    singular_values = S[:n_components]
    coords = centered @ components.T

    return ProjectionResult(
        coords=coords,
        components=components,
        singular_values=singular_values,
        mean=mean,
        centered=centered,
    )


def orthogonal_svd(data, weights):
    """rENA-exact ``orthogonal_svd`` (``rENA/R/RotationSet.R``).

    Builds an orthonormal basis whose first ``ncol(weights)`` columns coincide
    with ``weights``, then runs an SVD inside the orthogonal complement and
    composes the result into a full rotation matrix.

    Parameters
    ----------
    data : np.ndarray of shape (n, p)
        Centered data matrix.
    weights : np.ndarray of shape (p, k)
        Axes to fix as the first ``k`` columns of the output rotation.
        For means rotation, ``k == 1``.

    Returns
    -------
    np.ndarray of shape (p, p)
        Full rotation matrix ``[weights | SVD-of-orthogonal-complement]``.
    """
    Q, _ = np.linalg.qr(weights, mode='complete')  # (p, p)
    k = weights.shape[1]

    # Project data into the subspace orthogonal to `weights`
    X_bar = data @ Q[:, k:]  # (n, p - k)

    # rENA uses prcomp(X_bar, scale=F), which centers by default.
    X_bar_centered = X_bar - X_bar.mean(axis=0)
    _, _, Vt = np.linalg.svd(X_bar_centered, full_matrices=False)
    V = Vt.T

    return np.hstack([Q[:, :k], Q[:, k:] @ V])


def means_rotation(centered_matrix, group_labels, group_a, group_b):
    """rENA-exact means rotation (``ena.rotate.by.mean``).

    Reimplements rENA 0.3.1's ``ena.rotate.by.mean`` (``rENA/R/RotationSet.R``)
    by reverse-engineering its R source and porting it to NumPy. Verified
    against rENA at machine epsilon on the included toy dataset
    (max |Δ| ≈ 2e-16).

    Algorithm:

      1. Re-center the input matrix (defensive; matches rENA's
         ``scale(data, scale=FALSE, center=TRUE)``).
      2. Compute the unit vector in the direction of
         ``mean(group_a) - mean(group_b)``; this becomes the ``MR1`` axis.
      3. Deflate the data along that axis.
      4. Apply :func:`orthogonal_svd` to build the full rotation matrix,
         keeping ``MR1`` as the first axis and using SVD on the orthogonal
         complement for the remaining axes.

    Parameters
    ----------
    centered_matrix : np.ndarray of shape (n_units, n_features)
        Sphere-normalized and centered adjacency-vector matrix.
    group_labels : array-like of shape (n_units,)
        Group label per unit.
    group_a, group_b : object
        The two group labels that define the ``MR1`` axis. The positive
        direction of ``MR1`` corresponds to ``mean(group_a) - mean(group_b)``.

    Returns
    -------
    MeansRotationResult

    Raises
    ------
    ValueError
        If either ``group_a`` or ``group_b`` is not present in
        ``group_labels``, or if the two group means are identical so that
        the rotation axis is undefined.
    """
    M = np.asarray(centered_matrix, dtype=float)
    labels = np.asarray(group_labels)

    if group_a not in labels or group_b not in labels:
        raise ValueError(
            f"group_a={group_a!r} and group_b={group_b!r} must both be present "
            f"in group_labels. Got: {sorted(set(labels.tolist()))}"
        )

    # rENA line 22: scale(data, scale=FALSE, center=TRUE)
    M = M - M.mean(axis=0)

    # rENA lines 39-44: unit vector in the direction of group-mean difference
    g1 = M[labels == group_a].mean(axis=0)
    g2 = M[labels == group_b].mean(axis=0)
    diff = g1 - g2
    norm = np.sqrt(np.sum(diff ** 2))
    if norm == 0:
        raise ValueError(
            "Group means are identical; means-rotation axis is undefined."
        )
    col_mean_diff_sq = diff / norm
    weights = col_mean_diff_sq[:, np.newaxis]

    # rENA lines 45-46: deflation
    deflated = M - (M @ weights) @ weights.T

    # rENA line 50: orthogonal_svd produces the full rotation matrix
    rotation = orthogonal_svd(deflated, weights)

    points = M @ rotation

    return MeansRotationResult(
        points=points,
        rotation=rotation,
        weights=weights,
        groups_used=(group_a, group_b),
    )

def lws_lsq_positions(line_weights, points, n_codes):
    """Compute ENA network node positions via least-squares placement.

    rENA-exact port of ``lws_lsq_positions`` (``ena.cpp`` lines 461-524).
    Node positions are placed so that each unit's weighted node centroid
    approximates that unit's ENA point, in a least-squares sense.

    Each code-pair weight is split half-and-half onto its two endpoint
    nodes; per-unit node weights are L1-normalized; then for each dimension
    the node coordinates solve ``W @ nodes ≈ points`` in least squares.

    Parameters
    ----------
    line_weights : np.ndarray of shape (n_units, n_pairs)
        Per-unit edge (adjacency) weights. rENA uses the sphere-normalized
        line weights; raw adjacency vectors give the same node positions
        up to the least-squares fit.
    points : np.ndarray of shape (n_units, n_dims)
        Rotated ENA coordinates for each unit.
    n_codes : int
        Number of codes (nodes).

    Returns
    -------
    np.ndarray of shape (n_codes, n_dims)
        Node positions in the ENA coordinate space.
    """
    W = np.asarray(line_weights, dtype=float)
    P = np.asarray(points, dtype=float)
    n_units = W.shape[0]
    n_dims = P.shape[1]

    # Distribute each pair's weight half-and-half onto its endpoint nodes.
    # Pair order is the lower-triangular traversal used by vector_to_ut:
    # for x in range(n_codes-1): for y in range(x+1): -> pair (y, x+1)
    node_w = np.zeros((n_units, n_codes))
    z = 0
    for x in range(n_codes - 1):
        for y in range(x + 1):
            node_w[:, x + 1] += 0.5 * W[:, z]
            node_w[:, y] += 0.5 * W[:, z]
            z += 1

    # L1-normalize each unit's node weights (rENA clamps tiny lengths).
    length = np.abs(node_w).sum(axis=1)
    length[length < 1e-4] = 1e-4
    node_w = node_w / length[:, None]

    # Per-dimension least squares: W @ nodes ≈ points.
    nodes = np.zeros((n_codes, n_dims))
    for i in range(n_dims):
        sol, *_ = np.linalg.lstsq(node_w, P[:, i], rcond=None)
        nodes[:, i] = sol
    return nodes