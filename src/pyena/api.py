"""scikit-learn-style ENA estimator — the recommended entry point of pyena.

Basic usage
-----------
.. code-block:: python

    from pyena import ENA

    # SVD coordinates
    ena = ENA(codes=['c1', 'c2', 'c3'], window_size=4)
    ena.fit(df)
    coords = ena.transform(df)        # or simply ena.coords_

    # Means-rotation coordinates (for two-group analyses)
    ena = ENA(codes=codes, rotation='means', mr_groups=('A', 'B'))
    ena.fit(df)                       # df must contain a 'group' column
    ena.plot()
    ena.compare(axis='x')
    ena.reproducibility()

Fitted attributes (scikit-learn convention: trailing ``_``)
-----------------------------------------------------------
- ``av_``                : ``(n_units, n_pairs)`` — rENA-exact adjacency vectors.
- ``av_normalized_``     : ``(n_units, n_pairs)`` — sphere-normalized AVs.
- ``av_centered_``       : ``(n_units, n_pairs)`` — centered (training mean).
- ``mean_``              : ``(n_pairs,)`` — column means used for centering.
- ``components_``        : ``(n_components, n_pairs)`` — projection axes;
                          ``coords == centered @ components_.T``.
- ``singular_values_``   : ``(n_components,)`` — only when ``rotation='svd'``.
- ``rotation_matrix_``   : ``(n_pairs, n_pairs)`` — only when ``rotation='means'``.
- ``mr_groups_used_``    : tuple ``(group_a, group_b)`` — only for MR.
- ``coords_``            : ``(n_units, n_components)`` — coordinates of the
                          training data.
- ``unit_ids_``, ``group_ids_``, ``codes_``, ``n_features_in_``.
"""

from __future__ import annotations

import numpy as np
from sklearn.base import BaseEstimator

from pyena.core.adjacency import compute_all_avs
from pyena.core.normalize import sphere_normalize
from pyena.core.projection import lws_lsq_positions, means_rotation, svd_project

__all__ = ["ENA"]


class ENA(BaseEstimator):
    """rENA-exact Epistemic Network Analysis estimator (scikit-learn style).

    Wraps the low-level building blocks of pyena (adjacency vector
    construction, sphere normalization, SVD projection, and optional
    means rotation) behind a single ``fit`` / ``transform`` interface that
    follows scikit-learn conventions.

    Parameters
    ----------
    codes : list of str
        Names of the binary code columns in the input frame.
    unit_col : str, default 'unit'
        Column defining ENA units.
    conversation_col : str, default 'conversation'
        Column bounding the moving-stanza window (windows do not cross
        conversation boundaries).
    group_col : str, default 'group'
        Column holding group labels. Used by ``rotation='means'`` and by the
        convenience methods ``plot``, ``compare``, and ``reproducibility``.
    window_size : int, default 4
        Moving-stanza window size.
    window_forward : int, default 0
        Forward lookahead. rENA's default is 0.
    binary : bool, default True
        If True, co-occurrence counts are clipped to 0/1 (rENA default).
    n_components : int, default 2
        Number of components to retain.
    rotation : {'svd', 'means'}, default 'svd'
        Post-projection rotation strategy. ``'svd'`` keeps the top singular
        components (matches rENA's default ``ena.svd``). ``'means'`` rotates
        so that the first axis aligns with the difference of two group means
        (matches rENA's ``ena.rotate.by.mean``) and requires ``mr_groups``.
    mr_groups : tuple of (object, object), optional
        Required when ``rotation='means'``. Defines the rotation axis:
        ``mean(group_a) - mean(group_b)`` points in the positive direction of
        the first axis.

    See Also
    --------
    pyena.core.adjacency.compute_all_avs : Adjacency-vector construction.
    pyena.core.projection.svd_project    : SVD projection.
    pyena.core.projection.means_rotation : rENA-exact means rotation.
    """

    def __init__(
        self,
        codes,
        unit_col='unit',
        conversation_col='conversation',
        group_col='group',
        window_size=4,
        window_forward=0,
        binary=True,
        n_components=2,
        rotation='svd',
        mr_groups=None,
    ):
        self.codes = codes
        self.unit_col = unit_col
        self.conversation_col = conversation_col
        self.group_col = group_col
        self.window_size = window_size
        self.window_forward = window_forward
        self.binary = binary
        self.n_components = n_components
        self.rotation = rotation
        self.mr_groups = mr_groups

    def _prep_df(self, X):
        """Rename ``group_col`` to ``'group'`` for downstream compatibility."""
        if self.group_col != 'group' and self.group_col in X.columns:
            return X.rename(columns={self.group_col: 'group'})
        return X

    def fit(self, X, y=None):
        """Fit the ENA model: adjacency → normalize → project → (optional) rotate.

        Parameters
        ----------
        X : pd.DataFrame
            Long-format utterance frame.
        y : ignored
            Present for scikit-learn API compatibility. Group labels are read
            from the column named ``group_col``.

        Returns
        -------
        self : ENA
        """
        if self.rotation not in ('svd', 'means'):
            raise ValueError(
                f"rotation must be 'svd' or 'means'. Got: {self.rotation!r}"
            )
        if self.rotation == 'means' and self.mr_groups is None:
            raise ValueError(
                "rotation='means' requires mr_groups=(group_a, group_b)."
            )

        df_use = self._prep_df(X)

        av, unit_ids, group_ids = compute_all_avs(
            df_use, self.codes,
            unit_col=self.unit_col,
            conversation_col=self.conversation_col,
            window_size=self.window_size,
            window_forward=self.window_forward,
            binary=self.binary,
        )

        av_n = sphere_normalize(av)
        proj = svd_project(av_n, n_components=self.n_components)

        if self.rotation == 'svd':
            self.coords_ = proj.coords
            self.components_ = proj.components
            self.singular_values_ = proj.singular_values
            self.rotation_matrix_ = None
            self.mr_groups_used_ = None
        else:  # 'means'
            if not group_ids:
                raise ValueError(
                    f"rotation='means' requires a {self.group_col!r} column "
                    f"in X."
                )
            group_arr = np.array(group_ids)
            ga, gb = self.mr_groups
            mr = means_rotation(proj.centered, group_arr, ga, gb)
            # Keep only the first n_components axes.
            self.coords_ = mr.points[:, :self.n_components]
            # scikit-learn convention: coords = centered @ components_.T.
            # In MR: coords = centered @ rotation[:, :n], so
            # components_ = rotation[:, :n].T → shape (n_components, n_features).
            self.components_ = mr.rotation[:, :self.n_components].T
            self.singular_values_ = None
            self.rotation_matrix_ = mr.rotation
            self.mr_groups_used_ = mr.groups_used

        # Shared fitted attributes
        self.av_ = av
        self.av_normalized_ = av_n
        self.av_centered_ = proj.centered
        self.mean_ = proj.mean
        self.unit_ids_ = unit_ids
        self.group_ids_ = group_ids
        self.codes_ = list(self.codes)
        self.n_features_in_ = av.shape[1]
        # ENA network node positions (rENA-exact least-squares placement).
        self.nodes_ = lws_lsq_positions(av_n, self.coords_, len(self.codes))

        return self

    def transform(self, X):
        """Project new data into the fitted ENA space.

        Pipeline: AV construction → sphere normalization → centering with the
        training mean → projection through ``components_``.

        Parameters
        ----------
        X : pd.DataFrame
            Long-format utterance frame.

        Returns
        -------
        np.ndarray of shape (n_units, n_components)
        """
        if not hasattr(self, 'mean_'):
            raise RuntimeError("Call fit() before transform().")

        df_use = self._prep_df(X)
        av, _, _ = compute_all_avs(
            df_use, self.codes,
            unit_col=self.unit_col,
            conversation_col=self.conversation_col,
            window_size=self.window_size,
            window_forward=self.window_forward,
            binary=self.binary,
        )
        av_n = sphere_normalize(av)
        centered = av_n - self.mean_
        return centered @ self.components_.T

    def fit_transform(self, X, y=None):
        """Fit the model and return the training-data coordinates."""
        return self.fit(X, y).coords_

    # ===== Convenience methods =====

    def plot(self, kind="network", ax=None, code_labels=None, **kwargs):
        """Visualize the fitted ENA space.

        Parameters
        ----------
        kind : {'network', 'reproducibility'}, default 'network'
            ``'network'`` draws the standard ENA representation: code nodes
            at their least-squares positions, edges weighted by mean
            co-occurrence (subtracted between the two groups when applicable),
            plus unit points, centroids, and 95% confidence ellipses.
            ``'reproducibility'`` draws only the unit-point scatter with
            per-group centroids and confidence ellipses.
        ax : matplotlib.axes.Axes, optional
            Axis to draw on; created if omitted.
        code_labels : list of str, optional
            Node labels for the network plot; defaults to ``codes_``.
        **kwargs
            Forwarded to the underlying plotting function.

        Returns
        -------
        matplotlib.axes.Axes
        """
        if kind == "network":
            from pyena.viz.plots import plot_network
            xlabel = "MR1" if self.rotation_matrix_ is not None else "SVD1"
            group_labels = (
                np.array(self.group_ids_) if self.group_ids_ else None
            )
            return plot_network(
                self.coords_,
                self.nodes_,
                self.av_normalized_,
                group_labels=group_labels,
                code_labels=code_labels if code_labels is not None
                else self.codes_,
                ax=ax,
                xlabel=kwargs.pop("xlabel", xlabel),
                ylabel=kwargs.pop("ylabel", "SVD2"),
                **kwargs,
            )
        elif kind == "reproducibility":
            from pyena.viz.plots import plot_reproducibility
            if not self.group_ids_:
                raise ValueError(
                    f"plot(kind='reproducibility') requires a "
                    f"{self.group_col!r} column in X."
                )
            return plot_reproducibility(
                self.coords_, np.array(self.group_ids_), ax=ax, **kwargs
            )
        else:
            raise ValueError(
                f"Unknown kind={kind!r}; expected 'network' or "
                f"'reproducibility'."
            )

    def compare(self, axis='x', groups=None, n_permutations=10000, random_state=42):
        """Two-group comparison (Welch + Mann-Whitney + permutation).

        Parameters
        ----------
        axis : {'x', 'y'}, default 'x'
            Which axis to test.
        groups : tuple of (object, object), optional
            The two groups to compare. If ``None``, ``mr_groups_used_`` is
            used (available only when ``rotation='means'``).
        n_permutations, random_state
            Passed to :func:`pyena.stats.compare.ena_group_comparison`.

        Returns
        -------
        dict
            Results dictionary as returned by ``ena_group_comparison``.
        """
        from pyena.stats.compare import ena_group_comparison
        if not self.group_ids_:
            raise ValueError(
                f"compare() requires a {self.group_col!r} column in X."
            )
        if groups is None:
            if self.mr_groups_used_ is None:
                raise ValueError(
                    "More than two groups present. Pass groups=(a, b) "
                    "explicitly, or fit with rotation='means' to set "
                    "mr_groups_used_."
                )
            groups = self.mr_groups_used_
        labels = np.array(self.group_ids_)
        mask = np.isin(labels, groups)
        return ena_group_comparison(
            self.coords_[mask], labels[mask],
            axis=axis, n_permutations=n_permutations, random_state=random_state,
        )

    def reproducibility(self):
        """Compute the four reproducibility indicators per group.

        Returns
        -------
        pd.DataFrame
            As returned by :func:`pyena.stats.reproducibility.reproducibility_metrics`.
        """
        from pyena.stats.reproducibility import reproducibility_metrics
        labels = np.array(self.group_ids_) if self.group_ids_ else None
        return reproducibility_metrics(self.coords_, labels)
