"""pyena: a rENA-exact Python implementation of Epistemic Network Analysis.

See the README for installation and quickstart. The recommended entry point
is the :class:`ENA` estimator::

    from pyena import ENA
    ena = ENA(codes=[...], rotation='means', mr_groups=('A', 'B')).fit(df)
    ena.plot()
"""

from pyena.api import ENA

__all__ = ["ENA"]
__version__ = "0.1.0.dev0"
