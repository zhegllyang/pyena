"""pyena: a rENA-exact Python implementation of Epistemic Network Analysis.

See the README for installation and quickstart. The recommended entry point
is the :class:`ENA` estimator::

    from pyena import ENA
    ena = ENA(codes=[...], rotation='means', mr_groups=('A', 'B')).fit(df)
    ena.plot()
"""

from pyena.api import ENA

__all__ = ["ENA"]
from pyena._version import __version__
