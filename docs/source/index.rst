pyena
=====

**pyena** is a Python implementation of Epistemic Network Analysis (ENA),
numerically verified against the reference R package
`rENA <https://cran.r-project.org/package=rENA>`_ 0.3.1.

It reproduces the full ENA pipeline — adjacency construction, sphere-normalized
SVD projection, means rotation, and least-squares node placement — and exposes
it through a scikit-learn-style estimator that composes with pandas, NumPy, and
matplotlib.

Installation
------------

.. code-block:: bash

   pip install pyena

Requires Python 3.10 or later.

Quickstart
----------

.. code-block:: python

   import pandas as pd
   from pyena import ENA

   df = pd.read_csv("your_long_format_utterances.csv")

   ena = ENA(
       codes=["Data", "Theory", "Question", "Example", "Critique"],
       unit_col="unit",
       conversation_col="conversation",
       window_size=4,
       rotation="means",
       mr_groups=("A", "B"),
   )
   ena.fit(df)
   ena.plot()                      # standard ENA network
   result = ena.compare(axis="x")  # Welch, Mann-Whitney, permutation

See :doc:`tutorial` for a complete worked example on a real dataset.

.. toctree::
   :maxdepth: 2
   :caption: Contents

   tutorial
   api/modules

Validation
----------

pyena is verified against rENA 0.3.1 on nine datasets (the RS.data corpus and
eight synthetic datasets). Adjacency vectors match exactly; SVD and
means-rotation coordinates agree to machine precision; least-squares node
positions match within 1e-13. See the ``tests/`` suite and the R scripts in
``validation_rsdata/`` for the regression fixtures.

Citation
--------

If you use pyena, please cite the archived release
(`DOI 10.5281/zenodo.21090822 <https://doi.org/10.5281/zenodo.21090822>`_) and
the accompanying SoftwareX paper (in preparation).

Indices
-------

* :ref:`genindex`
* :ref:`modindex`
