More infrastructure
===================

In addition to the query model and database interaction, PostBOUND also provides a lot of utility or infrastructure-focused
functionality. Generally speaking, this functionality belongs to one of three categories: benchmarks and workloads,
visualization and miscellaneous utilities.


Benchmarks and workloads
------------------------

The `postbound.experiments` package provides a representation of workloads that consist of a number of labelled queries in the
`workloads` module. The ``Workloads`` class offers high-level access to these queries, including shuffling, sampling and
merging different workloads, as well as utilities to load benchmark queries from disk.

The `runner` module builds on top of workloads and optimization pipeline to provide a reproducible and transparent execution
of benchmarks with optimized queries. The different methods provide results as Pandas DataFrames [1]_ to enable an easy
integration into data analysis tools. The utilities of the `analysis` module can facilitate this even further, especially when
writing the DataFrames to disk.


Visualization
--------------

The `postbound.vis` package provides utilities to generate graphical representations of different parts of the optimization
process. This includes tools to draw join orders, query plans and join graphs. These representation utilize Graphviz [2]_ to
enable easy usage in other tools. Furthermore, the `fdl` module provides a simple implementation of a force-directed layout
algorithm to draw large sets of data points without any spatial information in a 2D-plane. This only requires a distance
measure for the pairwise comparison of data points.


Utilities
---------

The `postbound.util` package contains all those general purpose methods and classes that do not have anything to do with
PostBOUND in particulary and are much more general in nature. Take a look at the individual modules for an overview of the
different methods.

Links
-----

.. [1] Pandas project: https://pandas.pydata.org/pandas-docs/stable/index.html
.. [2] Graphviz project: https://graphviz.org/
