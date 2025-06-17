More infrastructure
===================

In addition to the query model and database interaction, PostBOUND also provides a lot of utility or infrastructure-focused
functionality. Generally speaking, this functionality belongs to one of three categories: benchmarks and workloads,
visualization and miscellaneous utilities.


Benchmarks and workloads
------------------------

The `postbound.experiments` package provides a representation of workloads that consist of a number of labelled queries in the
`workloads` module. The ``Workloads`` class offers high-level access to these queries, including shuffling, sampling and
merging different workloads, as well as utilities to load benchmark queries from disk. In addition to the workload abstraction
PostBOUND also ships a number of pre-defined workloads. The module contains utility functions to directly load these workloads.
For these functions to work, the location of the workloads must be known. By default, the functions assume that they are
executed in the *postbound* directory and that the queries themselves are contained in the root-level *workloads* directory.
If this is not the case, the ``workloads_base_dir`` variable should be set to the correct location. The module documentation
contains more details on this aspect.

The `executor` module builds on top of workloads and optimization pipeline to provide a reproducible and transparent execution
of benchmarks with optimized queries. The different methods provide results as Pandas DataFrames [1]_ to enable an easy
integration into data analysis tools.


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
