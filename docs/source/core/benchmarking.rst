Benchmarking
============

A central goal of PostBOUND is to enable an easy execution of query plans on actual database systems to better judge how
modifications of real-world query optimizers impact various workloads. We treat the query execution time as the ultimate
metric to evaluate different optimization algorithms [#eval-metrics]_.
PostBOUND provides a suite of basic tools to make common taks in benchmarking and reproducibility easier and to reduce the
amount of boilerplate required by researchers. 

Benchmarking Functions
----------------------

The fundamental benchmarking functions are :func:`~postbound.experiments.executor.execute_workload` and
:func:`~postbound.experiments.executor.optimize_and_execute_workload`. Both run a collection of benchmark queries on a
database system and the latter function also optimizes the queries using an :doc:`OptimizationPipeline <optimization>`.
The results are provided as a standard
`Pandas DataFrame <https://pandas.pydata.org/docs/user_guide/dsintro.html#dataframe>`_ that can be used for further
analysis or to pass into other data science tools such as `Matplotlib <https://matplotlib.org/>`_.

.. tip::
    To export the benchmark results in a CSV file, make sure to call :func:`~postbound.experiments.executor.prepare_export`
    first. This function handles the conversion of all complex objects to an equivalent JSON representation that can be
    easily deserialized later on.


Query Preparation
-----------------

The benchmark execution utilities provide many parameters to customize repetitions, progress logging, etc.
One important such option is the *query preparation*. Preparation consists of (optional) preprocessing steps that are
applied to each database system or to each individual query just before it is executed. For example, query preparation can
be used to execute all queries as *EXPLAIN ANALYZE* to capture the query plans along with important runtime statistics.
Likewise, the shared buffer of the database system can be modified to simulate a perfectly prepared page cache which
prevents disk I/O from influencing the overall query execution time (i.e. a hot-start experiment).
All of these modifications are specified in the :class:`~postbound.experiments.executor.QueryPreparationService`.
Note that for the pre-warming of the shared buffer to work, the target database system needs to support it. This is
indicated by the :class:`~postbound.db.PrewarmingSupport` protocol. All database interfaces that allow simulated hot starts
implement this protocol. Notably, this includes the Postgres interface.


Utilities
---------

To make setting up common benchmarks easier and to aid reproducibility, PostBOUND ships a number of utility scripts to
quickly create common databases such as IMDB, Stats, StackOverflow or SSB. The setup scripts are system-dependent and
located in the ``db-support/`` directory. Due to the current development focus on Postgres, these scripts are most complete
and most stable for this database system.

For example, to setup a new JOB/IMDB instance on a new Postgres server, the following commands are all that is necessary:

.. code-block:: bash

    cd db-support/postgres
    ./postgres-setup.sh --stop --pg-ver 17
    . ./postgres-start.sh
    ./workload-job-setup.sh

In terms of Postgres support, we also provide a simple configuration tool based on
`PGTune <https://pgtune.leopard.in.ua/>`_ to optimize the server parameters for the current hardware.
See :doc:`postgres` for more details.


.. rubric:: Footnotes

.. [#eval-metrics] Actually, that's not entirely true. It does not have to be the query execution time, but this is
                   currently the most commonly used metric. The general idea is that we need measurements from an actual
                   database system to assesss different optimization algorithms. But these measurements could also capture
                   other metrics e.g., memory footprint. Use the metric that best fits your analysis but always make sure
                   that your testing on a real database system.