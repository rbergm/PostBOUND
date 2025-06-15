10 minutes to PostBOUND
=======================

Inspired by the `10 minutes to Pandas <https://pandas.pydata.org/docs/user_guide/10min.html>`__ tutorial, this is a quick
introduction to PostBOUND, geared towards new users.
It assumes that you have a basic understanding of Python and a decent understanding of relational databases, especially
query optimization.

The primary target audience for PostBOUND are database researchers that want to implement their own ideas for query
optimization and evaluate them on real-world data.
To learn how to install PostBOUND see the :doc:`Setup Guide<setup>` guide.

By convention, we use the following imports in this tutorial:

.. ipython:: python

    import pandas as pd
    import postbound as pb


Optimization Pipelines
-----------------------

Optimization pipelines are at the core of PostBOUND.
They provide a simple model for optimization workflows oriented around different optimizer architectures.
Each pipeline has different interfaces that allow to customize different parts of the optimization process (these are
*stages* in PostBOUND-lingo).
Implementing a new optimization algorithm boils down to selecting the appropriate optimization pipeline and stage.

The two most commonly used pipelines are:

The :class:`TextBookOptimizationPipeline <postbound.TextBookOptimizationPipeline>`, which is modelled after the traditional
query optimizer architecture.
It uses a :class:`PlanEnumerator <postbound.PlanEnumerator>` to produce different candidate plans and ultimately select
the best one.
To assess the quality of each candidate plan, a  :class:`CostModel <postbound.CostModel>` is used.
The cost model typically uses the :class:`CardinalityEstimator <postbound.CardinalityEstimator>` to estimate the number of
rows produced by each operator in the plan.

.. tip::

    If you are only concerned with cardinality estimation, it is best to implement the
    :class:`CardinalityGenerator <postbound.CardinalityGenerator>` instead of the
    :class:`CardinalityEstimator <postbound.CardinalityEstimator>`.
    The reasoning is explained in the the :ref:`Cookbook <cardinality-estimation>`.

The :class:`MultiStageOptimizationPipeline <postbound.MultiStageOptimizationPipeline>` performs the query optimization in
multiple sequential steps.
Initially, it computes a join order using the :class:`JoinOrderOptimization <postbound.JoinOrderOptimization>` stage.
Afterwards, it selects the best physical operators for the join order in the
:class:`PhysicalOperatorSelection <postbound.PhysicalOperatorSelection>` stage.
Finally, the :class:`ParameterGeneration <postbound.ParameterGeneration>` can be used to add additional metadata to the
query plan.
This stage is especially well-suited for optimization scenarios where only part of the decisions of the native optimizer
are overridden.
For example, you can implement your own join ordering and cardinality estimator and leave the operator selection to the
database system (given your cardinality estimates).
See the :ref:`10min-db-connection` for how this actually works under the hood.

.. note::

    Users do not need to implement all stages of a pipeline.
    Instead, PostBOUND automatically "fills the gaps" with reasonable defaults.
    This allows users to focus only on the parts of the optimization process that are relevant for their research.
    For example, if you want to implement a new cardinality estimator in the textbook pipeline, PostBOUND will
    automatically use the cost model and plan enumerator of the target database system.

.. _10min-db-connection:

Database connection
-------------------

A key philosophy of PostBOUND is to always execute queries on real database systems instead of research prototypes or
simulated environments.
We treat the query execution time as the ultimate measure of quality of a query plan.
But, since PostBOUND is implemented as a Python framework, we cannot interfere with the optimizer directly.
Instead, PostBOUND uses query hints to restrict the native optimizer of the database system and to enforce the optimization
decisions made within the framework.

As a consequence, PostBOUND requires a connection to a database system for much of its functionality.
For Postgres, you can connect to the database like so:

.. ipython:: python

    pg_instance = pb.postgres.connect(config_file=".psycopg_connection")
    pg_instance

Here, the ``config_file`` parameter points to a file that contains the connection parameters as a
`psycopg-compatible <https://www.psycopg.org/psycopg3/docs/api/connections.html#psycopg.Connection.connect>`__ string.

.. note::

    PostgreSQL does not provide hinting support out-of-the-box.
    Therefore, PostBOUND uses the `pg_hint_plan <https://github.com/ossc-db/pg_hint_plan>`__ extension to add query hints.
    If you set up your own Postgres instance, make sure to install the extension.
    As an alternative, you can use `pg_lab <https://github.com/rbergm/pg_lab>`__, which extends Postgres with more advanced
    hinting capabilities and additional extension points for optimizer research.


Workload handling
-----------------

A :class:`Workload <postbound.experiments.Workload>` is a collection of queries that can be used to benchmark the
performance of different optimization strategies.
All queries are associated with labels that are typically used to retrieve them, e.g., ``job["1a"]``.
A workload provides rich functionality to retrieve (subsets of) the queries, such as by specific properties or randomly to
obtain a test set.

Following the *batteries included* philosophy, PostBOUND already ships some of the commonly used workloads in query
optimization.
These can be accessed from the :mod:`postbound.workloads` module.
Specifically, the Join Order Benchmark (JOB), the Stats Benchmark and the Stack Benchmark are available out-of-the-box:

.. ipython:: python

    stats = pb.workloads.stats()
    stats

You can also load your own workloads by using :func:`read_workload() <postbound.experiments.workloads.read_workload>` or
:func:`read_csv_workload() <postbound.experiments.workloads.read_csv_workload>`.

Benchmarking
------------

Once you have implemented you own optimization algorithm, you can benchmark it using the
:func:`execute_workload() <postbound.experiments.executor.execute_workload>` and
:func:`optimize_and_execute_workload() <postbound.experiments.executor.optimize_and_execute_workload>` utilities.

Both take provide a pandas DataFrame with the results of the executed queries:

.. ipython:: python

    results = pb.execute_workload(stats.first(3), pg_instance)
    results

If you want to export the results to a CSV file, you can use
:func:`prepare_export() <postbound.experiments.executor.prepare_export>` to serialize all columns to JSON as necessary.

The :class:`QueryPreparationService <postbound.experiments.executor.QueryPreparationService>` enables you to customize the
execution of the queries.
For example, you can ensure that all queries are executed as *EXPLAIN ANALYZE* to capture their query plans, or you can
prewarm the shared buffer before execution to ensure that timing measurements are not affected by I/O activity.
