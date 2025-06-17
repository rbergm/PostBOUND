Optimizer Abstraction
=====================

.. ipython:: python

    import postbound as pb
    pg_instance = pb.postgres.connect(config_file=".psycopg_connection")
    pg_instance

.. _textbook-optimizer:

Textbook Optimizer Pipeline
---------------------------

The textbook optimizer pipeline models the traditional optimizer architecture consisting of plan enumerator, cost model,
and cardinality estimator.
The plan enumerator generates a set of candidate plans.
They are evaluated by the cost model and ultimately the plan with the lowest cost is selected.
The cost model uses cardinality estimates as its principal input to assess how much work each plan will require.

You can create a textbook pipeline by simply supplying the target database:

.. ipython:: python

    pipeline = pb.TextBookOptimizationPipeline(pg_instance)

See the :class:`TextBookOptimizationPipeline <postbound.TextBookOptimizationPipeline>` for
a full rundown of all available methods.

.. tip::

    If you do not implement your own plan enumerator, PostBOUND will select a dynamic programming-based algorithm (see
    :class:`DynamicProgrammingEnumerator <postbound.optimizer.strategies.dynprog.DynamicProgrammingEnumerator>`) by
    default.
    If your target database happens to be a Postgres system, a DP algorithm specifically designed to mimic the Postgres
    algorithm will be used (see :class:`PostgresDynProg <postbound.optimizer.strategies.dynprog.PostgresDynProg>`).

    In contrast to the :ref:`multi-stage pipeline <multistage-optimizer>`, we cannot simply let the target database system
    supply its own enumerator, because it is typically the enumerators job to request cost and cardinality estimates.
    This would mean that the target database system would need some way to call back into PostBOUND to obtain these
    estimates.
    While that is certainly an exciting possibility, we do currently not have the resources to investigate it further.


.. _multistage-optimizer:

Multi-stage Optimizer Pipeline
------------------------------
