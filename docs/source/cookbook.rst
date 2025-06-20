Cookbook
========

The cookbook demonstrates how to perform certain, frequently used patterns.
Throughout the examples, we use the following setup:

.. ipython:: python

    import postbound as pb
    pg_instance = pb.postgres.connect(config_file=".psycopg_connection")
    stats = pb.workloads.stats()


.. _cookbook-cardinality-estimation:

Cardinality estimation
----------------------

TODO


.. _cookbook-partial-hinting:

Manual hinting
--------------

TODO


.. _cookbook-postgres-plans:

Postgres Query Plans
--------------------

When working with Postgres, there are three basic ways to access query plans:

1. You can retrieve the raw plan JSON using a plain :meth:`execute_query() <postbound.db.postgres.PostgresInterface.execute_query>`
2. You can parse a raw plan into a :class:`PostgresExplainPlan <postbound.db.postgres.PostgresExplainPlan>`, which is pretty
   much a 1:1 model of the raw plan with more expressive attribute access and some high-level access methods
3. You can convert an explain into a proper normalized :class:`QueryPlan <postbound.optimizer.QueryPlan>` object

The conversion between the different formats works as follows:

.. ipython:: python

    query = stats["q-10"]
    explain_query = pb.transform.as_explain(query)
    raw_plan = pg_instance.execute_query(explain_query)
    raw_plan
    postgres_plan = pb.postgres.PostgresExplainPlan(raw_plan)
    print(postgres_plan.inspect())
    qep = postgres_plan.as_qep()
    print(qep.inspect())

.. _jsonize:

JSON export
-----------


Miscellaneous utilities
-----------------------

simplify

enlist

