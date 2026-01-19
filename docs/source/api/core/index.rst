Core Functionality
==================

The core data structures and abstractions are available at the top-level of the package, i.e. simply via ``postbound``
as in ``postbound.TableReference``. We list the core abstractions here. In addition, several important components such as
the hinting infrastructure and optimization pipelines are described in separate documents:

.. toctree::
    :maxdepth: 1

    optimization-pipelines
    plans
    hinting

Core Abstractions
-----------------

.. automodule:: postbound

.. autoclass:: postbound.TableReference
    :members:

.. autoclass:: postbound.ColumnReference
    :members:

.. autoclass:: postbound.ScanOperator
    :members:

.. autoclass:: postbound.JoinOperator
    :members:

.. autoclass:: postbound.IntermediateOperator
    :members:

.. autodata:: postbound.PhysicalOperator

.. autodata:: postbound.Cost

.. autoclass:: postbound.Cardinality
    :members:

.. autoclass:: postbound.SqlQuery
    :members:

.. autofunction:: postbound.parse_query

.. autoclass:: postbound.Database
    :members:

.. autoclass:: postbound.Workload
    :members:
