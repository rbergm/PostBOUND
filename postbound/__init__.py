"""PostBOUND - A research framework for query optimization in relational database systems.

PostBOUND allows to rapidly prototype novel ideas in query optimization and to evaluate them in a transparent and
reproducible manner. On a high level, the framework uses the following concepts:

- **optimization pipelines** provide models for different optimizer architecures. Each pipeline provides different *hooks*
  (called *optimization stages*) where users can plug in their own optimization strategies.
- **database backends** enable the evaluation of optimization pipelines on real-world database systems. Backends translate
  the optimization decisions into system-specific query hints that enforce the selected execution plan at runtime.
- **workloads and benchmarking utilities** allow to evaluate optimization pipelines on popular benchmarks in a reproducible
  manner.
- additional **infrastructure modules** handle the boilerplate part of query optimization, such as parsing and representing
  SQL queries, retrieving schema information, or obtaining database statistics.


Documentation
-------------

Visit https://postbound.readthedocs.io/ for the full documentation of PostBOUND, including guides, tutorials and API
references.

Package Structure
-----------------

- Core data structures and utilities are globally available (e.g. `OptimizationPipeline`, `Database`, or `SqlQuery`)
- SQL representation and parsing is handled by the `qal` and `parser` modules
- Query modification is implemented in the `transform` module
- The database interface is defined in the `db` module. `postgres` and `duckdb` provide concrete backends
- Workloads and benchmarking utilities are available in `workloads` and `bench`
- Simple optimization algorithms (e.g. dynamic-programming-based plan enumeration) are provided in the `opt` module
"""

import lazy_loader

__getattr__, __dir__, __all__ = lazy_loader.attach_stub(__name__, __file__)

__version__ = "0.21.0-dev"
