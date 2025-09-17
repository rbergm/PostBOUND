"""Contains utilities to conveniently execute individual queries or entire workloads and to evaluate their results.

This module provides direct access to some frequently-used functionality, mostly related to workload modelling and execution.
Other modules need to be imported explicitly.

Specifically, this package provides the following modules:

- `analysis` provides a loose collection of utilities and formulas somewhat related to query optimization
- `querygen` provides a simple random query generator
- `ceb` provides an implementation of the Cardinality Estimation Benchmark workload generator
- `interactive` contains a simple interactive join order optimizer
"""

from .executor import (
    QueryPreparationService,
    execute_query,
    optimize_and_execute_query,
    prepare_export,
)

__all__ = [
    "QueryPreparationService",
    "execute_query",
    "optimize_and_execute_query",
    "prepare_export",
]
