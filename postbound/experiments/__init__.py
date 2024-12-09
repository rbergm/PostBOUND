"""Contains utilities to conveniently execute individual queries or entire workloads and to evaluate their results.

This module provides direct access to some frequently-used functionality, mostly related to workload modelling and execution.
Other modules need to be imported explicitly.

Specifically, this package provides the following modules:

- `analysis` provides a loose collection of utilities and formulas somewhat related to query optimization
- `ceb` provides an implementation of the Cardinality Estimation Benchmark workload generator
- `executor` contains utilities to benchmark individual queries or entire workloads
- `interactive` contains a simple interactive join order optimizer
- `workloads` provides an abstraction for workloads (collections of queries for benchmarking) along with tools to read them
"""

from .executor import (
    QueryPreparationService,
    execute_query, optimize_and_execute_query,
    execute_workload, optimize_and_execute_workload,
    prepare_export
)
from .workloads import Workload, read_workload, read_csv_workload, read_batch_workload

__all__ = [
    "QueryPreparationService",
    "execute_query", "optimize_and_execute_query",
    "execute_workload", "optimize_and_execute_workload",
    "prepare_export",
    "Workload", "read_workload", "read_csv_workload", "read_batch_workload"
]
