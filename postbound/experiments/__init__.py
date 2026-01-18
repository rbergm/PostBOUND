"""Contains simple tools to generate query workloads for experiments.

- `querygen` provides a simple random query generator
- `ceb` provides an implementation of the Cardinality Estimation Benchmark workload generator
"""

import warnings

import lazy_loader

warnings.warn(
    "The 'postbound.experiments' module is deprecated and will be moved to the "
    "separate optimizer repository with version 0.21.0",
    category=DeprecationWarning,
    stacklevel=2,
)

__getattr__, __dir__, __all__ = lazy_loader.attach_stub(__name__, __file__)
