"""The optimizer package defines the central interfaces to implement optimization algorithms.

TODO: detailed documentation
"""

#
# Important note for maintainers:
# since we now use lazy loading for the optimization algorithms, we need some additional scaffolding.
# Specifically, we need an additional __init__.pyi file which contains all imports that we normally do in this package
# (lazy and otherwise). This file is only used by type checkers to resolve the lazy modules correctly.
# All changes to the imports below must also be reflected in the __init__.pyi file
# See https://scientific-python.org/specs/spec-0001/#usage and https://scientific-python.org/specs/spec-0001/#type-checkers
# for details.
#

import lazy_loader

from .. import _validation as validation
from ._cardinalities import (
    CardinalityDistortion,
    PreciseCardinalityHintGenerator,
    PreComputedCardinalities,
)
from ._helpers import (
    explode_query_plan,
    read_jointree_json,
    read_operator_assignment_json,
    read_operator_json,
    read_plan_params_json,
    read_query_plan_json,
    to_query_plan,
    update_plan,
)
from ._joingraph import (
    IndexInfo,
    JoinGraph,
    JoinPath,
    TableInfo,
)

# lazy import setup
submodules = [
    "dynprog",
    "enumeration",
    "native",
    "noopt",
    "presets",
    "randomized",
    "tonic",
    "ues",
]

__getattr__, __dir__, _ = lazy_loader.attach(__name__, submodules)

__all__ = [
    "validation",
    "CardinalityDistortion",
    "PreciseCardinalityHintGenerator",
    "PreComputedCardinalities",
    "read_operator_json",
    "read_operator_assignment_json",
    "read_plan_params_json",
    "update_plan",
    "read_jointree_json",
    "to_query_plan",
    "read_query_plan_json",
    "explode_query_plan",
    "JoinGraph",
    "JoinPath",
    "IndexInfo",
    "TableInfo",
]

__all__ += submodules
