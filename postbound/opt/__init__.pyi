# Type stubs for postbound.optimizer package
# See comment in __init__.py for details.

from .. import _validation as validation
from .._jointree import (
    JoinTree,
    explode_query_plan,
    jointree_from_plan,
    parameters_from_plan,
    read_jointree_json,
    read_query_plan_json,
    to_query_plan,
)

# Lazy-loaded modules
from . import dynprog, enumeration, native, noopt, presets, randomized, tonic, ues
from ._cardinalities import (
    CardinalityDistortion,
    PreciseCardinalityHintGenerator,
    PreComputedCardinalities,
)
from ._hints import (
    HintType,
    operators_from_plan,
    read_operator_assignment_json,
    read_operator_json,
    read_plan_params_json,
    update_plan,
)
from ._joingraph import (
    IndexInfo,
    JoinGraph,
    JoinPath,
    TableInfo,
)

__all__ = [
    "validation",
    "CardinalityDistortion",
    "PreciseCardinalityHintGenerator",
    "PreComputedCardinalities",
    "read_operator_json",
    "operators_from_plan",
    "parameters_from_plan",
    "read_operator_assignment_json",
    "read_plan_params_json",
    "update_plan",
    "HintType",
    "JoinTree",
    "jointree_from_plan",
    "read_jointree_json",
    "to_query_plan",
    "read_query_plan_json",
    "explode_query_plan",
    "JoinGraph",
    "JoinPath",
    "IndexInfo",
    "TableInfo",
    "dynprog",
    "enumeration",
    "native",
    "noopt",
    "presets",
    "randomized",
    "tonic",
    "ues",
]
