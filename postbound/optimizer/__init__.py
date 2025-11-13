"""The optimizer package defines the central interfaces to implement optimization algorithms.

TODO: detailed documentation
"""

from .. import _validation as validation
from .._hints import (
    DirectionalJoinOperatorAssignment,
    HintType,
    JoinOperatorAssignment,
    ScanOperatorAssignment,
    operators_from_plan,
    read_operator_assignment_json,
    read_operator_json,
    read_plan_params_json,
    update_plan,
)
from .._jointree import (
    JoinTree,
    explode_query_plan,
    jointree_from_plan,
    parameters_from_plan,
    read_jointree_json,
    read_query_plan_json,
    to_query_plan,
)
from ._cardinalities import (
    CardinalityDistortion,
    PreciseCardinalityHintGenerator,
    PreComputedCardinalities,
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
    "ScanOperatorAssignment",
    "JoinOperatorAssignment",
    "DirectionalJoinOperatorAssignment",
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
]
