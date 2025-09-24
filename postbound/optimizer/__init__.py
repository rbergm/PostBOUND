"""The optimizer package defines the central interfaces to implement optimization algorithms.

These include:

- a representation for join trees and join graphs
- interfaces for policies that can be used to parameterize actual optimization algorithms, e.g. to inject different cardinality
  estimation strategies. These are contained in the `policies` package and need to be imported explicitly.
- utilities to load pre-configured optimization strategies into optimization pipelines in the `presets` module
- supporting code to ensure that custom optimization strategies are only applied to supported databases and queries using the
  `validation` module
- a collection of published optimization algorithms in the `strategies` package

The most important data structures are made available directly from this package. This includes all parts of physical query
plans and basic hinting support. All other modules need to be imported explicitly. Especially, this applies to the `presets`
module and the `strategies` package.
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
    "policies",
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
