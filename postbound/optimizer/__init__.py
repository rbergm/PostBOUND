"""The optimizer package defines the central interfaces to implement optimization algorithms.

These include:

- a representation for logical join trees and physical query execution plans in the `jointree` module
- a representation for join graphs in the `joingraph` module
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

from .._qep import (
  PlanEstimates,
  PlanMeasures,
  PlanParams,
  QueryPlan,
  SortKey,
  Subplan,
)
from ._hints import (
  DirectionalJoinOperatorAssignment,
  HintType,
  JoinOperator,
  JoinOperatorAssignment,
  PhysicalOperator,
  PhysicalOperatorAssignment,
  PlanParameterization,
  ScanOperator,
  ScanOperatorAssignment,
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
from ._jointree import (
  JoinTree,
  LogicalJoinTree,
  explode_query_plan,
  jointree_from_plan,
  parameters_from_plan,
  read_jointree_json,
  read_query_plan_json,
  to_query_plan,
)

__all__ = [
  "policies",
  "ScanOperator",
  "JoinOperator",
  "PhysicalOperator",
  "ScanOperatorAssignment",
  "JoinOperatorAssignment",
  "DirectionalJoinOperatorAssignment",
  "read_operator_json",
  "PhysicalOperatorAssignment",
  "PlanParameterization",
  "operators_from_plan",
  "parameters_from_plan",
  "read_operator_assignment_json",
  "read_plan_params_json",
  "update_plan",
  "HintType",
  "JoinTree",
  "LogicalJoinTree",
  "jointree_from_plan",
  "read_jointree_json",
  "to_query_plan",
  "read_query_plan_json",
  "explode_query_plan",
  "SortKey",
  "PlanParams",
  "PlanEstimates",
  "PlanMeasures",
  "Subplan",
  "QueryPlan",
  "JoinGraph",
  "JoinPath",
  "IndexInfo",
  "TableInfo",
]
