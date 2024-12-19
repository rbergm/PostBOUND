"""The optimizer package defines the central interfaces to implement optimization algorithms.

These include:

- a representation for logical join trees and physical query execution plans in the `jointree` module
- a representation for join graphs in the `joingraph` module
- interfaces for policies that can be used to parameterize actual optimization algorithms, e.g. to inject different cardinality
  estimation strategies. These are contained in the `policies` package
- utilities to load pre-configured optimization strategies into optimization pipelines in the `presets` module
- supporting code to ensure that custom optimization strategies are only applied to supported databases and queries using the
  `validation` module
- a collection of published optimization algorithms in the `strategies` package

The most important data structures are made available directly from this package. This includes all parts of physical query
plans and basic hinting support. All other modules need to be imported explicitly. Especially, this applies to the `presets`
module and the `strategies` package.
"""

from . import joingraph, jointree, policies
from ._hints import (
  ScanOperators,
  JoinOperators,
  PhysicalOperator,
  ScanOperatorAssignment,
  JoinOperatorAssignment,
  DirectionalJoinOperatorAssignment,
  read_operator_json,
  PhysicalOperatorAssignment,
  PlanParameterization,
  HintType
)
from .jointree import (
  JoinTreeVisitor,
  PhysicalJoinMetadata, PhysicalBaseTableMetadata, PhysicalPlanMetadata,
  PhysicalQueryPlan,
  LogicalJoinMetadata, LogicalBaseTableMetadata, LogicalTreeMetadata,
  LogicalJoinTree,
  read_from_json
)

__all__ = [
  "joingraph",
  "jointree",
  "policies",
  "ScanOperators",
  "JoinOperators",
  "PhysicalOperator",
  "ScanOperatorAssignment",
  "JoinOperatorAssignment",
  "DirectionalJoinOperatorAssignment",
  "read_operator_json",
  "PhysicalOperatorAssignment",
  "PlanParameterization",
  "HintType",
  "JoinTreeVisitor",
  "PhysicalJoinMetadata",
  "PhysicalBaseTableMetadata",
  "PhysicalPlanMetadata",
  "PhysicalQueryPlan",
  "LogicalJoinMetadata",
  "LogicalBaseTableMetadata",
  "LogicalTreeMetadata",
  "LogicalJoinTree",
  "read_from_json"
]
