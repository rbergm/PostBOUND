"""Provides empty (dummy) strategies for the individual optimization stages."""
from __future__ import annotations

from typing import Optional

from postbound.qal import qal
from postbound.optimizer import jointree, physops, planparams, stages


class EmptyJoinOrderOptimizer(stages.JoinOrderOptimization):
    """Dummy implementation of the join order optimizer that does not actually optimize anything."""

    def __init__(self) -> None:
        super().__init__()

    def optimize_join_order(self, query: qal.SqlQuery) -> Optional[jointree.LogicalJoinTree]:
        return None

    def describe(self) -> dict:
        return {"name": "no_ordering"}


class EmptyPhysicalOperatorSelection(stages.PhysicalOperatorSelection):
    """Dummy implementation of operator optimization that does not actually optimize anything."""

    def chain_with(self, next_selection: stages.PhysicalOperatorSelection) -> stages.PhysicalOperatorSelection:
        return next_selection

    def _apply_selection(self, query: qal.SqlQuery,
                         join_order: Optional[jointree.LogicalJoinTree | jointree.PhysicalQueryPlan]
                         ) -> physops.PhysicalOperatorAssignment:
        return physops.PhysicalOperatorAssignment()

    def _description(self) -> dict:
        return {"name": "no_selection"}


class EmptyParameterization(stages.ParameterGeneration):
    """Dummy implementation of the plan parameterization that does not actually generate any parameters."""

    def generate_plan_parameters(self, query: qal.SqlQuery,
                                 join_order: Optional[jointree.LogicalJoinTree | jointree.PhysicalQueryPlan],
                                 operator_assignment: Optional[physops.PhysicalOperatorAssignment]
                                 ) -> Optional[planparams.PlanParameterization]:
        return None

    def describe(self) -> dict:
        return {"name": "no_parameterization"}
