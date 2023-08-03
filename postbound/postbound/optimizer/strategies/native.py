"""Provides access to the optimization decisions made by the native database system optimizer"""
from __future__ import annotations

from typing import Optional

from postbound.db import db
from postbound.qal import qal
from postbound.optimizer import jointree, physops, planparams, stages


class NativeJoinOrderOptimizer(stages.JoinOrderOptimization):

    def __init__(self, db_instance: db.Database) -> None:
        super().__init__()
        self.db_instance = db_instance

    def optimize_join_order(self, query: qal.SqlQuery) -> Optional[jointree.PhysicalQueryPlan]:
        query_plan = self.db_instance.optimizer().query_plan(query)
        return jointree.PhysicalQueryPlan.load_from_query_plan(query_plan, query)

    def describe(self) -> dict:
        return {"name": "native", "database_system": self.db_instance.describe()}


class NativePhysicalOperatorSelection(stages.PhysicalOperatorSelection):

    def __init__(self, db_instance: db.Database) -> None:
        super().__init__()
        self.db_instance = db_instance

    def select_physical_operators(self, query: qal.SqlQuery,
                                  join_order: Optional[jointree.LogicalJoinTree | jointree.PhysicalQueryPlan]
                                  ) -> physops.PhysicalOperatorAssignment:
        if join_order:
            query = self.db_instance.hinting().generate_hints(query, join_order)
        query_plan = self.db_instance.optimizer().query_plan(query)
        join_tree = jointree.PhysicalQueryPlan.load_from_query_plan(query_plan)
        return join_tree.physical_operators()

    def describe(self) -> dict:
        return {"name": "native", "database_system": self.db_instance.describe()}


class NativePlanParameterization(stages.ParameterGeneration):

    def __init__(self, db_instance: db.Database) -> None:
        super().__init__()
        self.db_instance = db_instance

    def generate_plan_parameters(self, query: qal.SqlQuery,
                                 join_order: Optional[jointree.LogicalJoinTree | jointree.PhysicalQueryPlan],
                                 operator_assignment: Optional[physops.PhysicalOperatorAssignment]
                                 ) -> Optional[planparams.PlanParameterization]:
        if join_order or operator_assignment:
            query = self.db_instance.hinting().generate_hints(query, join_order, operator_assignment)
        query_plan = self.db_instance.optimizer().query_plan(query)
        join_tree = jointree.PhysicalQueryPlan.load_from_query_plan(query_plan)
        return join_tree.plan_parameters()

    def describe(self) -> dict:
        return {"name": "native", "database_system": self.db_instance.describe()}
