"""Native strategies obtain execution plans from actual database management systems.

Instead of performing optimizations on their own, the native stages delegate all decisions to a specific database system.
Afterwards, they analyze the query plan and encode the relevant information in a stage-specific format.

Notes
-----
By combining native stages with different target database systems, the optimizers of the respective systems can be combined.
For example, combining a join ordering stage with an Oracle backend and an operator selection stage with a Postgres backend
would provide a combined query optimizer with Oracle's join ordering algorithm and Postgres' operator selection.
"""
from __future__ import annotations

from typing import Optional

from postbound.qal import qal
from .. import jointree, physops, planparams, stages
from ... import db


class NativeJoinOrderOptimizer(stages.JoinOrderOptimization):
    """Obtains the join order for an input query by using the optimizer of an actual database system.

    Parameters
    ----------
    db_instance : db.Database
        The target database whose optimization algorithm should be used.
    """

    def __init__(self, db_instance: db.Database) -> None:
        super().__init__()
        self.db_instance = db_instance

    def optimize_join_order(self, query: qal.SqlQuery) -> Optional[jointree.LogicalJoinTree]:
        query_plan = self.db_instance.optimizer().query_plan(query)
        return jointree.LogicalJoinTree.load_from_query_plan(query_plan, query)

    def describe(self) -> dict:
        return {"name": "native", "database_system": self.db_instance.describe()}


class NativePhysicalOperatorSelection(stages.PhysicalOperatorSelection):
    """Obtains the physical operators for an input query by using the optimizer of an actual database system.

    Since this process normally is the second stage in the optimization pipeline, the operators are selected according to a
    specific join order. If no such order exists, it is also determined by the database system.

    Parameters
    ----------
    db_instance : db.Database
        The target database whose optimization algorithm should be used.
    """

    def __init__(self, db_instance: db.Database) -> None:
        super().__init__()
        self.db_instance = db_instance

    def select_physical_operators(self, query: qal.SqlQuery,
                                  join_order: Optional[jointree.LogicalJoinTree | jointree.PhysicalQueryPlan]
                                  ) -> physops.PhysicalOperatorAssignment:
        if join_order:
            query = self.db_instance.hinting().generate_hints(query, join_order)
        query_plan = self.db_instance.optimizer().query_plan(query)
        join_tree = jointree.PhysicalQueryPlan.load_from_query_plan(query_plan, query, operators_only=True)
        return join_tree.physical_operators()

    def describe(self) -> dict:
        return {"name": "native", "database_system": self.db_instance.describe()}


class NativePlanParameterization(stages.ParameterGeneration):
    """Obtains the plan parameters for an inpuit querry by using the optimizer of an actual database system.

    This process determines the parameters according to a join order and physical operators. If no such information exists, it
    is also determined by the database system.

    Parameters
    ----------
    db_instance : db.Database
        The target database whose optimization algorithm should be used.
    """

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
        join_tree = jointree.PhysicalQueryPlan.load_from_query_plan(query_plan, query)
        return join_tree.plan_parameters()

    def describe(self) -> dict:
        return {"name": "native", "database_system": self.db_instance.describe()}


class NativeOptimizer(stages.CompleteOptimizationAlgorithm):
    """Obtains a complete query execution plan by using the optimizer of an actual database system.

    Parameters
    ----------
    db_instance : db.Database
        The target database whose optimization algorithm should be used.
    """

    def __init__(self, db_instance: db.Database) -> None:
        super().__init__()
        self.db_instance = db_instance

    def optimize_query(self, query: qal.SqlQuery) -> jointree.PhysicalQueryPlan:
        query_plan = self.db_instance.optimizer().query_plan(query)
        return jointree.PhysicalQueryPlan.load_from_query_plan(query_plan)

    def describe(self) -> dict:
        return {"name": "native", "database_system": self.db_instance.describe()}
