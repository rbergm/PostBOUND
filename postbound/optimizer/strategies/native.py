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

from .._hints import PhysicalOperatorAssignment, PlanParameterization
from ..jointree import LogicalJoinTree, PhysicalQueryPlan
from ..._core import Cost, Cardinality, TableReference
from ..._stages import (
    CostModel, CardinalityEstimator,
    JoinOrderOptimization, PhysicalOperatorSelection, ParameterGeneration,
    CompleteOptimizationAlgorithm
)
from ... import db, qal
from ...util import jsondict


class NativeCostModel(CostModel):
    """Obtains the cost of a query plan by using the cost model of an actual database system."""

    def __init__(self) -> None:
        super().__init__()
        self._target_db: Optional[db.Database] = None

    def estimate_cost(self, query: qal.SqlQuery, plan: PhysicalQueryPlan) -> Cost:
        hinted_query = self._target_db.hinting().generate_hints(query, plan)
        return self._target_db.optimizer().cost_estimate(hinted_query)

    def describe(self) -> jsondict:
        return {"name": "native", "database_system": self._target_db.describe()}

    def initialize(self, target_db: db.Database, query: qal.SqlQuery) -> None:
        self._target_db = target_db

    def cleanup(self) -> None:
        self._target_db = None


class NativeCardinalityEstimator(CardinalityEstimator):
    """Obtains the cardinality of a query plan by using the cardinality estimator of an actual database system."""

    def __init__(self) -> None:
        super().__init__()
        self._target_db: Optional[db.Database] = None

    def estimate_cardinality(self, query: qal.SqlQuery, intermediate: frozenset[TableReference]) -> Cardinality:
        subquery = qal.transform.extract_query_fragment(query, intermediate)
        return self._target_db.optimizer().cardinality_estimate(subquery)

    def describe(self) -> jsondict:
        return {"name": "native", "database_system": self._target_db.describe()}

    def initialize(self, target_db: db.Database, query: qal.SqlQuery) -> None:
        self._target_db = target_db

    def cleanup(self) -> None:
        self._target_db = None


class NativeJoinOrderOptimizer(JoinOrderOptimization):
    """Obtains the join order for an input query by using the optimizer of an actual database system.

    Parameters
    ----------
    db_instance : db.Database
        The target database whose optimization algorithm should be used.
    """

    def __init__(self, db_instance: db.Database) -> None:
        super().__init__()
        self.db_instance = db_instance

    def optimize_join_order(self, query: qal.SqlQuery) -> Optional[LogicalJoinTree]:
        query_plan = self.db_instance.optimizer().query_plan(query)
        return LogicalJoinTree.load_from_query_plan(query_plan, query)

    def describe(self) -> jsondict:
        return {"name": "native", "database_system": self.db_instance.describe()}


class NativePhysicalOperatorSelection(PhysicalOperatorSelection):
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
                                  join_order: Optional[LogicalJoinTree]) -> PhysicalOperatorAssignment:
        if join_order:
            query = self.db_instance.hinting().generate_hints(query, join_order)
        query_plan = self.db_instance.optimizer().query_plan(query)
        join_tree = PhysicalQueryPlan.load_from_query_plan(query_plan, query, operators_only=True)
        return join_tree.physical_operators()

    def describe(self) -> jsondict:
        return {"name": "native", "database_system": self.db_instance.describe()}


class NativePlanParameterization(ParameterGeneration):
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

    def generate_plan_parameters(self, query: qal.SqlQuery, join_order: Optional[LogicalJoinTree | PhysicalQueryPlan],
                                 operator_assignment: Optional[PhysicalOperatorAssignment]) -> Optional[PlanParameterization]:
        if join_order or operator_assignment:
            query = self.db_instance.hinting().generate_hints(query, join_order, operator_assignment)
        query_plan = self.db_instance.optimizer().query_plan(query)
        join_tree = PhysicalQueryPlan.load_from_query_plan(query_plan, query)
        return join_tree.plan_parameters()

    def describe(self) -> jsondict:
        return {"name": "native", "database_system": self.db_instance.describe()}


class NativeOptimizer(CompleteOptimizationAlgorithm):
    """Obtains a complete query execution plan by using the optimizer of an actual database system.

    Parameters
    ----------
    db_instance : db.Database
        The target database whose optimization algorithm should be used.
    """

    def __init__(self, db_instance: db.Database) -> None:
        super().__init__()
        self.db_instance = db_instance

    def optimize_query(self, query: qal.SqlQuery) -> PhysicalQueryPlan:
        query_plan = self.db_instance.optimizer().query_plan(query)
        return PhysicalQueryPlan.load_from_query_plan(query_plan)

    def describe(self) -> jsondict:
        return {"name": "native", "database_system": self.db_instance.describe()}
