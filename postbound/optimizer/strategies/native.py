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

import math
from collections.abc import Iterable
from typing import Optional

from .._hints import PhysicalOperatorAssignment, PlanParameterization, operators_from_plan
from .._jointree import JoinTree, jointree_from_plan, parameters_from_plan
from ..._core import Cost, Cardinality, TableReference
from ..._qep import QueryPlan
from ..._stages import (
    CostModel,
    JoinOrderOptimization, PhysicalOperatorSelection, ParameterGeneration,
    CompleteOptimizationAlgorithm
)
from ...db import DatabaseServerError, DatabaseUserError
from ..policies.cardinalities import CardinalityHintsGenerator
from ... import db, qal, util
from ...qal import SqlQuery
from ...util import jsondict


class NativeCostModel(CostModel):
    """Obtains the cost of a query plan by using the cost model of an actual database system.

    Parameters
    ----------
    raise_on_error : bool
        Whether the cost model should raise an error if anything goes wrong during the estimation. For example, this can
        happen if the query plan cannot be executed on the target database system. If this is off (the default), failure
        results in an infinite cost.
    """

    def __init__(self, raise_on_error: bool = False) -> None:
        super().__init__()
        self._target_db: Optional[db.Database] = None
        self._raise_on_error = raise_on_error

    def estimate_cost(self, query: SqlQuery, plan: QueryPlan) -> Cost:
        hinted_query = self._target_db.hinting().generate_hints(query, plan.with_actual_card())
        if self._raise_on_error:
            cost = self._target_db.optimizer().cost_estimate(hinted_query)
        else:
            try:
                cost = self._target_db.optimizer().cost_estimate(hinted_query)
            except DatabaseServerError | DatabaseUserError:
                cost = math.inf
        return cost

    def describe(self) -> jsondict:
        return {"name": "native", "database_system": self._target_db.describe()}

    def initialize(self, target_db: db.Database, query: SqlQuery) -> None:
        self._target_db = target_db

    def cleanup(self) -> None:
        self._target_db = None


class NativeCardinalityEstimator(CardinalityHintsGenerator):
    """Obtains the cardinality of a query plan by using the cardinality estimator of an actual database system."""

    def __init__(self, target_db: Optional[db.Database] = None) -> None:
        super().__init__(True)
        self._target_db: Optional[db.Database] = target_db

    def calculate_estimate(self, query: SqlQuery, intermediate: TableReference | Iterable[TableReference]) -> Cardinality:
        intermediate = util.enlist(intermediate)
        subquery = qal.transform.extract_query_fragment(query, intermediate)
        return self._target_db.optimizer().cardinality_estimate(subquery)

    def describe(self) -> jsondict:
        return {"name": "native", "database_system": self._target_db.describe()}

    def initialize(self, target_db: db.Database, query: SqlQuery) -> None:
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

    def optimize_join_order(self, query: SqlQuery) -> Optional[JoinTree]:
        query_plan = self.db_instance.optimizer().query_plan(query)
        return jointree_from_plan(query_plan)

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

    def select_physical_operators(self, query: SqlQuery, join_order: Optional[JoinTree]) -> PhysicalOperatorAssignment:
        if join_order:
            query = self.db_instance.hinting().generate_hints(query, join_order=join_order)
        query_plan = self.db_instance.optimizer().query_plan(query)
        return operators_from_plan(query_plan)

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

    def generate_plan_parameters(self, query: SqlQuery, join_order: Optional[JoinTree],
                                 operator_assignment: Optional[PhysicalOperatorAssignment]) -> Optional[PlanParameterization]:
        if join_order or operator_assignment:
            query = self.db_instance.hinting().generate_hints(query, join_order=join_order,
                                                              physical_operators=operator_assignment)
        query_plan = self.db_instance.optimizer().query_plan(query)
        parameters_from_plan(query_plan)

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

    def optimize_query(self, query: SqlQuery) -> QueryPlan:
        return self.db_instance.optimizer().query_plan(query)

    def describe(self) -> jsondict:
        return {"name": "native", "database_system": self.db_instance.describe()}
