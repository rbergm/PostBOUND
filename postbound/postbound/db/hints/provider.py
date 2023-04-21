"""Contains the services that enable the generation of database-specific hints for optimized queries."""
from __future__ import annotations

import abc

from postbound.qal import qal
from postbound.optimizer import data
from postbound.optimizer.physops import operators
from postbound.optimizer.planmeta import hints as plan_params


class HintProvider(abc.ABC):
    """Basic interface to generate query hints from join orders and operator selections."""

    @abc.abstractmethod
    def adapt_query(self, query: qal.SqlQuery, *, join_order: data.JoinTree | None = None,
                    physical_operators: operators.PhysicalOperatorAssignment | None = None,
                    plan_parameters: plan_params.PlanParameterization | None = None) -> qal.SqlQuery:
        """Generates the appropriate hints to enforce the optimized execution of the given query.

        All hints are placed in a `Hint` clause on the query. In addition, if the query needs to be transformed in some
        way, this also happens here.
        """
        raise NotImplementedError
