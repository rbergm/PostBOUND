from __future__ import annotations

import abc

from postbound.qal import qal
from postbound.optimizer import data
from postbound.optimizer.physops import operators


class PhysicalOperatorSelection(abc.ABC):

    @abc.abstractmethod
    def select_physical_operators(self, query: qal.ImplicitSqlQuery,
                                  join_order: data.JoinTree | None) -> operators.PhysicalOperatorAssignment:
        raise NotImplementedError


class UESOperatorSelection(PhysicalOperatorSelection):
    def select_physical_operators(self, query: qal.ImplicitSqlQuery,
                                  join_order: data.JoinTree | None) -> operators.PhysicalOperatorAssignment:
        assignment = operators.PhysicalOperatorAssignment(query)
        assignment.set_operator_enabled_globally(operators.JoinOperators.NestedLoopJoin, False)
        return assignment


class EmptyPhysicalOperatorSelection(PhysicalOperatorSelection):
    def select_physical_operators(self, query: qal.ImplicitSqlQuery,
                                  join_order: data.JoinTree | None) -> operators.PhysicalOperatorAssignment:
        return operators.PhysicalOperatorAssignment(query)
