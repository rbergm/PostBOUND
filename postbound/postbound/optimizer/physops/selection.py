from __future__ import annotations

import abc

from postbound.qal import qal
from postbound.optimizer import data
from postbound.optimizer.physops import operators


class PhysicalOperatorSelection(abc.ABC):

    def __init__(self) -> None:
        self.next_selection: PhysicalOperatorSelection | None = None

    def select_physical_operators(self, query: qal.ImplicitSqlQuery,
                                  join_order: data.JoinTree | None) -> operators.PhysicalOperatorAssignment:
        assignment = self._apply_selection(query, join_order)
        if self.next_selection:
            next_assignment = self.next_selection.select_physical_operators(query, join_order)
            assignment = assignment.merge_with(next_assignment)
        return assignment

    def chain_with(self, next_selection: PhysicalOperatorSelection) -> None:
        self.next_selection = next_selection

    @abc.abstractmethod
    def _apply_selection(self, query: qal.ImplicitSqlQuery,
                         join_order: data.JoinTree | None) -> operators.PhysicalOperatorAssignment:
        raise NotImplementedError


class UESOperatorSelection(PhysicalOperatorSelection):
    def _apply_selection(self, query: qal.ImplicitSqlQuery,
                         join_order: data.JoinTree | None) -> operators.PhysicalOperatorAssignment:
        assignment = operators.PhysicalOperatorAssignment(query)
        assignment.set_operator_enabled_globally(operators.JoinOperators.NestedLoopJoin, False)
        return assignment


class EmptyPhysicalOperatorSelection(PhysicalOperatorSelection):
    def _apply_selection(self, query: qal.ImplicitSqlQuery,
                         join_order: data.JoinTree | None) -> operators.PhysicalOperatorAssignment:
        return operators.PhysicalOperatorAssignment(query)
