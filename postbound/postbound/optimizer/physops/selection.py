from __future__ import annotations

import abc

from postbound.qal import qal
from postbound.optimizer import data
from postbound.optimizer.physops import operators
from postbound.db.systems import systems


class PhysicalOperatorSelection(abc.ABC):

    def __init__(self, target_system: systems.DatabaseSystem) -> None:
        self.next_selection: PhysicalOperatorSelection | None = None
        self.target_system = target_system

    def select_physical_operators(self, query: qal.ImplicitSqlQuery,
                                  join_order: data.JoinTree | None) -> operators.PhysicalOperatorAssignment:
        assignment = self._apply_selection(query, join_order)
        if self.next_selection:
            next_assignment = self.next_selection.select_physical_operators(query, join_order)
            assignment = assignment.merge_with(next_assignment)
        return assignment

    def chain_with(self, next_selection: PhysicalOperatorSelection) -> None:
        self.next_selection = next_selection

    def describe(self) -> dict:
        description = self._description()
        if self.next_selection:
            description["next_selection"] = self.next_selection.describe()
        return description

    @abc.abstractmethod
    def _apply_selection(self, query: qal.ImplicitSqlQuery,
                         join_order: data.JoinTree | None) -> operators.PhysicalOperatorAssignment:
        raise NotImplementedError

    @abc.abstractmethod
    def _description(self) -> dict:
        raise NotImplementedError


class UESOperatorSelection(PhysicalOperatorSelection):

    def __init__(self, target_system: systems.DatabaseSystem) -> None:
        super().__init__(target_system)

    def _apply_selection(self, query: qal.ImplicitSqlQuery,
                         join_order: data.JoinTree | None) -> operators.PhysicalOperatorAssignment:
        assignment = operators.PhysicalOperatorAssignment(query)
        if self.target_system.supports_hint(operators.JoinOperators.NestedLoopJoin):
            assignment.set_operator_enabled_globally(operators.JoinOperators.NestedLoopJoin, False)
        return assignment

    def _description(self) -> dict:
        return {"name": "ues"}


class EmptyPhysicalOperatorSelection(PhysicalOperatorSelection):

    def _apply_selection(self, query: qal.ImplicitSqlQuery,
                         join_order: data.JoinTree | None) -> operators.PhysicalOperatorAssignment:
        return operators.PhysicalOperatorAssignment(query)

    def _description(self) -> dict:
        return {"name": "no_selection"}

    def _apply_selection(self, query: qal.ImplicitSqlQuery,
                         join_order: data.JoinTree | None) -> operators.PhysicalOperatorAssignment:
        return operators.PhysicalOperatorAssignment(query)


class UnsupportedSystemError(RuntimeError):
    def __init__(self, db_system: systems.DatabaseSystem, reason: str = "") -> None:
        error_msg = f"Unsupported database system: {db_system}"
        if reason:
            error_msg += f" ({reason})"
        super().__init__(error_msg)
        self.db_system = db_system
        self.reason = reason
