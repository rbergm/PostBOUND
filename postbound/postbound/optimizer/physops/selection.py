"""Contains the abstractions for the physical operator selection."""
from __future__ import annotations

import abc
from typing import Optional

from postbound.qal import qal
from postbound.optimizer import data, validation
from postbound.optimizer.physops import operators
from postbound.db.systems import systems


class PhysicalOperatorSelection(abc.ABC):
    """The `PhysicalOperatorSelection` assigns scan and join operators to the tables of the input query.

    This is the second stage of the optimization process, after the join order has been determined. This means that,
    depending on the actual input query, the operator selector can be confronted with different situations. Details
    are outlined in the `select_physical_operators` method.

    Multiple assignment strategies can be combined using the `chain_with` method. If this is the case, the current
    selector will transfer its assignment to the next strategy, such that this strategy can further customize or
    overwrite the previous selection.
    """

    def __init__(self, target_system: systems.DatabaseSystem) -> None:
        self.next_selection: PhysicalOperatorSelection | None = None
        self.target_system = target_system

    def select_physical_operators(self, query: qal.SqlQuery,
                                  join_order: data.JoinTree | None) -> operators.PhysicalOperatorAssignment:
        """Performs the operator assignment.

        The operator selection should handle a number of different special cases:

        - if for some reason no operators can be assigned, an empty assignment can be returned
        - if no join ordering has been performed, the join tree might be `None`. In that case, the join order
        optimization should be executed by the native database system during query execution
        - if the join order optimization algorithm already provided an initial choice of physical operators, this
        assignment can be further customized or overwritten entirely by the physical operator selection strategy. The
        initial assignment is contained in the provided `JoinTree`.

        In the end, the final operator assignment is returned by this method. At this point, eventual initial
        assignments contained in the join tree do not matter anymore.
        """
        assignment = self._apply_selection(query, join_order)
        if self.next_selection:
            next_assignment = self.next_selection.select_physical_operators(query, join_order)
            assignment = assignment.merge_with(next_assignment)
        return assignment

    def chain_with(self, next_selection: PhysicalOperatorSelection) -> PhysicalOperatorSelection:
        """Combines the current selection strategy with a followup assignment algorithm.

        After the current strategy has finished its operator selection, the followup algorithm can further customize
        the assignment and overwrite some or all of the choices.

        The returned selection strategy should be used for all future accesses to the physical operation strategy.
        """
        if self.next_selection:
            next_selection.chain_with(self.next_selection)
        self.next_selection = next_selection
        return self

    def describe(self) -> dict:
        """Provides a representation of the operator optimization strategy."""
        description = self._description()
        if self.next_selection:
            description["next_selection"] = self.next_selection.describe()
        return description

    def pre_check(self) -> Optional[validation.OptimizationPreCheck]:
        """Provides requirements that an input query has to satisfy in order for the optimizer to work properly."""
        return None

    @abc.abstractmethod
    def _apply_selection(self, query: qal.SqlQuery,
                         join_order: data.JoinTree | None) -> operators.PhysicalOperatorAssignment:
        """Performs the actual assignment of the physical operators and has to be implemented by every strategy.

        The more high-level `select_physical_operators` method also takes care of the chaining rules and delegates the
        assignment to this very method. Its parameters are exactly the same as for the `select_physical_operators`
        method.
        """
        raise NotImplementedError

    @abc.abstractmethod
    def _description(self) -> dict:
        """Provides the actual representation of this operator assignment algorithm.

        This method follows the same pattern as the `select_physical_operators` and `_apply_selection` methods:
        The `describe` method acts as the high-level interface to access the descriptions and handles the chaining
        rules. This method takes care of the actual description.
        """
        raise NotImplementedError


class UESOperatorSelection(PhysicalOperatorSelection):
    """Implementation of the operator selection for the UES algorithm.

    UES is actually not concerned with operator selection and focuses exclusively on join orders. Therefore, this
    strategy simply disables nested loop joins since they can typically lead to performance degradation. Essentially
    this enforces the usage of hash joins for the vast majority of joins in a typical database system since they
    provide the most robust behavior.

    See Hertzschuch et al.: "Simplicity Done Right for Join Ordering", CIDR'2021 for more details.
    """

    def __init__(self, target_system: systems.DatabaseSystem) -> None:
        super().__init__(target_system)

    def _apply_selection(self, query: qal.SqlQuery,
                         join_order: data.JoinTree | None) -> operators.PhysicalOperatorAssignment:
        assignment = (join_order.operator_assignment.clone() if join_order and join_order.operator_assignment
                      else operators.PhysicalOperatorAssignment(query))
        if self.target_system.supports_hint(operators.JoinOperators.NestedLoopJoin):
            assignment.set_operator_enabled_globally(operators.JoinOperators.NestedLoopJoin, False,
                                                     overwrite_fine_grained_selection=True)
        return assignment

    def _description(self) -> dict:
        return {"name": "ues"}


class EmptyPhysicalOperatorSelection(PhysicalOperatorSelection):
    """Dummy implementation of operator optimization that does not actually optimize anything."""

    def chain_with(self, next_selection: PhysicalOperatorSelection) -> PhysicalOperatorSelection:
        return next_selection

    def _apply_selection(self, query: qal.SqlQuery,
                         join_order: data.JoinTree | None) -> operators.PhysicalOperatorAssignment:
        return operators.PhysicalOperatorAssignment(query)

    def _description(self) -> dict:
        return {"name": "no_selection"}
