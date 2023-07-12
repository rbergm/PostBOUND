from __future__ import annotations

import abc
from typing import Optional

from postbound.qal import qal
from postbound.optimizer import jointree, physops, planparams, validation


class CompleteOptimizationAlgorithm(abc.ABC):

    @abc.abstractmethod
    def optimize_query(self, query: qal.SqlQuery) -> jointree.PhysicalQueryPlan:
        raise NotImplementedError

    @abc.abstractmethod
    def describe(self) -> dict:
        raise NotImplementedError

    def pre_check(self) -> validation.OptimizationPreCheck:
        return validation.EmptyPreCheck()


class JoinOrderOptimization(abc.ABC):
    """The `JoinOrderOptimizer` handles the entire process of obtaining a join order for input queries.

    The join ordering is the first step in the optimization process. Therefore, the implemented optimization strategy
    can apply an entire green-field approach.
    """

    @abc.abstractmethod
    def optimize_join_order(self, query: qal.SqlQuery
                            ) -> Optional[jointree.LogicalJoinTree | jointree.PhysicalQueryPlan]:
        """Performs the actual join ordering process.

        If for some reason there is no valid join order for the given query (e.g. queries with just a single selected
        table), `None` can be returned. Otherwise, the selected join order has to be described using a `JoinTree`.

        The join tree can be further annotated with an initial operator assignment, if that is an inherent part of
        the specific optimization strategy (e.g. for integrated optimization algorithms that are used in many
        real-world systems).

        Other than the join order and operator assignment, the algorithm should add as much information to the join
        tree as possible, e.g. including join conditions and cardinality estimates that were calculated for the
        selected joins. This ensures that other parts of the code base work as expected.
        """
        raise NotImplementedError

    @abc.abstractmethod
    def describe(self) -> dict:
        """Provides a representation of the join order optimization strategy."""
        raise NotImplementedError

    def pre_check(self) -> validation.OptimizationPreCheck:
        """Provides requirements that an input query has to satisfy in order for the optimizer to work properly."""
        return validation.EmptyPreCheck()

    def __repr__(self) -> str:
        return str(self)

    def __str__(self) -> str:
        return type(self).__name__


class JoinOrderOptimizationError(RuntimeError):
    """Error to indicate that something went wrong while optimizing the join order."""

    def __init__(self, query: qal.SqlQuery, message: str = "") -> None:
        super().__init__(f"Join order optimization failed for query {query}" if not message else message)
        self.query = query


class PhysicalOperatorSelection(abc.ABC):
    """The `PhysicalOperatorSelection` assigns scan and join operators to the tables of the input query.

    This is the second stage of the optimization process, after the join order has been determined. This means that,
    depending on the actual input query, the operator selector can be confronted with different situations. Details
    are outlined in the `select_physical_operators` method.

    Multiple assignment strategies can be combined using the `chain_with` method. If this is the case, the current
    selector will transfer its assignment to the next strategy, such that this strategy can further customize or
    overwrite the previous selection.
    """

    def __init__(self) -> None:
        self.next_selection: PhysicalOperatorSelection | None = None

    def select_physical_operators(self, query: qal.SqlQuery,
                                  join_order: Optional[jointree.LogicalJoinTree | jointree.PhysicalQueryPlan]
                                  ) -> physops.PhysicalOperatorAssignment:
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
        join_order = (join_order.as_logical_join_tree() if isinstance(join_order, jointree.PhysicalQueryPlan)
                      else join_order)
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
                         join_order: Optional[jointree.LogicalJoinTree | jointree.PhysicalQueryPlan]
                         ) -> physops.PhysicalOperatorAssignment:
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

    def __repr__(self) -> str:
        return str(self)

    def __str__(self) -> str:
        return type(self).__name__


# TODO: Refactor: get rid of chaining behavior built into the selection stratgies themselves.
# Instead, introduce a new ChainedParameterizationStrategy


class ParameterGeneration(abc.ABC):
    """The `ParameterGeneration` assigns additional meta parameters to a query plan.

    Such parameters do not influence the previous choice of join order and physical operators directly, but affects
    their specific implementation. Therefore, this is the final stage of the optimization process.

    Currently, PostBOUND supports two such parameters: parallelization parameters that influence how many workers
    should be used to execute a specific operator and cardinality hints that steer the choice of join order and
    physical operators of the native database system for cases where the optimization pipeline did not make a final
    decision itself.

    Multiple parameterization strategies can be combined using the `chain_with` method. If this is the case, the
    current generator will transfer its assignment to the next strategy, such that this strategy can further customize
    or overwrite the previous parameters.
    """

    def __init__(self) -> None:
        self.next_generator: ParameterGeneration | None = None

    @abc.abstractmethod
    def generate_plan_parameters(self, query: qal.SqlQuery,
                                 join_order: Optional[jointree.LogicalJoinTree | jointree.PhysicalQueryPlan],
                                 operator_assignment: Optional[physops.PhysicalOperatorAssignment]
                                 ) -> Optional[planparams.PlanParameterization]:
        """Executes the actual parameterization.

        Since this is the final stage of the optimization process, a number of special cases have to be handled:

        - the previous phases might not have determined any join order or operator assignment
        - there might not have been a physical operator selection, but only a join ordering (which potentially included
        an initial selection of physical operators)
        - there might not have been a join order optimization, but only a selection of physical operators
        - both join order and physical operators might have been optimized (in which case only the actual operator
        assignment matters, not any assignment contained in the join order)
        """
        raise NotImplementedError

    def chain_with(self, next_generator: ParameterGeneration) -> ParameterGeneration:
        """Combines the current parameterization strategy with a followup assignment algorithm.

        After the current strategy has finished its parameter generation, the followup algorithm can further customize
        the assignment and overwrite some or even all of the parameters.

        The returned parameter generator should be used for all future accesses to the generation strategy.
        """
        if self.next_generator:
            next_generator.chain_with(self.next_generator)
        self.next_generator = next_generator
        return self

    @abc.abstractmethod
    def describe(self) -> dict:
        """Provides a representation of the parameter generation strategy."""
        raise NotImplementedError

    def pre_check(self) -> Optional[validation.OptimizationPreCheck]:
        """Provides requirements that an input query has to satisfy in order for the generator to work properly."""
        return None

    def __repr__(self) -> str:
        return str(self)

    def __str__(self) -> str:
        return type(self).__name__
