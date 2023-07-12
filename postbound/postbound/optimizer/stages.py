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
    """

    def __init__(self) -> None:
        self.next_selection: PhysicalOperatorSelection | None = None

    @abc.abstractmethod
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
        raise NotImplementedError

    @abc.abstractmethod
    def describe(self) -> dict:
        """Provides a representation of the operator optimization strategy."""
        raise NotImplementedError

    def pre_check(self) -> Optional[validation.OptimizationPreCheck]:
        """Provides requirements that an input query has to satisfy in order for the optimizer to work properly."""
        return None

    def __repr__(self) -> str:
        return str(self)

    def __str__(self) -> str:
        return type(self).__name__


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


class IncrementalOptimizationStep(abc.ABC):

    @abc.abstractmethod
    def optimize_query(self, query: qal.SqlQuery,
                       current_plan: jointree.PhysicalQueryPlan) -> jointree.PhysicalQueryPlan:
        raise NotImplementedError

    @abc.abstractmethod
    def describe(self) -> dict:
        raise NotImplementedError

    def pre_check(self) -> Optional[validation.OptimizationPreCheck]:
        return None
