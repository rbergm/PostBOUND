"""Contains the abstractions for the parameterization of query plans."""
from __future__ import annotations

import abc

from typing import Optional

from postbound.qal import qal
from postbound.optimizer import jointree, validation
from postbound.optimizer.physops import operators as physops
from postbound.optimizer.planmeta import hints as plan_params


# TODO: implement chaining rules

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
                                 ) -> Optional[plan_params.PlanParameterization]:
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


class EmptyParameterization(ParameterGeneration):
    """Dummy implementation of the plan parameterization that does not actually generate any parameters."""

    def generate_plan_parameters(self, query: qal.SqlQuery,
                                 join_order: Optional[jointree.LogicalJoinTree | jointree.PhysicalQueryPlan],
                                 operator_assignment: Optional[physops.PhysicalOperatorAssignment]
                                 ) -> Optional[plan_params.PlanParameterization]:
        return None

    def describe(self) -> dict:
        return {"name": "no_parameterization"}
