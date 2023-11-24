"""Stages define the interfaces of optimization strategies that can be used in the optimization pipelines.

To develop custom optimization algorithms and make use of PostBOUND's pipeline abstraction, the stages are what needs to be
implemented. They specify the basic interface that pipelines expect and provide additional information about the selected
strategies. Depending on the specific pipeline type, different stages have to be implemented. Refer to the documentation of
the respective pipelines for details.
"""
from __future__ import annotations

import abc
from typing import Optional

from postbound.db import db
from postbound.qal import qal
from postbound.optimizer import jointree, physops, planparams, validation


class CompleteOptimizationAlgorithm(abc.ABC):
    """Constructs an entire query plan for an input query in one integrated optimization process.

    This stage closely models the behaviour of traditional optimization algorithms, e.g. based on dynamic programming.
    """

    @abc.abstractmethod
    def optimize_query(self, query: qal.SqlQuery) -> jointree.PhysicalQueryPlan:
        """Constructs the optimized execution plan for an input query.

        Parameters
        ----------
        query : qal.SqlQuery
            The query to optimize

        Returns
        -------
        jointree.PhysicalQueryPlan
            The optimized query plan
        """
        raise NotImplementedError

    @abc.abstractmethod
    def describe(self) -> dict:
        """Provides a JSON-serializable representation of the specific strategy, as well as important parameters.

        Returns
        -------
        dict
            The description

        See Also
        --------
        postbound.postbound.OptimizationPipeline.describe
        """
        raise NotImplementedError

    def pre_check(self) -> validation.OptimizationPreCheck:
        """Provides requirements that an input query has to satisfy in order for the optimizer to work properly.

        Returns
        -------
        validation.OptimizationPreCheck
            The check instance. Can be an empty check if no specific requirements exist.
        """
        return validation.EmptyPreCheck()


class JoinOrderOptimization(abc.ABC):
    """The join order optimization generates a complete join order for an input query.

    This is the first step in a two-stage optimizer design.
    """

    @abc.abstractmethod
    def optimize_join_order(self, query: qal.SqlQuery
                            ) -> Optional[jointree.LogicalJoinTree | jointree.PhysicalQueryPlan]:
        """Performs the actual join ordering process.

        The join tree can be further annotated with an initial operator assignment, if that is an inherent part of
        the specific optimization strategy.

        Other than the join order and operator assignment, the algorithm should add as much information to the join
        tree as possible, e.g. including join conditions and cardinality estimates that were calculated for the
        selected joins. This enables other parts of the optimization process to re-use that information.

        Parameters
        ----------
        query : qal.SqlQuery
            The query to optimize

        Returns
        -------
        Optional[jointree.LogicalJoinTree | jointree.PhysicalQueryPlan]
            The join order. If for some reason there is no valid join order for the given query (e.g. queries with just a
            single selected table), `None` can be returned. Otherwise, the selected join order has to be described using a
            `JoinTree`.
        """
        raise NotImplementedError

    @abc.abstractmethod
    def describe(self) -> dict:
        """Provides a JSON-serializable representation of the specific strategy, as well as important parameters.

        Returns
        -------
        dict
            The description

        See Also
        --------
        postbound.postbound.OptimizationPipeline.describe
        """
        raise NotImplementedError

    def pre_check(self) -> validation.OptimizationPreCheck:
        """Provides requirements that an input query has to satisfy in order for the optimizer to work properly.

        Returns
        -------
        validation.OptimizationPreCheck
            The check instance. Can be an empty check if no specific requirements exist.
        """
        return validation.EmptyPreCheck()

    def __repr__(self) -> str:
        return str(self)

    def __str__(self) -> str:
        return type(self).__name__


class JoinOrderOptimizationError(RuntimeError):
    """Error to indicate that something went wrong while optimizing the join order.

    Parameters
    ----------
    query : qal.SqlQuery
        The query for which the optimization failed
    message : str, optional
        A message containing more details about the specific error. Defaults to an empty string.
    """

    def __init__(self, query: qal.SqlQuery, message: str = "") -> None:
        super().__init__(f"Join order optimization failed for query {query}" if not message else message)
        self.query = query


class PhysicalOperatorSelection(abc.ABC):
    """The physical operator selection assigns scan and join operators to the tables of the input query.

    This is the second stage in the two-phase optimization process, and takes place after the join order has been determined.
    """

    @abc.abstractmethod
    def select_physical_operators(self, query: qal.SqlQuery,
                                  join_order: Optional[jointree.LogicalJoinTree | jointree.PhysicalQueryPlan]
                                  ) -> physops.PhysicalOperatorAssignment:
        """Performs the operator assignment.

        Parameters
        ----------
        query : qal.SqlQuery
            The query to optimize
        join_order : Optional[jointree.LogicalJoinTree  |  jointree.PhysicalQueryPlan]
            The selected join order of the query

        Returns
        -------
        physops.PhysicalOperatorAssignment
            The operator assignment. If for some reason no operators can be assigned, an empty assignment can be returned

        Notes
        -----
        The operator selection should handle a number of different special cases:

        - if no join ordering has been performed, or no valid join order exists the join tree might be `None`.
        - if the join order optimization algorithm already provided an initial choice of physical operators, this
          assignment can be further customized or overwritten entirely by the physical operator selection strategy. The
          initial assignment is contained in the provided `PhysicalQueryPlan`.

        Depending on the specific optimization settings, it is also possible to raise an error if such a situation occurs and
        there is no reasonable way to deal with it.
        """
        raise NotImplementedError

    @abc.abstractmethod
    def describe(self) -> dict:
        """Provides a JSON-serializable representation of the specific strategy, as well as important parameters.

        Returns
        -------
        dict
            The description

        See Also
        --------
        postbound.postbound.OptimizationPipeline.describe
        """
        raise NotImplementedError

    def pre_check(self) -> validation.OptimizationPreCheck:
        """Provides requirements that an input query has to satisfy in order for the optimizer to work properly.

        Returns
        -------
        validation.OptimizationPreCheck
            The check instance. Can be an empty check if no specific requirements exist.
        """
        return validation.EmptyPreCheck()

    def __repr__(self) -> str:
        return str(self)

    def __str__(self) -> str:
        return type(self).__name__


class ParameterGeneration(abc.ABC):
    """The parameter generation assigns additional metadata to a query plan.

    Such parameters do not influence the previous choice of join order and physical operators directly, but affect their
    specific implementation. Therefore, this is an optional final step in a two-stage optimization process.
    """

    @abc.abstractmethod
    def generate_plan_parameters(self, query: qal.SqlQuery,
                                 join_order: Optional[jointree.LogicalJoinTree | jointree.PhysicalQueryPlan],
                                 operator_assignment: Optional[physops.PhysicalOperatorAssignment]
                                 ) -> planparams.PlanParameterization:
        """Executes the actual parameterization.

        Parameters
        ----------
        query : qal.SqlQuery
            The query to optimize
        join_order : Optional[jointree.LogicalJoinTree  |  jointree.PhysicalQueryPlan]
            The selected join order for the query.
        operator_assignment : Optional[physops.PhysicalOperatorAssignment]
            The selected operators for the query

        Returns
        -------
        planparams.PlanParameterization
            The parameterization. If for some reason no parameters can be determined, an empty parameterization can be returned

        Notes
        -----
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
        """Provides a JSON-serializable representation of the specific strategy, as well as important parameters.

        Returns
        -------
        dict
            The description

        See Also
        --------
        postbound.postbound.OptimizationPipeline.describe
        """
        raise NotImplementedError

    def pre_check(self) -> validation.OptimizationPreCheck:
        """Provides requirements that an input query has to satisfy in order for the optimizer to work properly.

        Returns
        -------
        validation.OptimizationPreCheck
            The check instance. Can be an empty check if no specific requirements exist.
        """
        return validation.EmptyPreCheck()

    def __repr__(self) -> str:
        return str(self)

    def __str__(self) -> str:
        return type(self).__name__


class IncrementalOptimizationStep(abc.ABC):
    """Incremental optimization allows to chain different smaller optimization strategies.

    Each step receives the query plan of its predecessor and can change its decisions in arbitrary ways. For example, this
    scheme can be used to gradually correct mistakes or risky decisions of individual optimizers.
    """

    @abc.abstractmethod
    def optimize_query(self, query: qal.SqlQuery,
                       current_plan: jointree.PhysicalQueryPlan) -> jointree.PhysicalQueryPlan:
        """Determines the next query plan.

        If no further optimization steps are configured in the pipeline, this is also the final query plan.

        Parameters
        ----------
        query : qal.SqlQuery
            The query to optimize
        current_plan : jointree.PhysicalQueryPlan
            The execution plan that has so far been built by predecessor strategies. If this step is the first step in the
            optimization pipeline, this might also be a plan from the target database system

        Returns
        -------
        jointree.PhysicalQueryPlan
            The optimized plan
        """
        raise NotImplementedError

    @abc.abstractmethod
    def describe(self) -> dict:
        """Provides a JSON-serializable representation of the specific strategy, as well as important parameters.

        Returns
        -------
        dict
            The description

        See Also
        --------
        postbound.postbound.OptimizationPipeline.describe
        """
        raise NotImplementedError

    def pre_check(self) -> validation.OptimizationPreCheck:
        """Provides requirements that an input query has to satisfy in order for the optimizer to work properly.

        Returns
        -------
        validation.OptimizationPreCheck
            The check instance. Can be an empty check if no specific requirements exist.
        """
        return validation.EmptyPreCheck()


class _CompleteAlgorithmEmulator(CompleteOptimizationAlgorithm):
    """Utility to use implementations of staged optimization strategies when a complete algorithm is expected.

    The emulation is enabled by supplying ``None`` values at all places where the stage expects input from previous stages.
    The output of the actual stage is used to obtain a query plan which in turn is used to generate the required optimizer
    information.

    Parameters
    ----------
    database : Optional[db.Database], optional
        The database for which the queries should be executed. This is required to obtain complete query plans for the input
        queries. If omitted, the database is inferred from the database pool.
    join_order_optimizer : Optional[JoinOrderOptimization], optional
        The join order optimizer if any.
    operator_selection : Optional[PhysicalOperatorSelection], optional
        The physical operator selector if any.
    plan_parameterization : Optional[ParameterGeneration], optional
        The plan parameterization (e.g. cardinality estimator) if any.

    Raises
    ------
    ValueError
        If all stages are ``None``.

    """
    def __init__(self, database: Optional[db.Database] = None, *,
                 join_order_optimizer: Optional[JoinOrderOptimization] = None,
                 operator_selection: Optional[PhysicalOperatorSelection] = None,
                 plan_parameterization: Optional[ParameterGeneration] = None) -> None:
        super().__init__()
        self.database = database if database is not None else db.DatabasePool.get_instance().current_database()
        if all(stage is None for stage in (join_order_optimizer, operator_selection, plan_parameterization)):
            raise ValueError("Exactly one stage has to be given")
        self._join_order_optimizer = join_order_optimizer
        self._operator_selection = operator_selection
        self._plan_parameterization = plan_parameterization

    def stage(self) -> JoinOrderOptimization | PhysicalOperatorSelection | ParameterGeneration:
        """Provides the actually specified stage.

        Returns
        -------
        JoinOrderOptimization | PhysicalOperatorSelection | ParameterGeneration
            The optimization stage.
        """
        return (self._join_order_optimizer if self._join_order_optimizer is not None
                else (self._operator_selection if self._operator_selection is not None
                      else self._plan_parameterization))

    def optimize_query(self, query: qal.SqlQuery) -> jointree.PhysicalQueryPlan:
        join_order = (self._join_order_optimizer.optimize_join_order(query)
                      if self._join_order_optimizer is not None else None)
        physical_operators = (self._operator_selection.select_physical_operators(query, None)
                              if self._operator_selection is not None else None)
        plan_params = (self._plan_parameterization.generate_plan_parameters(query, None, None)
                       if self._plan_parameterization is not None else None)
        hinted_query = self.database.hinting().generate_hints(query, join_order, physical_operators, plan_params)
        query_plan = self.database.optimizer().query_plan(hinted_query)
        return jointree.PhysicalQueryPlan(query_plan, query)

    def describe(self) -> dict:
        return self.stage().describe()

    def pre_check(self) -> validation.OptimizationPreCheck:
        return self.stage().pre_check()


def as_complete_algorithm(stage: JoinOrderOptimization | PhysicalOperatorSelection | ParameterGeneration, *,
                          database: Optional[db.Database] = None) -> CompleteOptimizationAlgorithm:
    """Enables using a partial optimization stage in situations where a complete optimizer is expected.

    This emulation is achieved by using the partial stage to obtain a partial query plan. The target database system is then
    tasked with filling the gaps to construct a complete execution plan.

    Basically this method is syntactic sugar in situations where a `TwoStageOptimizationPipeline` would be filled with only a
    single stage. Using `as_complete_algorithm`, the construction of an entire pipeline can be omitted. Furthermore it can seem
    more natural to "convert" the stage into a complete algorithm in this case.

    Parameters
    ----------
    stage : JoinOrderOptimization | PhysicalOperatorSelection | ParameterGeneration
        The stage that should become a complete optimization algorithm
    database : Optional[db.Database], optional
        The target database to execute the optimized queries in. This is required to fill the gaps of the partial query plans.
        If the database is omitted, it will be inferred based on the database pool.

    Returns
    -------
    CompleteOptimizationAlgorithm
        A emulated optimization algorithm for the optimization stage
    """
    join_order_optimizer = stage if isinstance(stage, JoinOrderOptimization) else None
    operator_selection = stage if isinstance(stage, PhysicalOperatorSelection) else None
    parameter_generation = stage if isinstance(stage, ParameterGeneration) else None
    return _CompleteAlgorithmEmulator(database, join_order_optimizer=join_order_optimizer,
                                      operator_selection=operator_selection, plan_parameterization=parameter_generation)
