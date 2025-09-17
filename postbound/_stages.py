from __future__ import annotations

import abc
from collections.abc import Iterable
from dataclasses import dataclass
from typing import Optional

from . import util
from ._core import Cardinality, Cost, TableReference
from ._hints import PhysicalOperatorAssignment, PlanParameterization
from ._jointree import JoinTree
from ._qep import QueryPlan
from .db._db import Database, DatabasePool
from .qal._qal import SqlQuery
from .util.jsonize import jsondict


class OptimizationPreCheck(abc.ABC):
    """The pre-check interface.

    This is the type that all concrete pre-checks must implement. It contains two check methods that correpond to the checks
    on the database system and to the check on the input query. Both methods pass on all input data by default and must be
    overwritten to execute the necessary checks.

    Parameters
    ----------
    name : str
        The name of the check. It should describe what features the check tests and will be used to represent the checks that
        are present in an optimization pipeline.
    """

    def __init__(self, name: str) -> None:
        self.name = name

    def check_supported_query(self, query: SqlQuery) -> PreCheckResult:
        """Validates that a specific query does not contain any features that cannot be handled by an optimization strategy.

        Examples of such features can be non-equi join predicates, dependent subqueries or aggregations.

        Parameters
        ----------
        query : SqlQuery
            The query to check

        Returns
        -------
        PreCheckResult
            A description of whether the check passed and an indication of the failures.
        """
        return PreCheckResult.with_all_passed()

    def check_supported_database_system(
        self, database_instance: Database
    ) -> PreCheckResult:
        """Validates that a specific database system provides all features that are required by an optimization strategy.

        Examples of such features can be support for cardinality hints or specific operators.

        Parameters
        ----------
        database_instance : Database
            The database to check

        Returns
        -------
        PreCheckResult
            A description of whether the check passed and an indication of the failures.
        """
        return PreCheckResult.with_all_passed()

    @abc.abstractmethod
    def describe(self) -> dict:
        """Provides a JSON-serializable representation of the specific check, as well as important parameters.

        Returns
        -------
        dict
            The description

        See Also
        --------
        postbound.postbound.OptimizationPipeline.describe
        """
        raise NotImplementedError

    def __contains__(self, item: object) -> bool:
        return item == self

    def __hash__(self) -> int:
        return hash(self.name)

    def __eq__(self, other: object) -> bool:
        return isinstance(other, type(self)) and self.name == other.name

    def __repr__(self) -> str:
        return f"OptimizationPreCheck [{self.name}]"

    def __str__(self) -> str:
        return self.name


class EmptyPreCheck(OptimizationPreCheck):
    """Dummy check that does not actually validate anything."""

    def __init__(self) -> None:
        super().__init__("empty")

    def check_supported_query(self, query: SqlQuery) -> PreCheckResult:
        return PreCheckResult.with_all_passed()

    def describe(self) -> dict:
        return {"name": "no_check"}


class CompleteOptimizationAlgorithm(abc.ABC):
    """Constructs an entire query plan for an input query in one integrated optimization process.

    This stage closely models the behaviour of traditional optimization algorithms, e.g. based on dynamic programming.
    """

    @abc.abstractmethod
    def optimize_query(self, query: SqlQuery) -> QueryPlan:
        """Constructs the optimized execution plan for an input query.

        Parameters
        ----------
        query : SqlQuery
            The query to optimize

        Returns
        -------
        QueryPlan
            The optimized query plan
        """
        raise NotImplementedError

    @abc.abstractmethod
    def describe(self) -> jsondict:
        """Provides a JSON-serializable representation of the specific strategy, as well as important parameters.

        Returns
        -------
        jsondict
            The description

        See Also
        --------
        postbound.postbound.OptimizationPipeline.describe
        """
        raise NotImplementedError

    def pre_check(self) -> OptimizationPreCheck:
        """Provides requirements that input query or database system have to satisfy for the optimizer to work properly.

        Returns
        -------
        OptimizationPreCheck
            The check instance. Can be an empty check if no specific requirements exist.
        """
        return EmptyPreCheck()

    def __repr__(self) -> str:
        return str(self)

    def __str__(self) -> str:
        return type(self).__name__


class CardinalityEstimator(abc.ABC):
    """The cardinality estimator calculates how many tuples specific operators will produce.

    See Also
    --------
    postbound.TextBookOptimizationPipeline
    """

    @abc.abstractmethod
    def calculate_estimate(
        self, query: SqlQuery, intermediate: TableReference | Iterable[TableReference]
    ) -> Cardinality:
        """Determines the cardinality of a specific intermediate.

        Parameters
        ----------
        query : SqlQuery
            The query being optimized
        intermediate : TableReference | Iterable[TableReference]
            The intermediate for which the cardinality should be estimated. All filter predicates, etc. that are applicable
            to the intermediate can be assumed to be applied.

        Returns
        -------
        Cardinality
            the estimate
        """
        raise NotImplementedError

    @abc.abstractmethod
    def describe(self) -> jsondict:
        """Provides a JSON-serializable representation of the specific estimator, as well as important parameters.

        Returns
        -------
        jsondict
            The description

        See Also
        --------
        postbound.postbound.OptimizationPipeline.describe
        """
        raise NotImplementedError

    def initialize(self, target_db: Database, query: SqlQuery) -> None:
        """Hook method that is called before the actual optimization process starts.

        This method can be overwritten to set up any necessary data structures, etc. and will be called before each query.

        Parameters
        ----------
        target_db : Database
            The database for which the optimized queries should be generated.
        query : SqlQuery
            The query to be optimized
        """
        pass

    def cleanup(self) -> None:
        """Hook method that is called after the optimization process has finished.

        This method can be overwritten to remove any temporary state that was specific to the last query being optimized
        and should not be shared with later queries.
        """
        pass

    def pre_check(self) -> OptimizationPreCheck:
        """Provides requirements that input query or database system have to satisfy for the optimizer to work properly.

        Returns
        -------
        OptimizationPreCheck
            The check instance. Can be an empty check if no specific requirements exist.
        """
        return EmptyPreCheck()

    def __repr__(self) -> str:
        return str(self)

    def __str__(self) -> str:
        return type(self).__name__


class CostModel(abc.ABC):
    """The cost model estimates how expensive computing a certain query plan is.

    See Also
    --------
    postbound.TextBookOptimizationPipeline
    """

    @abc.abstractmethod
    def estimate_cost(self, query: SqlQuery, plan: QueryPlan) -> Cost:
        """Computes the cost estimate for a specific plan.

        The following conventions are used for the estimation: the root node of the plan will not have any cost set. However,
        all input nodes will have already been estimated by earlier calls to the cost model. Hence, while estimating the cost
        of the root node, all earlier costs will be available as inputs. It is further assumed that all nodes already have
        associated cardinality estimates.
        This method explicitly does not make any assumption regarding the relationship between query and plan. Specifically,
        it does not assume that the plan is capable of computing the entire result set nor a correct result set. Instead,
        the plan might just be a partial plan that computes a subset of the query (e.g. a join of some of the tables).
        It is the implementation's responsibility to figure out the appropriate course of action.

        It is not the responsibility of the cost model to set the estimate on the plan, this is the task of the enumerator
        (which can decide whether the plan should be considered any further).

        Parameters
        ----------
        query : SqlQuery
            The query being optimized
        plan : QueryPlan
            The plan to estimate.

        Returns
        -------
        Cost
            The estimated cost
        """
        raise NotImplementedError

    @abc.abstractmethod
    def describe(self) -> jsondict:
        """Provides a JSON-serializable representation of the specific cost model, as well as important parameters.

        Returns
        -------
        jsondict
            The description

        See Also
        --------
        postbound.postbound.OptimizationPipeline.describe
        """
        raise NotImplementedError

    def initialize(self, target_db: Database, query: SqlQuery) -> None:
        """Hook method that is called before the actual optimization process starts.

        This method can be overwritten to set up any necessary data structures, etc. and will be called before each query.

        Parameters
        ----------
        target_db : Database
            The database for which the optimized queries should be generated.
        query : SqlQuery
            The query to be optimized
        """
        pass

    def cleanup(self) -> None:
        """Hook method that is called after the optimization process has finished.

        This method can be overwritten to remove any temporary state that was specific to the last query being optimized
        and should not be shared with later queries.
        """
        pass

    def pre_check(self) -> OptimizationPreCheck:
        """Provides requirements that input query or database system have to satisfy for the optimizer to work properly.

        Returns
        -------
        OptimizationPreCheck
            The check instance. Can be an empty check if no specific requirements exist.
        """
        return EmptyPreCheck()

    def __repr__(self) -> str:
        return str(self)

    def __str__(self) -> str:
        return type(self).__name__


class PlanEnumerator(abc.ABC):
    """The plan enumerator traverses the space of different candidate plans and ultimately selects the optimal one.

    See Also
    --------
    postbound.TextBookOptimizationPipeline
    """

    @abc.abstractmethod
    def generate_execution_plan(
        self,
        query: SqlQuery,
        *,
        cost_model: CostModel,
        cardinality_estimator: CardinalityEstimator,
    ) -> QueryPlan:
        """Computes the optimal plan to execute the given query.

        Parameters
        ----------
        query : SqlQuery
            The query to optimize
        cost_model : CostModel
            The cost model to compare different candidate plans
        cardinality_estimator : CardinalityEstimator
            The cardinality estimator to calculate the sizes of intermediate results

        Returns
        -------
        QueryPlan
            The query plan

        Notes
        -----
        The precise generation "style" (e.g. top-down vs. bottom-up, complete plans vs. plan fragments, etc.) is completely up
        to the specific algorithm. Therefore, it is really hard to provide a more expressive interface for the enumerator
        beyond just generating a plan. Generally the enumerator should query the cost model to compare different candidates.
        The top-most operator of each candidate will usually not have a cost estimate set at the beginning and it is the
        enumerator's responsibility to set the estimate correctly. The `jointree.update_cost_estimate` function can be used to
        help with this.
        """
        raise NotImplementedError

    @abc.abstractmethod
    def describe(self) -> jsondict:
        """Provides a JSON-serializable representation of the specific enumerator, as well as important parameters.

        Returns
        -------
        jsondict
            The description

        See Also
        --------
        postbound.postbound.OptimizationPipeline.describe
        """
        raise NotImplementedError

    def pre_check(self) -> OptimizationPreCheck:
        """Provides requirements that input query or database system have to satisfy for the optimizer to work properly.

        Returns
        -------
        OptimizationPreCheck
            The check instance. Can be an empty check if no specific requirements exist.
        """
        return EmptyPreCheck()

    def __repr__(self) -> str:
        return str(self)

    def __str__(self) -> str:
        return type(self).__name__


class JoinOrderOptimization(abc.ABC):
    """The join order optimization generates a complete join order for an input query.

    This is the first step in a multi-stage optimizer design.

    See Also
    --------
    postbound.MultiStageOptimizationPipeline
    """

    @abc.abstractmethod
    def optimize_join_order(self, query: SqlQuery) -> Optional[JoinTree]:
        """Performs the actual join ordering process.

        The join tree can be further annotated with an initial operator assignment, if that is an inherent part of
        the specific optimization strategy. However, this is generally discouraged and the multi-stage pipeline will discard
        such operators to prepare for the subsequent physical operator selection.

        Other than the join order and operator assignment, the algorithm should add as much information to the join
        tree as possible, e.g. including join conditions and cardinality estimates that were calculated for the
        selected joins. This enables other parts of the optimization process to re-use that information.

        Parameters
        ----------
        query : SqlQuery
            The query to optimize

        Returns
        -------
        Optional[LogicalJoinTree]
            The join order. If for some reason there is no valid join order for the given query (e.g. queries with just a
            single selected table), `None` can be returned. Otherwise, the selected join order has to be described using a
            `JoinTree`.
        """
        raise NotImplementedError

    @abc.abstractmethod
    def describe(self) -> jsondict:
        """Provides a JSON-serializable representation of the specific strategy, as well as important parameters.

        Returns
        -------
        jsondict
            The description

        See Also
        --------
        postbound.postbound.OptimizationPipeline.describe
        """
        raise NotImplementedError

    def pre_check(self) -> OptimizationPreCheck:
        """Provides requirements that input query or database system have to satisfy for the optimizer to work properly.

        Returns
        -------
        OptimizationPreCheck
            The check instance. Can be an empty check if no specific requirements exist.
        """
        return EmptyPreCheck()

    def __repr__(self) -> str:
        return str(self)

    def __str__(self) -> str:
        return type(self).__name__


class JoinOrderOptimizationError(RuntimeError):
    """Error to indicate that something went wrong while optimizing the join order.

    Parameters
    ----------
    query : SqlQuery
        The query for which the optimization failed
    message : str, optional
        A message containing more details about the specific error. Defaults to an empty string.
    """

    def __init__(self, query: SqlQuery, message: str = "") -> None:
        super().__init__(
            f"Join order optimization failed for query {query}"
            if not message
            else message
        )
        self.query = query


class PhysicalOperatorSelection(abc.ABC):
    """The physical operator selection assigns scan and join operators to the tables of the input query.

    This is the second stage in the two-phase optimization process, and takes place after the join order has been determined.

    See Also
    --------
    postbound.MultiStageOptimizationPipeline
    """

    @abc.abstractmethod
    def select_physical_operators(
        self, query: SqlQuery, join_order: Optional[JoinTree]
    ) -> PhysicalOperatorAssignment:
        """Performs the operator assignment.

        Parameters
        ----------
        query : SqlQuery
            The query to optimize
        join_order : Optional[JoinTree]
            The selected join order of the query

        Returns
        -------
        PhysicalOperatorAssignment
            The operator assignment. If for some reason no operators can be assigned, an empty assignment can be returned

        Notes
        -----
        The operator selection should handle a `None` join order gracefully. This can happen if the query does not require
        any joins (e.g. processing of a single table.

        Depending on the specific optimization settings, it is also possible to raise an error if such a situation occurs and
        there is no reasonable way to deal with it.
        """
        raise NotImplementedError

    @abc.abstractmethod
    def describe(self) -> jsondict:
        """Provides a JSON-serializable representation of the specific strategy, as well as important parameters.

        Returns
        -------
        jsondict
            The description

        See Also
        --------
        postbound.postbound.OptimizationPipeline.describe
        """
        raise NotImplementedError

    def pre_check(self) -> OptimizationPreCheck:
        """Provides requirements that input query or database system have to satisfy for the optimizer to work properly.

        Returns
        -------
        OptimizationPreCheck
            The check instance. Can be an empty check if no specific requirements exist.
        """
        return EmptyPreCheck()

    def __repr__(self) -> str:
        return str(self)

    def __str__(self) -> str:
        return type(self).__name__


class ParameterGeneration(abc.ABC):
    """The parameter generation assigns additional metadata to a query plan.

    Such parameters do not influence the previous choice of join order and physical operators directly, but affect their
    specific implementation. Therefore, this is an optional final step in a multi-stage optimization process.

    See Also
    --------
    postbound.MultiStageOptimizationPipeline
    """

    @abc.abstractmethod
    def generate_plan_parameters(
        self,
        query: SqlQuery,
        join_order: Optional[JoinTree],
        operator_assignment: Optional[PhysicalOperatorAssignment],
    ) -> PlanParameterization:
        """Executes the actual parameterization.

        Parameters
        ----------
        query : SqlQuery
            The query to optimize
        join_order : Optional[JoinTree]
            The selected join order for the query.
        operator_assignment : Optional[PhysicalOperatorAssignment]
            The selected operators for the query

        Returns
        -------
        PlanParameterization
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
    def describe(self) -> jsondict:
        """Provides a JSON-serializable representation of the specific strategy, as well as important parameters.

        Returns
        -------
        jsondict
            The description

        See Also
        --------
        postbound.postbound.OptimizationPipeline.describe
        """
        raise NotImplementedError

    def pre_check(self) -> OptimizationPreCheck:
        """Provides requirements that input query or database system have to satisfy for the optimizer to work properly.

        Returns
        -------
        OptimizationPreCheck
            The check instance. Can be an empty check if no specific requirements exist.
        """
        return EmptyPreCheck()

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
    def optimize_query(self, query: SqlQuery, current_plan: QueryPlan) -> QueryPlan:
        """Determines the next query plan.

        If no further optimization steps are configured in the pipeline, this is also the final query plan.

        Parameters
        ----------
        query : SqlQuery
            The query to optimize
        current_plan : QueryPlan
            The execution plan that has so far been built by predecessor strategies. If this step is the first step in the
            optimization pipeline, this might also be a plan from the target database system

        Returns
        -------
        QueryPlan
            The optimized plan
        """
        raise NotImplementedError

    @abc.abstractmethod
    def describe(self) -> jsondict:
        """Provides a JSON-serializable representation of the specific strategy, as well as important parameters.

        Returns
        -------
        jsondict
            The description

        See Also
        --------
        postbound.postbound.OptimizationPipeline.describe
        """
        raise NotImplementedError

    def pre_check(self) -> OptimizationPreCheck:
        """Provides requirements that input query or database system have to satisfy for the optimizer to work properly.

        Returns
        -------
        OptimizationPreCheck
            The check instance. Can be an empty check if no specific requirements exist.
        """
        return EmptyPreCheck()

    def __repr__(self) -> str:
        return str(self)

    def __str__(self) -> str:
        return type(self).__name__


class _CompleteAlgorithmEmulator(CompleteOptimizationAlgorithm):
    """Utility to use implementations of staged optimization strategies when a complete algorithm is expected.

    The emulation is enabled by supplying ``None`` values at all places where the stage expects input from previous stages.
    The output of the actual stage is used to obtain a query plan which in turn is used to generate the required optimizer
    information.

    Parameters
    ----------
    database : Optional[Database], optional
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

    def __init__(
        self,
        database: Optional[Database] = None,
        *,
        join_order_optimizer: Optional[JoinOrderOptimization] = None,
        operator_selection: Optional[PhysicalOperatorSelection] = None,
        plan_parameterization: Optional[ParameterGeneration] = None,
    ) -> None:
        super().__init__()
        self.database = (
            database
            if database is not None
            else DatabasePool.get_instance().current_database()
        )
        if all(
            stage is None
            for stage in (
                join_order_optimizer,
                operator_selection,
                plan_parameterization,
            )
        ):
            raise ValueError("Exactly one stage has to be given")
        self._join_order_optimizer = join_order_optimizer
        self._operator_selection = operator_selection
        self._plan_parameterization = plan_parameterization

    def stage(
        self,
    ) -> JoinOrderOptimization | PhysicalOperatorSelection | ParameterGeneration:
        """Provides the actually specified stage.

        Returns
        -------
        JoinOrderOptimization | PhysicalOperatorSelection | ParameterGeneration
            The optimization stage.
        """
        return (
            self._join_order_optimizer
            if self._join_order_optimizer is not None
            else (
                self._operator_selection
                if self._operator_selection is not None
                else self._plan_parameterization
            )
        )

    def optimize_query(self, query: SqlQuery) -> QueryPlan:
        join_order = (
            self._join_order_optimizer.optimize_join_order(query)
            if self._join_order_optimizer is not None
            else None
        )
        physical_operators = (
            self._operator_selection.select_physical_operators(query, None)
            if self._operator_selection is not None
            else None
        )
        plan_params = (
            self._plan_parameterization.generate_plan_parameters(query, None, None)
            if self._plan_parameterization is not None
            else None
        )
        hinted_query = self.database.hinting().generate_hints(
            query,
            join_order=join_order,
            physical_operators=physical_operators,
            plan_parameters=plan_params,
        )
        return self.database.optimizer().query_plan(hinted_query)

    def describe(self) -> jsondict:
        return self.stage().describe()

    def pre_check(self) -> OptimizationPreCheck:
        return self.stage().pre_check()


def as_complete_algorithm(
    stage: JoinOrderOptimization | PhysicalOperatorSelection | ParameterGeneration,
    *,
    database: Optional[Database] = None,
) -> CompleteOptimizationAlgorithm:
    """Enables using a partial optimization stage in situations where a complete optimizer is expected.

    This emulation is achieved by using the partial stage to obtain a partial query plan. The target database system is then
    tasked with filling the gaps to construct a complete execution plan.

    Basically this method is syntactic sugar in situations where a `MultiStageOptimizationPipeline` would be filled with only a
    single stage. Using `as_complete_algorithm`, the construction of an entire pipeline can be omitted. Furthermore it can seem
    more natural to "convert" the stage into a complete algorithm in this case.

    Parameters
    ----------
    stage : JoinOrderOptimization | PhysicalOperatorSelection | ParameterGeneration
        The stage that should become a complete optimization algorithm
    database : Optional[Database], optional
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
    return _CompleteAlgorithmEmulator(
        database,
        join_order_optimizer=join_order_optimizer,
        operator_selection=operator_selection,
        plan_parameterization=parameter_generation,
    )


@dataclass
class PreCheckResult:
    """Wrapper for a validation result.

    The result is used in two different ways: to model the check for supported database systems for optimization strategies and
    to model the check for supported queries for optimization strategies.

    The `ensure_all_passed` method can be used to quickly assert that no problems occurred.

    Attributes
    ----------
    passed : bool
        Indicates whether problems were detected
    failure_reason : str | list[str], optional
        Gives details about the problem(s) that were detected
    """

    passed: bool = True
    failure_reason: str | list[str] = ""

    @staticmethod
    def with_all_passed() -> PreCheckResult:
        """Generates a check result without any problems.

        Returns
        -------
        PreCheckResult
            The check result
        """
        return PreCheckResult()

    @staticmethod
    def merge(checks: Iterable[PreCheckResult]) -> PreCheckResult:
        """Merges multiple check results into a single result.

        The result is passed if all input checks are passed. If any of the checks failed, the failure reasons are merged into
        a single list.

        Parameters
        ----------
        checks : Iterable[PreCheckResult]
            The check results to merge

        Returns
        -------
        PreCheckResult
            The merged check result
        """
        failures: list[str] = []
        for check in checks:
            if check.passed:
                continue
            failures.extend(util.enlist(check.failure_reason))
        return (
            PreCheckResult.with_all_passed()
            if not failures
            else PreCheckResult.with_failure(failures)
        )

    def with_failure(failure: str | list[str]) -> PreCheckResult:
        """Generates a check result for a specific failure.

        Parameters
        ----------
        failure : str | list[str]
            The failure message(s)

        Returns
        -------
        PreCheckResult
            The check result
        """
        return PreCheckResult(False, failure)

    def ensure_all_passed(self, context: SqlQuery | Database | None = None) -> None:
        """Raises an error if the check contains any failures.

        Depending on the context, a more specific error can be raised. The context is used to infer whether an optimization
        strategy does not work on a database system, or whether an input query is not supported by an optimization strategy.

        Parameters
        ----------
        context : SqlQuery | Database | None, optional
            An indicator of the kind of check that was performed. This influences the kind of error that will be raised in case
            of failure. Defaults to ``None`` if no further context is available.

        Raises
        ------
        util.StateError
            In case of failure if there is no additional context available
        UnsupportedQueryError
            In case of failure if the context is an SQL query
        UnsupportedSystemError
            In case of failure if the context is a database interface
        """
        if self.passed:
            return
        if context is None:
            raise util.StateError(f"Pre check failed {self._generate_failure_str()}")
        elif isinstance(context, SqlQuery):
            raise UnsupportedQueryError(context, self.failure_reason)
        elif isinstance(context, Database):
            raise UnsupportedSystemError(context, self.failure_reason)

    def _generate_failure_str(self) -> str:
        """Creates a nice string of the failure messages from `failure_reason`s.

        Returns
        -------
        str
            The failure message
        """
        if not self.failure_reason:
            return ""
        elif isinstance(self.failure_reason, str):
            inner_contents = self.failure_reason
        elif isinstance(self.failure_reason, Iterable):
            inner_contents = " | ".join(reason for reason in self.failure_reason)
        else:
            raise ValueError(
                "Unexpected failure reason type: " + str(self.failure_reason)
            )
        return f"[{inner_contents}]"


class UnsupportedQueryError(RuntimeError):
    """Error to indicate that a specific query cannot be optimized by a selected algorithms.

    Parameters
    ----------
    query : SqlQuery
        The unsupported query
    features : str | list[str], optional
        The features of the query that are unsupported. Defaults to an empty string
    """

    def __init__(self, query: SqlQuery, features: str | list[str] = "") -> None:
        if isinstance(features, list):
            features = ", ".join(features)
        features_str = f" [{features}]" if features else ""

        super().__init__(f"Query contains unsupported features{features_str}: {query}")
        self.query = query
        self.features = features


class UnsupportedSystemError(RuntimeError):
    """Error to indicate that a selected query plan cannot be enforced on a target system.

    Parameters
    ----------
    db_instance : Database
        The database system without a required feature
    reason : str, optional
        The features that are not supported. Defaults to an empty string
    """

    def __init__(self, db_instance: Database, reason: str = "") -> None:
        error_msg = f"Unsupported database system: {db_instance}"
        if reason:
            error_msg += f" ({reason})"
        super().__init__(error_msg)
        self.db_system = db_instance
        self.reason = reason
