from __future__ import annotations

import abc
import math
from collections.abc import Generator, Iterable, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Literal, Optional, Protocol, get_args, runtime_checkable

import pandas as pd

from . import util
from ._core import Cardinality, Cost, TableReference, TimeMs
from ._hints import JoinTree, PhysicalOperatorAssignment, PlanParameterization
from ._qep import QueryPlan
from .db import Database, DatabasePool, ResultSet
from .qal import SqlQuery
from .util.jsonize import jsondict
from .validation import CrossProductPreCheck, EmptyPreCheck, OptimizationPreCheck
from .workloads import Workload


class CompleteOptimizationAlgorithm(abc.ABC):
    """Constructs an entire query plan for an input query in one integrated optimization process.

    This stage closely models the behaviour of traditional optimization algorithms, e.g. based on dynamic programming.
    Implement the `optimize_query` method to provide the actual optimization logic.
    The `describe` and `pre_check` methods should be overridden to provide metadata about the specific algorithm for
    benchmarking and to ensure that the input query and database system are compatible with the algorithm.
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

    def describe(self) -> jsondict:
        """Provides a JSON-serializable representation of the specific strategy, as well as important parameters.

        Returns
        -------
        jsondict
            The description

        See Also
        --------
        OptimizationPipeline.describe
        """
        return {"name": type(self).__name__}

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
    Implement the `optimize_join_order` method to provide the actual optimization logic.
    The `describe` and `pre_check` methods should be overridden to provide metadata about the specific algorithm for
    benchmarking and to ensure that the input query and database system are compatible with the algorithm.

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

    def describe(self) -> jsondict:
        """Provides a JSON-serializable representation of the specific strategy, as well as important parameters.

        Returns
        -------
        jsondict
            The description

        See Also
        --------
        OptimizationPipeline.describe
        """
        return {"name": type(self).__name__}

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

    This is the second stage in the two-phase optimization process, and takes place after the join order has been
    determined.
    Implement the `select_physical_operators` method to provide the actual optimization logic.
    The `describe` and `pre_check` methods should be overridden to provide metadata about the specific algorithm for
    benchmarking and to ensure that the input query and database system are compatible with the algorithm.


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

    def describe(self) -> jsondict:
        """Provides a JSON-serializable representation of the specific strategy, as well as important parameters.

        Returns
        -------
        jsondict
            The description

        See Also
        --------
        OptimizationPipeline.describe
        """
        return {"name": type(self).__name__}

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
    Implement the `generate_plan_parameters` method to provide the actual optimization logic.
    The `describe` and `pre_check` methods should be overridden to provide metadata about the specific algorithm for
    benchmarking and to ensure that the input query and database system are compatible with the algorithm.

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

    def describe(self) -> jsondict:
        """Provides a JSON-serializable representation of the specific strategy, as well as important parameters.

        Returns
        -------
        jsondict
            The description

        See Also
        --------
        OptimizationPipeline.describe
        """
        return {"name": type(self).__name__}

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


class CardinalityEstimator(ParameterGeneration, abc.ABC):
    """The cardinality estimator calculates how many tuples specific operators will produce.

    Implement the `calculate_estimate` method to provide the actual optimization logic.
    The `describe` and `pre_check` methods should be overridden to provide metadata about the specific algorithm for
    benchmarking and to ensure that the input query and database system are compatible with the algorithm.
    In addition, use `initialize` and `cleanup` methods to implement any necessary setup and teardown logic for the
    current query.

    See Also
    --------
    TextBookOptimizationPipeline
    ParameterGeneration

    Notes
    -----
    If you only care about cardinality estimation, you should generally use this class in a
    `MultiStageOptimizationPipeline` instead of a `TextBookOptimizationPipeline`. This is because a multi-stage pipeline
    has a simple control flow from one stage to the next. This allows us to just generate cardinality estimates for all
    possible intermediates if no join order is given, or just for the intermediates defined in a specific join order
    otherwise. In contrast, the textbook pipeline is controlled by the plan enumerator which decides which plans to
    construct and by extension for which intermediates cardinality estimates are required. However, the framework
    implementation does not provide any way for the actual query optimizer of a database system to hook back into the
    framework to request such data. Therefore, we rely on emulating the behaviour of the actual plan enumerator of the
    target database system (unless a enumerator is explicitly provided). While our approximation for Postgres works quite
    well, it is not entirely accurate and other backends are much less supported.

    The default implementation of all methods related to the `ParameterGeneration` either request cardinality estimates for
    all possible intermediate results (in the `estimate_cardinalities` method), or for exactly those intermediates that are
    defined in a specific join order (in the `generate_plan_parameters` method that implements the protocol of the
    `ParameterGeneration` class). Therefore, developers working on their own cardinality estimation algorithm only need to
    implement the `calculate_estimate` method. All related processes are provided by the generator with reasonable default
    strategies.

    However, special care is required when considering cross products: depending on the setting intermediates can either
    allow cross products at all stages (by passing ``allow_cross_products=True`` during instantiation), or to disallow them
    entirely. Therefore, the `calculate_estimate` method should act accordingly. Implementations of this class should pass
    the appropriate parameter value to the super *__init__* method. If they support both scenarios, the parameter can also
    be exposed to the client.
    """

    def __init__(self, *, allow_cross_products: bool = False) -> None:
        self.allow_cross_products = allow_cross_products
        self.target_db: Database = None  # type: ignore[assignment]
        self.query: SqlQuery = None  # type: ignore[assignment]

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
            The estimated cardinality of the specific intermediate
        """
        raise NotImplementedError

    def describe(self) -> jsondict:
        """Provides a JSON-serializable representation of the specific estimator, as well as important parameters.

        Returns
        -------
        jsondict
            The description

        See Also
        --------
        OptimizationPipeline.describe
        """
        return {"name": type(self).__name__}

    def initialize(self, target_db: Database, query: SqlQuery) -> None:
        """Hook method that is called before the actual optimization process starts.

        This method can be overwritten to set up any necessary data structures, etc. and will be called before each query.
        The default implementation stores the target database and query as attributes for later use.

        Parameters
        ----------
        target_db : Database
            The database for which the optimized queries should be generated.
        query : SqlQuery
            The query to be optimized
        """
        self.target_db = target_db
        self.query = query

    def cleanup(self) -> None:
        """Hook method that is called after the optimization process has finished.

        This method can be overwritten to remove any temporary state that was specific to the last query being optimized
        and should not be shared with later queries.

        The default implementation removes the references to the target database and query.
        """
        self.target_db = None  # type: ignore[assignment]
        self.query = None  # type: ignore[assignment]

    def generate_intermediates(
        self, query: SqlQuery
    ) -> Generator[frozenset[TableReference], None, None]:
        """Provides all intermediate results of a query.

        The inclusion of cross-products between arbitrary tables can be configured via the `allow_cross_products` attribute.

        Parameters
        ----------
        query : SqlQuery
            The query for which to generate the intermediates

        Yields
        ------
        Generator[frozenset[TableReference], None, None]
            The intermediates

        Warnings
        --------
        The default implementation of this method does not work for queries that naturally contain cross products. If such a
        query is passed, no intermediates with tables from different partitions of the join graph are yielded.
        """
        for candidate_join in util.powerset(query.tables()):
            if (
                not candidate_join
            ):  # skip empty set (which is an artefact of the powerset method)
                continue
            if not self.allow_cross_products and not query.predicates().joins_tables(
                candidate_join
            ):
                continue
            yield frozenset(candidate_join)

    def estimate_cardinalities(self, query: SqlQuery) -> PlanParameterization:
        """Produces all cardinality estimates for a specific query.

        The default implementation of this method delegates the actual estimation to the `calculate_estimate` method. It is
        called for each intermediate produced by `generate_intermediates`.

        Parameters
        ----------
        query : SqlQuery
            The query to optimize

        Returns
        ------
        PlanParameterization
            A parameterization containing cardinality hints for all intermediates. Other attributes of the parameterization are
            not modified.
        """
        parameterization = PlanParameterization()
        for join in self.generate_intermediates(query):
            estimate = self.calculate_estimate(query, join)
            if not math.isnan(estimate):
                parameterization.add_cardinality(join, estimate)
        return parameterization

    def generate_plan_parameters(
        self,
        query: SqlQuery,
        join_order: Optional[JoinTree],
        operator_assignment: Optional[PhysicalOperatorAssignment],
    ) -> PlanParameterization:
        if join_order is None:
            return self.estimate_cardinalities(query)

        parameterization = PlanParameterization()
        for intermediate in join_order.iternodes():
            estimate = self.calculate_estimate(query, intermediate.tables())
            if not math.isnan(estimate):
                parameterization.add_cardinality(intermediate.tables(), estimate)

        return parameterization

    def pre_check(self) -> OptimizationPreCheck:
        """Provides requirements that input query or database system have to satisfy for the optimizer to work properly.

        Returns
        -------
        OptimizationPreCheck
            The check instance. Can be an empty check if no specific requirements exist.
        """
        if self.allow_cross_products:
            return CrossProductPreCheck()
        return EmptyPreCheck()

    def __repr__(self) -> str:
        return str(self)

    def __str__(self) -> str:
        return type(self).__name__


class CostModel(abc.ABC):
    """The cost model estimates how expensive computing a certain query plan is.

    Implement the `estimate_cost` method to provide the actual optimization logic.
    The `describe` and `pre_check` methods should be overridden to provide metadata about the specific algorithm for
    benchmarking and to ensure that the input query and database system are compatible with the algorithm.
    In addition, use `initialize` and `cleanup` methods to implement any necessary setup and teardown logic for the
    current query.


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

    def describe(self) -> jsondict:
        """Provides a JSON-serializable representation of the specific cost model, as well as important parameters.

        Returns
        -------
        jsondict
            The description

        See Also
        --------
        OptimizationPipeline.describe
        """
        return {"name": type(self).__name__}

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

    Implement the `generate_execution_plan` method to provide the actual optimization logic.
    The `describe` and `pre_check` methods should be overridden to provide metadata about the specific algorithm for
    benchmarking and to ensure that the input query and database system are compatible with the algorithm.

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

    def describe(self) -> jsondict:
        """Provides a JSON-serializable representation of the specific enumerator, as well as important parameters.

        Returns
        -------
        jsondict
            The description

        See Also
        --------
        OptimizationPipeline.describe
        """
        return {"name": type(self).__name__}

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

    Implement the `optimize_query` method to provide the actual optimization logic.
    The `describe` and `pre_check` methods should be overridden to provide metadata about the specific algorithm for
    benchmarking and to ensure that the input query and database system are compatible with the algorithm.
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

    def describe(self) -> jsondict:
        """Provides a JSON-serializable representation of the specific strategy, as well as important parameters.

        Returns
        -------
        jsondict
            The description

        See Also
        --------
        OptimizationPipeline.describe
        """
        return {"name": type(self).__name__}

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
        if self._join_order_optimizer is not None:
            return self._join_order_optimizer

        if self._operator_selection is not None:
            return self._operator_selection

        assert self._plan_parameterization is not None
        return self._plan_parameterization

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


TrainingMetrics = jsondict


TrainingFeature = Literal["query", "runtime", "query-plan", "cost-estimate"]


class TrainingSpec:
    def __init__(self, features: Iterable[TrainingFeature]) -> None:
        self._features: list[TrainingFeature] = list(features)
        self._feature_set = frozenset(self._features)

    @property
    def features(self) -> Sequence[TrainingFeature]:
        return self._features

    @property
    def feature_set(self) -> frozenset[TrainingFeature]:
        return self._feature_set

    def provides(self, feature: TrainingFeature | Iterable[TrainingFeature]) -> bool:
        feature: set[TrainingFeature] = set(util.enlist(feature))
        return feature.issubset(self._feature_set)

    def requires(self, feature: TrainingFeature | Iterable[TrainingFeature]) -> bool:
        return self.provides(feature)

    def satisfies(self, other: TrainingSpec) -> SpecViolations:
        missing = other._feature_set - self._feature_set
        return SpecViolations(missing)

    def __iter__(self):
        return iter(self.features)


class SpecViolations:
    def __init__(self, missing_features: frozenset[TrainingFeature]) -> None:
        self.missing_features = missing_features

    def __bool__(self) -> bool:
        return not bool(self.missing_features)


def _df_reader(path: Path | str) -> pd.DataFrame:
    pass


@dataclass
class TrainingData:
    @staticmethod
    def from_df(
        df: pd.DataFrame | Path | str, *, source: Optional[Path | str] = None
    ) -> TrainingData:
        if isinstance(df, (str, Path)):
            source = Path(df)
            df = _df_reader(source)
        if isinstance(source, str):
            source = Path(source)

        detected_features: list[TrainingFeature] = []
        available_features = set(get_args(TrainingFeature))
        for col in df.columns:
            if col not in available_features:
                continue
            detected_features.append(col)

        feature_map: dict[TrainingFeature, str] = {
            feat: feat for feat in detected_features
        }
        return TrainingData(df, source=source, feature_map=feature_map)

    def __init__(
        self,
        samples: pd.DataFrame,
        *,
        source: Optional[Path] = None,
        feature_map: dict[TrainingFeature, str],
    ) -> None:
        self.source = source
        self.feature_map = feature_map
        self.samples = samples
        self._spec = TrainingSpec(self.feature_map.keys())

    def provides(self, feature: TrainingFeature) -> bool:
        return self._spec.provides(feature)

    def conform_to(
        self, features: Iterable[TrainingFeature] | TrainingSpec
    ) -> TrainingData:
        spec = (
            features if isinstance(features, TrainingSpec) else TrainingSpec(features)
        )
        if not self._spec.satisfies(spec):
            raise ValueError("Requested spec is not compatible with the training data")
        reduced_spec: dict[TrainingFeature, str] = {
            feature: col
            for feature, col in self.feature_map.items()
            if spec.requires(feature)
        }
        return TrainingData(self.samples, source=self.source, feature_map=reduced_spec)

    def as_df(self, requested_spec: TrainingSpec | None = None) -> pd.DataFrame:
        if requested_spec is None:
            target_cols = self.feature_map.values()
        elif not self._spec.satisfies(requested_spec):
            raise ValueError("Requested spec is not compatible with the training data")
            target_cols = [self.feature_map[feature] for feature in requested_spec]
        return self.samples[target_cols]

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx) -> list:
        sample = self.samples.iloc[idx]
        remapped = sample[self.feature_map.values()]
        return list(remapped)


class TrainingDataRepository:
    def __init__(self) -> None:
        self.specs: list[TrainingSpec] = []


@runtime_checkable
class DataDrivenOptimizer(Protocol):
    def fit_database(self, database: Database) -> TrainingMetrics: ...

    def database_is_setup(self) -> bool: ...


@runtime_checkable
class QueryDrivenOptimizer(Protocol):
    def fit_workload(self, queries: Workload) -> TrainingMetrics: ...

    def fit_training_data(self, samples: TrainingData) -> TrainingMetrics: ...

    def requires(self) -> Optional[TrainingSpec]: ...

    def workload_is_setup(self) -> bool: ...


@runtime_checkable
class OnlineOptimizer(Protocol):
    def learn_from_feedback(
        self, query: SqlQuery, result_set: ResultSet, *, exec_time: TimeMs
    ) -> TrainingMetrics: ...


OptimizationStage = (
    CompleteOptimizationAlgorithm
    | JoinOrderOptimization
    | PhysicalOperatorSelection
    | ParameterGeneration
    | PlanEnumerator
    | CostModel
    | CardinalityEstimator
    | IncrementalOptimizationStep
)
"""Type alias for all currently supported optimization stages."""
