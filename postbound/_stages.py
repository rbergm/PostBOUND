from __future__ import annotations

import abc
import math
from collections.abc import Generator, Iterable
from typing import Optional

from . import db, util
from ._core import Cardinality, Cost
from ._qep import QueryPlan
from .optimizer import validation
from .optimizer._hints import PhysicalOperatorAssignment, PlanParameterization
from .optimizer._jointree import JoinTree
from .optimizer.validation import OptimizationPreCheck
from .qal import SqlQuery, TableReference
from .util import jsondict


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
        return validation.EmptyPreCheck()

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

    def initialize(self, target_db: db.Database, query: SqlQuery) -> None:
        """Hook method that is called before the actual optimization process starts.

        This method can be overwritten to set up any necessary data structures, etc. and will be called before each query.

        Parameters
        ----------
        target_db : db.Database
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
        return validation.EmptyPreCheck()

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

    def initialize(self, target_db: db.Database, query: SqlQuery) -> None:
        """Hook method that is called before the actual optimization process starts.

        This method can be overwritten to set up any necessary data structures, etc. and will be called before each query.

        Parameters
        ----------
        target_db : db.Database
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
        return validation.EmptyPreCheck()

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
        return validation.EmptyPreCheck()

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
        return validation.EmptyPreCheck()

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
        return validation.EmptyPreCheck()

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
        return validation.EmptyPreCheck()

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

    def __init__(
        self,
        database: Optional[db.Database] = None,
        *,
        join_order_optimizer: Optional[JoinOrderOptimization] = None,
        operator_selection: Optional[PhysicalOperatorSelection] = None,
        plan_parameterization: Optional[ParameterGeneration] = None,
    ) -> None:
        super().__init__()
        self.database = (
            database
            if database is not None
            else db.DatabasePool.get_instance().current_database()
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
    database: Optional[db.Database] = None,
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
    return _CompleteAlgorithmEmulator(
        database,
        join_order_optimizer=join_order_optimizer,
        operator_selection=operator_selection,
        plan_parameterization=parameter_generation,
    )


class CardinalityGenerator(ParameterGeneration, CardinalityEstimator, abc.ABC):
    """End-to-end cardinality estimator.

    Implementations of this service calculate cardinalities for all relevant intermediate results of a query. In turn, these
    cardinalities can be used by the optimizer of an actual database system to overwrite the native estimates.

    The default implementations of all methods either request cardinality estimates for all possible intermediate results (in
    the `estimate_cardinalities` method), or for exactly those intermediates that are defined in a specific join order (in the
    `generate_plan_parameters` method that implements the protocol of the `ParameterGeneration` class). Therefore, developers
    working on their own cardinality estimation algorithm only need to implement the `calculate_estimate` method. All related
    processes are provided by the generator with reasonable default strategies.

    However, special care is required when considering cross products: depending on the setting intermediates can either allow
    cross products at all stages (by passing ``allow_cross_products=True`` during instantiation), or to disallow them entirely.
    Therefore, the `calculate_estimate` method should act accordingly. Implementations of this class should pass the
    appropriate parameter value to the super *__init__* method. If they support both scenarios, the parameter can also be
    exposed to the client.

    Notice that this strategies fails for queries which contain actual cross products. That is why the `pre_check` only
    accepts queries without cross products. Developers should overwrite the relevant methods as needed. See *Warnings* for more
    details.

    Parameters
    ----------
    allow_cross_products : bool
        Whether the default intermediate generation is allowed to emit cross products between arbitrary tables in the input
        query.

    Warnings
    --------
    The default implementation of this service does not work well for queries that naturally contain cross products. If you
    intend to use if for workloads that contain cross products, you should overwrite the `generate_intermediates` method to
    produce exactly those (partial) joins that you want to allow.
    """

    def __init__(self, allow_cross_products: bool) -> None:
        super().__init__()
        self.allow_cross_products = allow_cross_products

    @abc.abstractmethod
    def calculate_estimate(
        self, query: SqlQuery, tables: TableReference | Iterable[TableReference]
    ) -> Cardinality:
        """Determines the cardinality estimate for a specific intermediate result.

        Ideally this is the only functionality-related method that needs to be implemented by developers using the cardinality
        generator.

        Parameters
        ----------
        query : qal.SqlQuery
            The query to optimize
        tables : TableReference | Iterable[TableReference]
            The intermediate which should be estimated. The intermediate is described by its tables. It should be assumed that
            all filters and join predicates have been pushed down as far as possible.

        Returns
        -------
        Cardinality
            The estimated cardinality if it could be computed, *NaN* otherwise.
        """
        raise NotImplementedError

    def generate_intermediates(
        self, query: SqlQuery
    ) -> Generator[frozenset[TableReference], None, None]:
        """Provides all intermediate results of a query.

        The inclusion of cross-products between arbitrary tables can be configured via the `allow_cross_products` attribute.

        Parameters
        ----------
        query : qal.SqlQuery
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
        query : qal.SqlQuery
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
                parameterization.add_cardinality_hint(join, estimate)
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
                parameterization.add_cardinality_hint(intermediate.tables(), estimate)

        return parameterization

    def pre_check(self) -> validation.OptimizationPreCheck:
        return validation.CrossProductPreCheck()
