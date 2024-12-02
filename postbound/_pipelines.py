"""Provides PostBOUND's main optimization pipeline.

In fact, PostBOUND does not provide a single pipeline implementation. Rather, different pipeline types exists to accomodate
different use-cases. See the documentation of the general `OptimizationPipeline` base class for details. That class serves as
the smallest common denominator among all pipeline implementations.
"""
from __future__ import annotations

import abc
from typing import Optional, Protocol


from .optimizer import validation
from .optimizer.jointree import LogicalJoinTree, PhysicalQueryPlan
from .optimizer.validation import OptimizationPreCheck
from .optimizer._hints import PhysicalOperatorAssignment, PlanParameterization
from . import db
from .qal import SqlQuery, TableReference
from .util import jsondict, StateError


class OptimizationPipeline(abc.ABC):
    """The optimization pipeline is the main tool to apply different strategies to optimize SQL queries.

    Depending on the specific scenario, different concrete pipeline implementations exist. For example, to apply a two-stage
    optimization design (e.g. consisting of join ordering and a subsequent physical operator selection), the
    `TwoStageOptimizationPipeline` exists. Similarly, for optimization algorithms that perform join ordering and operator
    selection in one process, an `IntegratedOptimizationPipeline` is available. The `TextBookOptimizationPipeline` is modelled
    after the traditional interplay of cardinality estimator, cost model and plan enumerator. Lastly, to model approaches that
    subsequently improve query plans by correcting some previous optimization decisions (e.g. transforming a hash join to a
    nested loop join), the `IncrementalOptimizationPipeline` is provided. Consult the individual pipeline documentation for
    more details. This class only describes the basic interface that is shared by all the pipeline implementations.

    If in doubt what the best pipeline implementation is, it is probably best to start with the `TwoStageOptimizationPipeline`
    or the `TextBookOptimizationPipeline`, since they are the most flexible.
    """

    @abc.abstractmethod
    def query_execution_plan(self, query: SqlQuery) -> PhysicalQueryPlan:
        """Applies the current pipeline configuration to obtain an optimized plan for the input query.

        Parameters
        ----------
        query : SqlQuery
            The query that should be optimized

        Returns
        -------
        PhysicalQueryPlan
            An optimized query execution plan for the input query.

            If the optimization strategies only provide partial optimization decisions (e.g. physical operators for a subset of
            the joins), it is up to the pipeline to fill the gaps in order to provide a complete execution plan. A typical
            approach could be to delegate this task to the optimizer of the target database by providing it the partial
            optimization information.

        Raises
        ------
        validation.UnsupportedQueryError
            If the selected optimization algorithms cannot be applied to the specific query, e.g. because it contains
            unsupported features.
        """
        raise NotImplementedError

    def optimize_query(self, query: SqlQuery) -> SqlQuery:
        """Applies the current pipeline configuration to optimize the input query.

        This process also involges the generation of appropriate optimization information that enforces the selected
        optimization decision when the query is executed on an actual database system.

        Parameters
        ----------
        query : SqlQuery
            The query that should be optimized

        Returns
        -------
        SqlQuery
            A transformed query that encapsulates all the optimization decisions made by the pipeline. What this
            actually means depends on the selected optimization strategies, as well as specifics of the target database
            system:

            Depending on the optimization strategy the optimization decisions can range from simple operator selections
            (such as "no nested loop join for this join") to entire physical query execution plans (consisting of a
            join order, as well as scan and join operators for all parts of the plan) and anything in between. For
            novel cardinality estimation approaches, the optimization info could also be structured such that the
            default cardinality estimates are overwritten.

            Secondly, the way the optimization info is expressed depends on the selected database system. Most systems
            do not allow direct a direct modification of the query optimizer's implementation. Therefore, PostBOUND
            takes an indirect approach: it emits system-specific hints that enable corrections for individual optimizer
            decisions (such as disabling a specific physical operator). For example, PostgreSQL allows to use planner
            options such as ``SET enable_nestloop = 'off'`` to disable nested loop joins for the all subsequent queries
            in the current connection. MySQL provides hints like ``BNL(R S)`` to recommend a block-nested loop join or
            hash join (depending on the MySQL version) to the optimizer for a specific join. These hints are inserted
            into comment blocks in the final SQL query. Likewise, some systems treat certain SQL keywords differently
            or provide their own extensions. This also allows to modify the underlying plans. For example, when SQLite
            encouters a ``CROSS JOIN`` syntax in the ``FROM`` clause, it does not try to optimize the join order and
            uses the order in which the tables are specified in the relation instead.

            Therefore, the resulting query will differ from the original input query in a number of ways. However, the
            produced result sets should still be equivalent. If this is not the case, something went severly wrong
            during query optimization. Take a look at the `db` module for more details on the database system support
            and the query generation capabilities.

        Raises
        ------
        validation.UnsupportedQueryError
            If the selected optimization algorithms cannot be applied to the specific query, e.g. because it contains
            unsupported features.


        References
        ----------

        .. PostgreSQL query planning options: https://www.postgresql.org/docs/15/runtime-config-query.html
        .. MySQL optimizer hints: https://dev.mysql.com/doc/refman/8.0/en/optimizer-hints.html
        .. SQLite ``CROSS JOIN`` handling: https://www.sqlite.org/optoverview.html#crossjoin
        """
        execution_plan = self.query_execution_plan(query)
        hinting_service = self.target_database().hinting()
        return hinting_service.generate_hints(query, execution_plan)

    @abc.abstractmethod
    def target_database(self) -> db.Database:
        """Provides the current target database.

        Returns
        -------
        db.Database
            The database for which the input queries should be optimized
        """
        raise NotImplementedError

    @abc.abstractmethod
    def describe(self) -> dict:
        """Generates a description of the current pipeline configuration.

        This description is intended to transparently document which optimization strategies have been selected and
        how they have been instantiated. It can be JSON-serialized and will be included by most of the output of the
        utilities in the `runner` module of the `experiments` package.

        Returns
        -------
        dict
            The actual description
        """
        raise NotImplementedError


class CompleteOptimizationAlgorithm(abc.ABC):
    """Constructs an entire query plan for an input query in one integrated optimization process.

    This stage closely models the behaviour of traditional optimization algorithms, e.g. based on dynamic programming.
    """

    @abc.abstractmethod
    def optimize_query(self, query: SqlQuery) -> PhysicalQueryPlan:
        """Constructs the optimized execution plan for an input query.

        Parameters
        ----------
        query : SqlQuery
            The query to optimize

        Returns
        -------
        PhysicalQueryPlan
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


class IntegratedOptimizationPipeline(OptimizationPipeline):
    """This pipeline is intended for algorithms that calculate the entire query plan in a single process.

    To configure the pipeline, assign the selected strategy to the `optimization_algorithm` property.

    Parameters
    ----------
    target_db : Optional[db.Database], optional
        The database for which the optimized queries should be generated. If this is not given, he default database is
        extracted from the `DatabasePool`.
    """

    def __init__(self, target_db: Optional[db.Database] = None) -> None:
        self._target_db = (target_db if target_db is not None
                           else db.DatabasePool.get_instance().current_database())
        self._optimization_algorithm: Optional[CompleteOptimizationAlgorithm] = None
        super().__init__()

    @property
    def target_db(self) -> db.Database:
        """The database for which optimized queries should be generated.

        When assigning a new target database, compatibility with the current `optimization_algorithm` will be checked.

        Returns
        -------
        db.Database
            The currently selected database system

        Raises
        ------
        validation.UnsupportedSystemError
            If the current optimization algorithm is not compatible with the new target database system

        See Also
        --------
        CompleteOptimizationAlgorithm.pre_check
        """
        return self._target_db

    @target_db.setter
    def target_db(self, system: db.Database) -> None:
        if self._optimization_algorithm is not None:
            pre_check = self._optimization_algorithm.pre_check()
            pre_check.check_supported_database_system(system).ensure_all_passed()
        self._target_db = system

    @property
    def optimization_algorithm(self) -> Optional[CompleteOptimizationAlgorithm]:
        """The optimization algorithm is used each time a query should be optimized.

        When assigning a new algorithm, it is checked for compatibility with `target_db`.

        Returns
        -------
        Optional[CompleteOptimizationAlgorithm]
            The currently selected optimization algorithm, if any.

        Raises
        ------
        validation.UnsupportedSystemError
            If the new optimization algorithm is not compatible with the current target database system.

        See Also
        --------
        CompleteOptimizationAlgorithm.pre_check

        """
        return self._optimization_algorithm

    @optimization_algorithm.setter
    def optimization_algorithm(self, algorithm: CompleteOptimizationAlgorithm) -> None:
        pre_check = algorithm.pre_check()
        if pre_check:
            pre_check.check_supported_database_system(self._target_db).ensure_all_passed()
        self._optimization_algorithm = algorithm

    def query_execution_plan(self, query: SqlQuery) -> PhysicalQueryPlan:
        if self.optimization_algorithm is None:
            raise StateError("No algorithm has been selected")

        pre_check = self.optimization_algorithm.pre_check()
        if pre_check is not None:
            pre_check.check_supported_query(query).ensure_all_passed()

        physical_qep = self.optimization_algorithm.optimize_query(query)
        return physical_qep

    def target_database(self) -> db.Database:
        return self._target_db

    def describe(self) -> dict:
        algorithm_description = (self._optimization_algorithm.describe() if self._optimization_algorithm is not None
                                 else "no_algorithm")
        return {"name": "integrated_pipeline",
                "database_system": self._target_db.describe(),
                "optimization_algorithm": algorithm_description}


Cost = float
"""Type alias for a cost estimate."""

Cardinality = int
"""Type alias for a cardinality estimate."""


class CardinalityEstimator(abc.ABC):
    """The cardinality estimator calculates how many tuples specific operators will produce."""

    @abc.abstractmethod
    def calculate_estimate(self, query: SqlQuery, intermediate: frozenset[TableReference]) -> Cardinality:
        """Determines the cardinality of a specific intermediate.

        Parameters
        ----------
        query : SqlQuery
            The query being optimized
        intermediate : frozenset[TableReference]
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

    def pre_check(self) -> OptimizationPreCheck:
        """Provides requirements that input query or database system have to satisfy for the optimizer to work properly.

        Returns
        -------
        OptimizationPreCheck
            The check instance. Can be an empty check if no specific requirements exist.
        """
        return validation.EmptyPreCheck()


class CostModel(abc.ABC):
    """The cost model estimates how expensive computing a certain query plan is."""

    @abc.abstractmethod
    def estimate_cost(self, query: SqlQuery, plan: PhysicalQueryPlan) -> Cost:
        """Computes the cost estimate for a specific plan.

        The following conventions are used for the estimation: the root node of the plan will not have any cost set. However,
        all input nodes will have already been estimated by earlier calls to the cost model. Hence, while estimating the cost
        of the root node, all earlier costs will be available as inputs.

        It is not the responsibility of the cost model to set the estimate on the plan, this is the task of the enumerator
        (which can decide whether the plan should be considered any further).

        Parameters
        ----------
        query : SqlQuery
            The query being optimized
        plan : PhysicalQueryPlan
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

    def pre_check(self) -> OptimizationPreCheck:
        """Provides requirements that input query or database system have to satisfy for the optimizer to work properly.

        Returns
        -------
        OptimizationPreCheck
            The check instance. Can be an empty check if no specific requirements exist.
        """
        return validation.EmptyPreCheck()


class PlanEnumerator(abc.ABC):
    """The plan enumerator traverses the space of different candidate plans and ultimately selects the optimal one."""

    @abc.abstractmethod
    def generate_execution_plan(self, query: SqlQuery, *, cost_model: CostModel,
                                cardinality_estimator: CardinalityEstimator) -> PhysicalQueryPlan:
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
        PhysicalQueryPlan
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


class TextBookOptimizationPipeline(OptimizationPipeline):
    """This pipeline is modelled after the traditional approach to query optimization as used in most real-world systems.

    The optimizer consists of a cardinality estimator that calculates the size of intermediate results, a cost model that
    quantifies how expensive specific access paths for the intermediates are, and an enumerator that generates the
    intermediates in the first place.

    To configure the pipeline, specific strategies for each of the three components have to be assigned. Notice that it is
    currently not possible to leave any component empty.

    Parameters
    ----------
    target_db : db.Database
        The database for which the optimized queries should be generated.
    """

    def __init__(self, target_db: db.Database) -> None:
        self._target_db = target_db
        self._card_est: Optional[CardinalityEstimator] = None
        self._cost_model: Optional[CostModel] = None
        self._plan_enumerator: Optional[PlanEnumerator] = None
        self._support_check = validation.EmptyPreCheck()
        self._build = False

    def setup_cardinality_estimator(self, estimator: CardinalityEstimator) -> TextBookOptimizationPipeline:
        """Configures the cardinality estimator of the optimizer.

        Setting a new algorithm requires the pipeline to be build again.

        Parameters
        ----------
        estimator : CardinalityEstimator
            The estimator to be used

        Returns
        -------
        self
            The current pipeline to allow for easy method-chaining.
        """
        self._build = False
        self._card_est = estimator
        return self

    def setup_cost_model(self, cost_model: CostModel) -> TextBookOptimizationPipeline:
        """Configures the cost model of the optimizer.

        Setting a new algorithm requires the pipeline to be build again.

        Parameters
        ----------
        cost_model : CostModel
            The cost model to be used

        Returns
        -------
        self
            The current pipeline to allow for easy method-chaining.
        """
        self._build = False
        self._cost_model = cost_model
        return self

    def setup_plan_enumerator(self, plan_enumerator: PlanEnumerator) -> TextBookOptimizationPipeline:
        """Configures the plan enumerator of the optimizer.

        Setting a new algorithm requires the pipeline to be build again.

        Parameters
        ----------
        plan_enumerator : PlanEnumerator
            The enumerator to be used

        Returns
        -------
        self
            The current pipeline to allow for easy method-chaining.
        """
        self._build = False
        self._plan_enumerator = plan_enumerator
        return self

    def build(self) -> TextBookOptimizationPipeline:
        """Constructs the optimization pipeline.

        This includes checking all strategies for compatibility with the `target_db`. Afterwards, the pipeline is ready to
        optimize queries.

        Returns
        -------
        self
            The current pipeline to allow for easy method-chaining.

        Raises
        ------
        validation.UnsupportedSystemError
            If any of the selected optimization stages is not compatible with the `target_db`.
        """
        if self._card_est is None:
            raise StateError("Missing cardinality estimator")
        if self._cost_model is None:
            raise StateError("Missing cost model")
        if self._plan_enumerator is None:
            raise StateError("Missing plan enumerator")

        self._support_check = validation.merge_checks([self._card_est.pre_check(),
                                                       self._cost_model.pre_check(),
                                                       self._plan_enumerator.pre_check()])
        self._support_check.check_supported_database_system(self._target_db).ensure_all_passed(self._target_db)

        self._build = True
        return self

    def query_execution_plan(self, query: SqlQuery) -> PhysicalQueryPlan:
        if not self._build:
            raise StateError("Pipeline has not been build")
        self._support_check.check_supported_query(query).ensure_all_passed(query)

        return self._plan_enumerator.generate_execution_plan(query, cardinality_estimator=self._card_est,
                                                             cost_model=self._cost_model)

    def describe(self) -> dict:
        return {
            "name": "textbook_pipeline",
            "database_system": self._target_db.describe(),
            "plan_enumerator": self._plan_enumerator.describe() if self._plan_enumerator is not None else None,
            "cost_model": self._cost_model.describe() if self._cost_model is not None else None,
            "cardinality_estimator": self._card_est.describe() if self._card_est is not None else None
        }


class JoinOrderOptimization(abc.ABC):
    """The join order optimization generates a complete join order for an input query.

    This is the first step in a two-stage optimizer design.
    """

    @abc.abstractmethod
    def optimize_join_order(self, query: SqlQuery) -> Optional[LogicalJoinTree]:
        """Performs the actual join ordering process.

        The join tree can be further annotated with an initial operator assignment, if that is an inherent part of
        the specific optimization strategy. However, this is generally discouraged and the two-stage pipeline will discard such
        operators to prepare for the subsequent physical operator selection.

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
        super().__init__(f"Join order optimization failed for query {query}" if not message else message)
        self.query = query


class PhysicalOperatorSelection(abc.ABC):
    """The physical operator selection assigns scan and join operators to the tables of the input query.

    This is the second stage in the two-phase optimization process, and takes place after the join order has been determined.
    """

    @abc.abstractmethod
    def select_physical_operators(self, query: SqlQuery, join_order: Optional[LogicalJoinTree]) -> PhysicalOperatorAssignment:
        """Performs the operator assignment.

        Parameters
        ----------
        query : SqlQuery
            The query to optimize
        join_order : Optional[LogicalJoinTree]
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
    specific implementation. Therefore, this is an optional final step in a two-stage optimization process.
    """

    @abc.abstractmethod
    def generate_plan_parameters(self, query: SqlQuery, join_order: Optional[LogicalJoinTree],
                                 operator_assignment: Optional[PhysicalOperatorAssignment]) -> PlanParameterization:
        """Executes the actual parameterization.

        Parameters
        ----------
        query : SqlQuery
            The query to optimize
        join_order : Optional[LogicalJoinTree]
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


class TwoStageOptimizationPipeline(OptimizationPipeline):
    """This optimization pipeline is the main tool to apply and combine different optimization strategies.

    The pipeline is organized in two large stages (join ordering and physical operator selection), which are
    accompanied by initial pre check and a final plan parameterization steps. In total, those four individual steps
    completely specify the optimization settings that should be applied to an incoming query. For each of the steps
    general interface exist that must be implemented by the selected strategies.

    The steps are applied in consecutive order and perform the following tasks:

    1. the incoming query is checked for unsupported features
    2. an optimized join order for the query is calculated
    3. appropriate physical operators are determined, depending on the join order
    4. the query plan (join order + physical operators) is further parameterized

    All steps are optional. If they are not specified, no operation will be performed at the specific stage.

    Once the optimization settings have been selected via the *setup* methods (or alternatively via the `load_settings`
    functionality), the pipeline has to be build using the `build` method. Afterwards, it is ready to optimize
    input queries.

    A pipeline depends on a specific database system. This is necessary to produce the appropriate metadata for an
    input query (i.e. to apply the specifics that enforce the optimized query plan during query execution for the
    database system). This field can be changed between optimization calls to use the same pipeline for different
    systems.

    Parameters
    ----------
    target_db : db.Database
        The database for which the optimized queries should be generated.


    Examples
    --------
    >>> pipeline = pb.TwoStageOptimizationPipline(postgres_db)
    >>> pipeline.load_settings(ues_settings)
    >>> pipeline.build()
    >>> pipeline.optimize_query(join_order_benchmark["1a"])
    """

    def __init__(self, target_db: db.Database) -> None:
        self._target_db = target_db
        self._pre_check: OptimizationPreCheck | None = validation.EmptyPreCheck()
        self._join_order_enumerator: JoinOrderOptimization | None = None
        self._physical_operator_selection: PhysicalOperatorSelection | None = None
        self._plan_parameterization: ParameterGeneration | None = None
        self._build = False

    @property
    def target_db(self) -> db.Database:
        """The database for which optimized queries should be generated.

        When assigning a new target database, the pipeline needs to be build again.

        Returns
        -------
        db.Database
            The currently selected database system
        """
        return self._target_db

    @target_db.setter
    def target_db(self, new_db: db.Database) -> None:
        self._target_db = new_db
        self._build = False

    @property
    def pre_check(self) -> Optional[OptimizationPreCheck]:
        """An overarching check that should be applied to all queries before they are optimized.

        This check complements the pre checks of the individual stages and can be used to enforce experiment-specific
        constraints.

        Returns
        -------
        Optional[OptimizationPreCheck]
            The current check, if any. Can also be an `validation.EmptyPreCheck` instance.
        """
        return self._pre_check

    @property
    def join_order_enumerator(self) -> Optional[JoinOrderOptimization]:
        """The selected join order optimization algorithm.

        Returns
        -------
        Optional[JoinOrderOptimization]
            The current algorithm, if any has been selected.
        """
        return self._join_order_enumerator

    @property
    def physical_operator_selection(self) -> Optional[PhysicalOperatorSelection]:
        """The selected operator selection algorithm.

        Returns
        -------
        Optional[PhysicalOperatorSelection]
            The current algorithm, if any has been selected.
        """
        return self._physical_operator_selection

    @property
    def plan_parameterization(self) -> Optional[ParameterGeneration]:
        """The selected parameterization algorithm.

        Returns
        -------
        Optional[ParameterGeneration]
            The current algorithm, if any has been selected.
        """
        return self._plan_parameterization

    def setup_query_support_check(self, check: OptimizationPreCheck) -> TwoStageOptimizationPipeline:
        """Configures the pre-check that should be executed for each query.

        This check will be combined with any additional checks that are required by the actual optimization strategies.
        Setting a new check requires the pipeline to be build again.

        Parameters
        ----------
        check : OptimizationPreCheck
            The new check

        Returns
        -------
        self
            The current pipeline to allow for easy method-chaining.
        """
        self._pre_check = check
        self._build = False
        return self

    def setup_join_order_optimization(self, enumerator: JoinOrderOptimization) -> TwoStageOptimizationPipeline:
        """Configures the pipeline to obtain an optimized join order.

        The actual strategy can either produce a purely logical join order, or an initial physical query execution plan
        that also specifies how the individual joins should be executed. All later stages are expected to work with
        these two cases.

        Setting a new algorithm requires the pipeline to be build again.

        Parameters
        ----------
        enumerator : JoinOrderOptimization
            The new join order optimization algorithm

        Returns
        -------
        self
            The current pipeline to allow for easy method-chaining.
        """
        self._join_order_enumerator = enumerator
        self._build = False
        return self

    def setup_physical_operator_selection(self, selector: PhysicalOperatorSelection) -> TwoStageOptimizationPipeline:
        """Configures the algorithm to assign physical operators to the query.

        This algorithm receives the input query as well as the join order (if there is one) as input. In a special
        case, this join order can also provide an initial assignment of physical operators. These settings can then
        be further adapted by the selected algorithm (or completely overwritten).

        Setting a new algorithm requires the pipeline to be build again.

        Paramters
        ---------
        selector : PhysicalOperatorSelection
            The new operator selection algorithm

        Returns
        -------
        self
            The current pipeline to allow for easy method-chaining.
        """
        self._physical_operator_selection = selector
        self._build = False
        return self

    def setup_plan_parameterization(self, param_generator: ParameterGeneration) -> TwoStageOptimizationPipeline:
        """Configures the algorithm to parameterize the query plan.

        This algorithm receives the input query as well as the join order and the physical operators (if those have
        been determined yet) as input.

        Setting a new algorithm requires the pipeline to be build again.

        Parameters
        ----------
        param_generator : ParameterGeneration
            The new parameterization algorithm

        Returns
        -------
        self
            The current pipeline to allow for easy method-chaining.
        """
        self._plan_parameterization = param_generator
        self._build = False
        return self

    def load_settings(self, optimization_settings: OptimizationSettings) -> TwoStageOptimizationPipeline:
        """Applies all the optimization settings from a pre-defined optimization strategy to the pipeline.

        This is just a shorthand method to skip calling all setup methods individually for a fixed combination of
        optimization settings. After the settings have been loaded, they can be overwritten again using the *setup*
        methods.

        Loading new presets requires the pipeline to be build again.

        Parameters
        ----------
        optimization_settings : OptimizationSettings
            The specific settings

        Returns
        -------
        self
            The current pipeline to allow for easy method-chaining.
        """
        support_check = optimization_settings.query_pre_check()
        if support_check:
            self.setup_query_support_check(support_check)
        join_ordering = optimization_settings.build_join_order_optimizer()
        if join_ordering:
            self.setup_join_order_optimization(join_ordering)
        operator_selection = optimization_settings.build_physical_operator_selection()
        if operator_selection:
            self.setup_physical_operator_selection(operator_selection)
        plan_parameterization = optimization_settings.build_plan_parameterization()
        if plan_parameterization:
            self.setup_plan_parameterization(plan_parameterization)
        self._build = False
        return self

    def build(self) -> TwoStageOptimizationPipeline:
        """Constructs the optimization pipeline.

        This includes filling all undefined optimization steps with empty strategies and checking all strategies for
        compatibility with the `target_db`. Afterwards, the pipeline is ready to optimize queries.

        Returns
        -------
        self
            The current pipeline to allow for easy method-chaining.

        Raises
        ------
        validation.UnsupportedSystemError
            If any of the selected optimization stages is not compatible with the `target_db`.
        """
        all_checks = [self.pre_check]
        if self.join_order_enumerator is not None:
            all_checks.append(self.join_order_enumerator.pre_check())
        if self.physical_operator_selection is not None:
            all_checks.append(self.physical_operator_selection.pre_check())
        if self.plan_parameterization is not None:
            all_checks.append(self.plan_parameterization.pre_check())

        self._pre_check = validation.merge_checks(all_checks)

        db_check_result = self._pre_check.check_supported_database_system(self._target_db)
        if not db_check_result.passed:
            raise validation.UnsupportedSystemError(self.target_db, db_check_result.failure_reason)

        self._build = True
        return self

    def target_database(self) -> db.Database:
        return self.target_db

    def query_execution_plan(self, query: SqlQuery) -> PhysicalQueryPlan:
        optimized_query = self.optimize_query(query)
        db_plan = self.target_db.optimizer().query_plan(optimized_query)
        physical_qep = PhysicalQueryPlan.load_from_query_plan(db_plan, query=query)
        return physical_qep

    def optimize_query(self, query: SqlQuery) -> SqlQuery:
        self._assert_is_build()
        supported_query_check = self._pre_check.check_supported_query(query)
        if not supported_query_check.passed:
            raise validation.UnsupportedQueryError(query, supported_query_check.failure_reason)

        if self.join_order_enumerator is None:
            join_order = None
        else:
            join_order = self.join_order_enumerator.optimize_join_order(query)

        if self.physical_operator_selection is None:
            physical_operators = PhysicalOperatorAssignment()
        else:
            join_order = join_order.as_logical_join_tree() if isinstance(join_order, PhysicalQueryPlan) else join_order
            physical_operators = self.physical_operator_selection.select_physical_operators(query, join_order)

        if self.plan_parameterization is None:
            plan_parameters = PlanParameterization()
        else:
            join_order = join_order.as_logical_join_tree() if isinstance(join_order, PhysicalQueryPlan) else join_order
            plan_parameters = self._plan_parameterization.generate_plan_parameters(query, join_order, physical_operators)

        return self._target_db.hinting().generate_hints(query, join_order=join_order,
                                                        physical_operators=physical_operators,
                                                        plan_parameters=plan_parameters)

    def describe(self) -> dict:
        return {
            "name": "two_stage_pipeline",
            "database_system": self._target_db.describe(),
            "query_pre_check": self._pre_check.describe() if self._pre_check else None,
            "join_ordering": self._join_order_enumerator.describe() if self._join_order_enumerator else None,
            "operator_selection": (self._physical_operator_selection.describe() if self._physical_operator_selection
                                   else None),
            "plan_parameterization": self._plan_parameterization.describe() if self._plan_parameterization else None
        }

    def _assert_is_build(self) -> None:
        """Raises an error if the pipeline has not been build yet."""
        if not self._build:
            raise StateError("Pipeline has not been build")

    def __repr__(self) -> str:
        return str(self)

    def __str__(self) -> str:
        components = [self._join_order_enumerator, self._physical_operator_selection, self._plan_parameterization]
        opt_chain = " -> ".join(str(comp) for comp in components)
        return f"OptimizationPipeline [{opt_chain}]"


class IncrementalOptimizationStep(abc.ABC):
    """Incremental optimization allows to chain different smaller optimization strategies.

    Each step receives the query plan of its predecessor and can change its decisions in arbitrary ways. For example, this
    scheme can be used to gradually correct mistakes or risky decisions of individual optimizers.
    """

    @abc.abstractmethod
    def optimize_query(self, query: SqlQuery,
                       current_plan: PhysicalQueryPlan) -> PhysicalQueryPlan:
        """Determines the next query plan.

        If no further optimization steps are configured in the pipeline, this is also the final query plan.

        Parameters
        ----------
        query : SqlQuery
            The query to optimize
        current_plan : PhysicalQueryPlan
            The execution plan that has so far been built by predecessor strategies. If this step is the first step in the
            optimization pipeline, this might also be a plan from the target database system

        Returns
        -------
        PhysicalQueryPlan
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

    def optimize_query(self, query: SqlQuery) -> PhysicalQueryPlan:
        join_order = (self._join_order_optimizer.optimize_join_order(query)
                      if self._join_order_optimizer is not None else None)
        physical_operators = (self._operator_selection.select_physical_operators(query, None)
                              if self._operator_selection is not None else None)
        plan_params = (self._plan_parameterization.generate_plan_parameters(query, None, None)
                       if self._plan_parameterization is not None else None)
        hinted_query = self.database.hinting().generate_hints(query, join_order, physical_operators, plan_params)
        query_plan = self.database.optimizer().query_plan(hinted_query)
        return PhysicalQueryPlan(query_plan, query)

    def describe(self) -> jsondict:
        return self.stage().describe()

    def pre_check(self) -> OptimizationPreCheck:
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


class IncrementalOptimizationPipeline(OptimizationPipeline):
    """This optimization pipeline can be thought of as a generalization of the `TwoStageOptimizationPipeline`.

    Instead of only operating in two stages, an arbitrary amount of optimization steps can be applied. During each
    step an entire physical query execution plan is received as input and also produced as output. Therefore, partial
    operator assignments or cardinality estimates are not supported by this pipeline. The incremental nature probably
    makes it the most usefull for optimization strategies that continously improve query plans.

    Parameters
    ----------
    target_db : db.Database
        The database for which the optimized queries should be generated.
    """

    def __init__(self, target_db: db.Database) -> None:
        self._target_db = target_db
        self._initial_plan_generator: Optional[CompleteOptimizationAlgorithm] = None
        self._optimization_steps: list[IncrementalOptimizationStep] = []

    @property
    def target_db(self) -> db.Database:
        """The database for which optimized queries should be generated.

        When a new target database is selected, all optimization steps are checked for support of the new database.

        Returns
        -------
        db.Database
            _description_

        Raises
        ------
        validation.UnsupportedSystemError
            If any of the optimization steps or the initial plan generator cannot work with the target database
        """
        return self._target_db

    @target_db.setter
    def target_db(self, database: db.Database) -> None:
        self._ensure_pipeline_integrity(database=database)
        self._target_db = database

    @property
    def initial_plan_generator(self) -> Optional[CompleteOptimizationAlgorithm]:
        """Strategy to construct the first physical query execution plan to start the incremental optimization.

        If no initial generator is selected, the initial plan will be derived from the optimizer of the target
        database.

        Returns
        -------
        Optional[CompleteOptimizationAlgorithm]
            The current initial generator.

        Raises
        ------
        validation.UnsupportedSystemError
            If the initial generator does not work with the current `target_db`
        """
        return self._initial_plan_generator

    @initial_plan_generator.setter
    def initial_plan_generator(self, plan_generator: Optional[CompleteOptimizationAlgorithm]) -> None:
        self._ensure_pipeline_integrity(initial_plan_generator=plan_generator)
        self._initial_plan_generator = plan_generator

    def add_optimization_step(self, next_step: IncrementalOptimizationStep) -> IncrementalOptimizationPipeline:
        """Expands the optimization pipeline by another stage.

        The given step will be applied at the end of the pipeline. The very first optimization steps receives an
        initial plan that has either been generated via the `initial_plan_generator` (if it has been setup), or by
        retrieving the query execution plan from the `target_db`.

        Parameters
        ----------
        next_step : IncrementalOptimizationStep
            The next optimization stage

        Returns
        -------
        IncrementalOptimizationPipeline
            If any of the optimization steps does not work with the target database
        """
        self._ensure_pipeline_integrity(additional_optimization_step=next_step)
        self._optimization_steps.append(next_step)
        return self

    def target_database(self) -> db.Database:
        return self.target_db

    def query_execution_plan(self, query: SqlQuery) -> PhysicalQueryPlan:
        self._ensure_supported_query(query)
        current_plan = (self.initial_plan_generator.optimize_query(query) if self.initial_plan_generator is not None
                        else self.target_db.optimizer().query_plan(query))
        for optimization_step in self._optimization_steps:
            current_plan = optimization_step.optimize_query(query, current_plan)
        return current_plan

    def describe(self) -> dict:
        return {
            "name": "incremental_pipeline",
            "database_system": self._target_db.describe(),
            "initial_plan": (self._initial_plan_generator.describe() if self._initial_plan_generator is not None
                             else "native"),
            "steps": [step.describe() for step in self._optimization_steps]
        }

    def _ensure_pipeline_integrity(self, *, database: Optional[db.Database] = None,
                                   initial_plan_generator: Optional[CompleteOptimizationAlgorithm] = None,
                                   additional_optimization_step: Optional[IncrementalOptimizationStep] = None,
                                   ) -> None:
        """Checks that all selected optimization strategies work with the target database.

        This method should be called when individual parts of the pipeline have been updated. The updated parts are
        supplied as parameters. All other parameters are inferred from the current pipeline state.

        Parameters
        ----------
        database : Optional[db.Database], optional
            The new target database system if it has been updated, by default None
        initial_plan_generator : Optional[CompleteOptimizationAlgorithm], optional
            The new initial plan generator if it has been updated, by default None
        additional_optimization_step : Optional[IncrementalOptimizationStep], optional
            The next optimization step, if a new one has been added, by default None

        Raises
        ------
        validation.UnsupportedSystemError
            If one of the optimization algorithms is not compatible with the target database
        """
        database = self.target_db if database is None else database
        initial_plan_generator = (self._initial_plan_generator if initial_plan_generator is None
                                  else initial_plan_generator)

        if initial_plan_generator is not None and initial_plan_generator.pre_check() is not None:
            initial_plan_generator.pre_check().check_supported_database_system(database).ensure_all_passed(database)

        if additional_optimization_step is not None and additional_optimization_step.pre_check() is not None:
            (additional_optimization_step
             .pre_check()
             .check_supported_database_system(database)
             .ensure_all_passed(database))

        for incremental_step in self._optimization_steps:
            if incremental_step.pre_check() is None:
                continue
            incremental_step.pre_check().check_supported_database_system(database).ensure_all_passed(database)

    def _ensure_supported_query(self, query: SqlQuery) -> None:
        """Applies all relevant pre-checks to the input query.

        Parameters
        ----------
        query : SqlQuery
            The input query

        Raises
        ------
        validation.UnsupportedQueryError
            If one of the optimization algorithms is not compatible with the input query
        """
        if self._initial_plan_generator is not None and self._initial_plan_generator.pre_check() is not None:
            self._initial_plan_generator.pre_check().check_supported_query(query).ensure_all_passed(query)
        for incremental_step in self._optimization_steps:
            if incremental_step.pre_check() is None:
                continue
            incremental_step.pre_check().check_supported_query(query).ensure_all_passed(query)


class OptimizationSettings(Protocol):
    """Captures related settings for the optimization pipeline to make them more easily accessible.

    All components are optional, depending on the specific optimization scenario/approach.
    """

    def query_pre_check(self) -> Optional[OptimizationPreCheck]:
        """The required query pre-check.

        Returns
        -------
        Optional[OptimizationPreCheck]
            The pre-check if one is necessary, or ``None`` otherwise.
        """
        return None

    def build_complete_optimizer(self) -> Optional[CompleteOptimizationAlgorithm]:
        return None

    def build_join_order_optimizer(self) -> Optional[JoinOrderOptimization]:
        """The algorithm that is used to obtain the optimized join order.

        Returns
        -------
        Optional[JoinOrderOptimization]
            The optimization strategy for the join order, or ``None`` if the scenario does not include a join order
            optimization.
        """
        return None

    def build_physical_operator_selection(self) -> Optional[PhysicalOperatorSelection]:
        """The algorithm that is used to determine the physical operators.

        Returns
        -------
        Optional[PhysicalOperatorSelection]
            The optimization strategy for the physical operators, or ``None`` if the scenario does not include an operator
            optimization.
        """
        return None

    def build_plan_parameterization(self) -> Optional[ParameterGeneration]:
        """The algorithm that is used to further parameterize the query plan.

        Returns
        -------
        Optional[ParameterGeneration]
            The parameter optimization strategy, or ``None`` if the scenario does not include such a stage.
        """
        return None

    def build_incremental_optimizer(self) -> Optional[IncrementalOptimizationStep]:
        return None
