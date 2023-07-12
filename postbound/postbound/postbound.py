"""Provides PostBOUND's main optimization pipeline."""
from __future__ import annotations

import abc
from typing import Optional, Protocol

from postbound.qal import qal, transform
from postbound.optimizer import presets, stages, validation
from postbound.optimizer.strategies import noopt
from postbound.db import db
from postbound.util import errors


class OptimizationPipeline(Protocol):
    """The optimization pipeline is the main tool to apply different strategies to optimize SQL queries.

    Depending on the specific scenario, different concrete pipeline implementations exist. For example, to apply
    a two-stage optimization design (e.g. consisting of join ordering and a subsequent physical operator selection),
    the `TwoStageOptimizationPipeline` exists. Similarly, for optimization algorithms that perform join ordering and
    operator selection in one process (as in the traditional dynamic programming-based approach),
    an `IntegratedOptimizationPipeline` is available. Lastly, to model approaches that subsequently improve query plans
    by correcting some optimization decisions (e.g. transforming a hash join to a nested loop join), the
    `IncrementalOptimizationPipeline` is provided. Consult the individual pipeline documentation for more details. This
    protocol class only describes the basic interface that is shared by all the pipeline implementations.
    """

    @abc.abstractmethod
    def optimize_query(self, query: qal.SqlQuery) -> qal.SqlQuery:
        raise NotImplementedError

    @abc.abstractmethod
    def describe(self) -> dict:
        raise NotImplementedError


class IntegratedOptimizationPipeline(OptimizationPipeline):
    def __init__(self, target_db: Optional[db.Database] = None) -> None:
        self.target_db = (target_db if target_db is not None
                          else db.DatabasePool.get_instance().current_database())
        self._optimization_algorithm: Optional[stages.CompleteOptimizationAlgorithm] = None
        super().__init__()

    @property
    def optimization_algorithm(self) -> Optional[stages.CompleteOptimizationAlgorithm]:
        return self._optimization_algorithm

    @optimization_algorithm.setter
    def optimization_algorithm(self, algorithm: stages.CompleteOptimizationAlgorithm) -> None:
        pre_check = algorithm.pre_check()
        if pre_check:
            pre_check.check_supported_database_system(self.target_db)
        self._optimization_algorithm = algorithm

    def optimize_query(self, query: qal.SqlQuery) -> qal.SqlQuery:
        if self.optimization_algorithm is None:
            raise errors.StateError("No algorithm has been selected")

        pre_check = self.optimization_algorithm.pre_check()
        if pre_check is not None:
            pre_check.check_supported_query(query)

        physical_qep = self.optimization_algorithm.optimize_query(query)

        return self.target_db.hinting().generate_hints(query, physical_qep)

    def describe(self) -> dict:
        algorithm_description = (self._optimization_algorithm.describe() if self._optimization_algorithm is not None
                                 else "no_algorithm")
        return {"target_dbs": self.target_db.describe(), "optimization_algorithm": algorithm_description}


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

    Once the optimization settings have been selected via the _setup_ methods (or alternatively via the `load_settings`
    functionality), the pipeline has to be build using the `build` method. Afterwards, it is ready to optimize
    input queries.

    A pipeline depends on a specific database system. This is necessary to produce the appropriate metadata for an
    input query (i.e. to apply the specifics that enforce the optimized query plan during query execution for the
    database system). This field can be changed between optimization calls to use the same pipeline for different
    systems.
    """

    def __init__(self, target_db: db.Database) -> None:
        self._target_db = target_db
        self._pre_check: validation.OptimizationPreCheck | None = None
        self._join_order_enumerator: stages.JoinOrderOptimization | None = None
        self._physical_operator_selection: stages.PhysicalOperatorSelection | None = None
        self._plan_parameterization: stages.ParameterGeneration | None = None
        self._build = False

    @property
    def target_db(self) -> db.Database:
        return self._target_db

    @target_db.setter
    def target_db(self, new_db: db.Database) -> None:
        self._target_db = new_db
        self._build = False

    @property
    def pre_check(self) -> validation.OptimizationPreCheck:
        return self._pre_check

    @property
    def join_order_enumerator(self) -> stages.JoinOrderOptimization:
        return self._join_order_enumerator

    @property
    def physical_operator_selection(self) -> stages.PhysicalOperatorSelection:
        return self._physical_operator_selection

    @property
    def plan_parameterization(self) -> stages.ParameterGeneration:
        return self._plan_parameterization

    def setup_query_support_check(self, check: validation.OptimizationPreCheck) -> TwoStageOptimizationPipeline:
        """Configures the pre-check to be executed for each query.

        This check will be combined with any additional checks that are required by the actual optimization strategies.
        """
        self._pre_check = check
        self._build = False
        return self

    def setup_join_order_optimization(self,
                                      enumerator: stages.JoinOrderOptimization) -> TwoStageOptimizationPipeline:
        """Configures the algorithm to obtain an optimized join order.

        This algorithm may optionally also determine an initial assignment of physical operators.
        """
        self._join_order_enumerator = enumerator
        self._build = False
        return self

    def setup_physical_operator_selection(self, selector: stages.PhysicalOperatorSelection, *,
                                          overwrite: bool = False) -> TwoStageOptimizationPipeline:
        """Configures the algorithm to assign physical operators to the query.

        This algorithm receives the input query as well as the join order (if there is one) as input. In a special
        case, this join order can also provide an initial assignment of physical operators. These settings can then
        be further adapted by the selected algorithm (or completely overwritten).

        The `overwrite` parameter specifies what should happen if this setup method is called multiple times:
        if `overwrite` is `True`, the new algorithm completely replaces any old strategy. Otherwise, the new strategy
        is chained with the older strategy, i.e. the new strategy can overwrite assignments produced by the old
        strategy. See `PhysicalOperatorSelection.chain_with` for details.
        """
        if not overwrite and self._physical_operator_selection:
            self._physical_operator_selection = self._physical_operator_selection.chain_with(selector)
        else:
            self._physical_operator_selection = selector
        self._build = False
        return self

    def setup_plan_parameterization(self, param_generator: stages.ParameterGeneration, *,
                                    overwrite: bool = False) -> TwoStageOptimizationPipeline:
        """Configures the algorithm to parameterize the query plan.

        This algorithm receives the input query as well as the join order and the physical operators (if those have
        been determined yet) as input.

        The `overwrite` parameter specifies what should happen if this setup method is called multiple times:
        if `overwrite` is `True`, the new algorithm completely replaces any old strategy. Otherwise, the new strategy
        is chained with the older strategy, i.e. the new strategy can overwrite parameters produced by the old
        strategy. See `ParameterGeneration.chain_with` for details.
        """
        if not overwrite and self._plan_parameterization:
            self._plan_parameterization = self._plan_parameterization.chain_with(param_generator)
        else:
            self._plan_parameterization = param_generator
        self._build = False
        return self

    def load_settings(self, optimization_settings: presets.OptimizationSettings) -> TwoStageOptimizationPipeline:
        """Applies all the optimization settings from a pre-defined optimization strategy to the pipeline.

        This is just a shorthand method to skip calling all setup methods individually for a fixed combination of
        optimization.
        """
        support_check = optimization_settings.query_pre_check()
        if support_check:
            self.setup_query_support_check(support_check)
        join_ordering = optimization_settings.build_join_order_optimizer()
        if join_ordering:
            self.setup_join_order_optimization(join_ordering)
        operator_selection = optimization_settings.build_physical_operator_selection()
        if operator_selection:
            self.setup_physical_operator_selection(operator_selection, overwrite=True)
        plan_parameterization = optimization_settings.build_plan_parameterization()
        if plan_parameterization:
            self.setup_plan_parameterization(plan_parameterization, overwrite=True)
        self._build = False
        return self

    def build(self) -> TwoStageOptimizationPipeline:
        """Constructs the optimization pipeline.

         This includes filling all undefined optimization steps with empty strategies. Afterwards, the pipeline is
         ready to optimize queries.
         """
        if not self._pre_check:
            self._pre_check = validation.EmptyPreCheck()
        if not self._join_order_enumerator:
            self._join_order_enumerator = noopt.EmptyJoinOrderOptimizer()
        if not self._physical_operator_selection:
            self._physical_operator_selection = noopt.EmptyPhysicalOperatorSelection()
        if not self._plan_parameterization:
            self._plan_parameterization = noopt.EmptyParameterization()

        all_pre_checks = [self._pre_check] + [check for check in [self._join_order_enumerator.pre_check(),
                                                                  self._physical_operator_selection.pre_check(),
                                                                  self._plan_parameterization.pre_check()]
                                              if check]
        self._pre_check = validation.merge_checks(all_pre_checks)
        self._pre_check.check_supported_database_system(self._target_db)

        self._build = True
        return self

    def optimize_query(self, query: qal.SqlQuery) -> qal.SqlQuery:
        """Optimizes the given input query.

        The output query will be optimized such that the selected target database system is forced to adhere to the
        optimized query plan. What that means exactly depends on the target dbs.

        For example, for Postgres the join order could be enforced using a combination of JOIN ON statements instead
        of implicit joins in combination with the `SET join_collapse_limit = 1` parameter.
        MySQL queries could contain a query hint block in the SELECT clause that specifies physical operators and
        join order.
        """
        self._assert_is_build()
        supported_query_check = self._pre_check.check_supported_query(query)
        if not supported_query_check.passed:
            raise validation.UnsupportedQueryError(query, supported_query_check.failure_reason)

        if isinstance(query, qal.ExplicitSqlQuery):
            query = transform.explicit_to_implicit(query)
        elif not isinstance(query, qal.ImplicitSqlQuery):
            raise ValueError(f"Unknown query type '{type(query)}' for query '{query}'")

        join_order = self._join_order_enumerator.optimize_join_order(query)
        physical_operators = self._physical_operator_selection.select_physical_operators(query, join_order)
        plan_parameters = self._plan_parameterization.generate_plan_parameters(query, join_order, physical_operators)

        return self._target_db.hinting().generate_hints(query, join_order=join_order,
                                                        physical_operators=physical_operators,
                                                        plan_parameters=plan_parameters)

    def describe(self) -> dict:
        """Provides a representation of the selected optimization strategies and the database settings."""
        return {
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
            raise errors.StateError("Pipeline has not been build")

    def __repr__(self) -> str:
        return str(self)

    def __str__(self) -> str:
        components = [self._join_order_enumerator, self._physical_operator_selection, self._plan_parameterization]
        opt_chain = " -> ".join(str(comp) for comp in components)
        return f"OptimizationPipeline [{opt_chain}]"


class IncrementalOptimizationPipeline(OptimizationPipeline):
    pass
