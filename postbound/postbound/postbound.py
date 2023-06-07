"""Provides PostBOUND's main optimization pipeline."""
from __future__ import annotations

from postbound.qal import qal, transform
from postbound.optimizer import presets, validation
from postbound.optimizer.joinorder import enumeration
from postbound.optimizer.physops import selection
from postbound.optimizer.planmeta import parameterization as plan_param
from postbound.db import db
from postbound.util import errors


class OptimizationPipeline:
    """The optimization pipeline is the main tool to apply and combine different optimization strategies.

    Each pipeline consists of up to four steps, which completely specify the optimization settings that should be
    applied to an incoming query. For each of the steps general interface exist that must be implemented by the
    selected strategies.

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
        self.target_db = target_db
        self.pre_check: validation.OptimizationPreCheck | None = None
        self.join_order_enumerator: enumeration.JoinOrderOptimizer | None = None
        self.physical_operator_selection: selection.PhysicalOperatorSelection | None = None
        self.plan_parameterization: plan_param.ParameterGeneration | None = None
        self._build = False

    def setup_query_support_check(self, check: validation.OptimizationPreCheck) -> OptimizationPipeline:
        """Configures the pre-check to be executed for each query.

        This check will be combined with any additional checks that are required by the actual optimization strategies.
        """
        self.pre_check = check
        return self

    def setup_join_order_optimization(self, enumerator: enumeration.JoinOrderOptimizer) -> OptimizationPipeline:
        """Configures the algorithm to obtain an optimized join order.

        This algorithm may optionally also determine an initial assignment of physical operators.
        """
        self.join_order_enumerator = enumerator
        return self

    def setup_physical_operator_selection(self, selector: selection.PhysicalOperatorSelection, *,
                                          overwrite: bool = False) -> OptimizationPipeline:
        """Configures the algorithm to assign physical operators to the query.

        This algorithm receives the input query as well as the join order (if there is one) as input. In a special
        case, this join order can also provide an initial assignment of physical operators. These settings can then
        be further adapted by the selected algorithm (or completely overwritten).

        The `overwrite` parameter specifies what should happen if this setup method is called multiple times:
        if `overwrite` is `True`, the new algorithm completely replaces any old strategy. Otherwise, the new strategy
        is chained with the older strategy, i.e. the new strategy can overwrite assignments produced by the old
        strategy. See `PhysicalOperatorSelection.chain_with` for details.
        """
        if not overwrite and self.physical_operator_selection:
            self.physical_operator_selection = self.physical_operator_selection.chain_with(selector)
        else:
            self.physical_operator_selection = selector
        return self

    def setup_plan_parameterization(self, param_generator: plan_param.ParameterGeneration, *,
                                    overwrite: bool = False) -> OptimizationPipeline:
        """Configures the algorithm to parameterize the query plan.

        This algorithm receives the input query as well as the join order and the physical operators (if those have
        been determined yet) as input.

        The `overwrite` parameter specifies what should happen if this setup method is called multiple times:
        if `overwrite` is `True`, the new algorithm completely replaces any old strategy. Otherwise, the new strategy
        is chained with the older strategy, i.e. the new strategy can overwrite parameters produced by the old
        strategy. See `ParameterGeneration.chain_with` for details.
        """
        if not overwrite and self.plan_parameterization:
            self.plan_parameterization = self.plan_parameterization.chain_with(param_generator)
        else:
            self.plan_parameterization = param_generator
        return self

    def load_settings(self, optimization_settings: presets.OptimizationSettings) -> OptimizationPipeline:
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
        return self

    def build(self) -> OptimizationPipeline:
        """Constructs the optimization pipeline.

         This includes filling all undefined optimization steps with empty strategies. Afterwards, the pipeline is
         ready to optimize queries.
         """
        if not self.pre_check:
            self.pre_check = validation.EmptyPreCheck()
        if not self.join_order_enumerator:
            self.join_order_enumerator = enumeration.EmptyJoinOrderOptimizer()
        if not self.physical_operator_selection:
            self.physical_operator_selection = selection.EmptyPhysicalOperatorSelection()
        if not self.plan_parameterization:
            self.plan_parameterization = plan_param.EmptyParameterization()

        all_pre_checks = [self.pre_check] + [check for check in [self.join_order_enumerator.pre_check(),
                                                                 self.physical_operator_selection.pre_check(),
                                                                 self.plan_parameterization.pre_check()]
                                             if check]
        self.pre_check = validation.merge_checks(all_pre_checks)

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
        supported_query_check = self.pre_check.check_supported_query(query)
        if not supported_query_check.passed:
            raise validation.UnsupportedQueryError(query, supported_query_check.failure_reason)

        if isinstance(query, qal.ExplicitSqlQuery):
            query = transform.explicit_to_implicit(query)
        elif not isinstance(query, qal.ImplicitSqlQuery):
            raise ValueError(f"Unknown query type '{type(query)}' for query '{query}'")

        join_order = self.join_order_enumerator.optimize_join_order(query)
        operators = self.physical_operator_selection.select_physical_operators(query, join_order)
        plan_parameters = self.plan_parameterization.generate_plan_parameters(query, join_order, operators)

        return self.target_db.hinting().generate_hints(query, join_order=join_order, physical_operators=operators,
                                                       plan_parameters=plan_parameters)

    def describe(self) -> dict:
        """Provides a representation of the selected optimization strategies and the database settings."""
        return {
            "database_system": self.target_db.describe(),
            "query_pre_check": self.pre_check.describe() if self.pre_check else None,
            "join_ordering": self.join_order_enumerator.describe() if self.join_order_enumerator else None,
            "operator_selection": (self.physical_operator_selection.describe() if self.physical_operator_selection
                                   else None),
            "plan_parameterization": self.plan_parameterization.describe() if self.plan_parameterization else None
        }

    def _assert_is_build(self) -> None:
        """Raises an error if the pipeline has not been build yet."""
        if not self._build:
            raise errors.StateError("Pipeline has not been build")
