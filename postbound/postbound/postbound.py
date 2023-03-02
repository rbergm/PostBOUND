""""""
from __future__ import annotations

from postbound.qal import qal, transform
from postbound.optimizer import presets, validation
from postbound.optimizer.joinorder import enumeration
from postbound.optimizer.physops import selection
from postbound.db.systems import systems as db_sys
from postbound.util import errors


class OptimizationPipeline:
    def __init__(self, target_dbs: db_sys.DatabaseSystem) -> None:
        self.target_dbs = target_dbs
        self.pre_check: validation.OptimizationPreCheck | None = None
        self.join_order_enumerator: enumeration.JoinOrderOptimizer | None = None
        self.physical_operator_selection: selection.PhysicalOperatorSelection | None = None
        self._build = False

    def setup_query_support_check(self, check: validation.OptimizationPreCheck) -> OptimizationPipeline:
        self.pre_check = check
        return self

    def setup_join_order_optimization(self, enumerator: enumeration.JoinOrderOptimizer) -> OptimizationPipeline:
        self.join_order_enumerator = enumerator
        return self

    def setup_physical_operator_selection(self, selector: selection.PhysicalOperatorSelection) -> OptimizationPipeline:
        self.physical_operator_selection = selector
        return self

    def load_settings(self, optimization_settings: presets.OptimizationSettings) -> None:
        support_check = optimization_settings.query_pre_check()
        if support_check:
            self.setup_query_support_check(support_check)
        join_ordering = optimization_settings.build_join_order_optimizer()
        if join_ordering:
            self.setup_join_order_optimization(join_ordering)
        operator_selection = optimization_settings.build_physical_operator_selection()
        if operator_selection:
            self.setup_physical_operator_selection(operator_selection)

    def build(self) -> None:
        if not self.pre_check:
            self.pre_check = validation.EmptyPreCheck()
        if not self.join_order_enumerator:
            self.join_order_enumerator = enumeration.EmptyJoinOrderOptimizer()
        if not self.physical_operator_selection:
            self.physical_operator_selection = selection.EmptyPhysicalOperatorSelection()
        self._build = True

    def optimize_query(self, query: qal.SqlQuery) -> qal.SqlQuery:
        self._assert_is_build()
        pre_check_passed, failure_reason = self.pre_check.check_supported_query(query)
        if not pre_check_passed:
            raise validation.UnsupportedQueryError(query, failure_reason)

        if query.is_implicit():
            implicit_query: qal.ImplicitSqlQuery = query
        else:
            implicit_query: qal.ImplicitSqlQuery = transform.explicit_to_implicit(query)

        join_order = self.join_order_enumerator.optimize_join_order(implicit_query)
        operators = self.physical_operator_selection.select_physical_operators(implicit_query, join_order)

        return self.target_dbs.query_adaptor().adapt_query(implicit_query, join_order, operators)

    def describe(self) -> dict:
        return {
            "database_system": {
                "name": self.target_dbs.interface().name,
                "statistics": {
                    "emulated": self.target_dbs.interface().statistics().emulated,
                    "cache_enabled": self.target_dbs.interface().statistics().cache_enabled
                }
            },
            "query_pre_check": self.pre_check.describe() if self.pre_check else None,
            "join_ordering": self.join_order_enumerator.describe() if self.join_order_enumerator else None,
            "operator_selection": (self.physical_operator_selection.describe() if self.physical_operator_selection
                                   else None)
        }

    def _assert_is_build(self) -> None:
        if not self._build:
            raise errors.StateError("Pipeline has not been build")
