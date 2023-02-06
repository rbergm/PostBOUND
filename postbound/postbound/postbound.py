""""""
from __future__ import annotations

from qal import qal, transform
from optimizer.joinorder import enumeration
from optimizer.physops import selection
from db.systems import systems as db_sys


class OptimizationPipeline:
    def __init__(self, target_dbs: db_sys.DatabaseSystem) -> None:
        self.target_dbs = target_dbs
        self.join_order_enumerator: enumeration.UESJoinOrderOptimizer | None = None
        self.physical_operator_selection: selection.PhysicalOperatorSelection | None = None

    def setup_join_order_optimization(self, enumerator: enumeration.JoinOrderOptimizer) -> OptimizationPipeline:
        self.join_order_enumerator = enumerator
        return self

    def setup_physical_operator_selection(self, selector: selection.PhysicalOperatorSelection) -> OptimizationPipeline:
        self.physical_operator_selection = selector
        return self

    def build(self) -> None:
        if not self.join_order_enumerator:
            self.join_order_enumerator = enumeration.EmptyJoinOrderOptimizer
        if not self.physical_operator_selection:
            self.physical_operator_selection = selection.EmptyPhysicalOperatorSelection

    def optimize_query(self, query: qal.SqlQuery) -> qal.SqlQuery:
        if query.is_implicit():
            implicit_query: qal.ImplicitSqlQuery = query
        else:
            implicit_query: qal.ImplicitSqlQuery = transform.explicit_to_implicit(query)
        join_order = self.join_order_enumerator.optimize_join_order(implicit_query)
        operators = self.physical_operator_selection.select_physical_operators(implicit_query, join_order)

        return self.target_dbs.query_adaptor().adapt_query(implicit_query, join_order, operators)
