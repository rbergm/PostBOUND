""""""

from qal import qal

class OptimizationPipeline:
    def __init__(self, target_dbs=None) -> None:
        pass

    def setup_join_order_optimization(self, *, join_enumerator=None,
                                      base_cardinality_estimator=None, join_cardinality_estimator=None,
                                      subquery_policy=None) -> "OptimizationPipeline":
        # TODO: implementation and types
        return self

    def setup_physical_operator_selection(self, *, operator_selector=None) -> "OptimizationPipeline":
        # TODO: implementation and types
        return self

    def build(self) -> None:
        pass

    def optimize_query(self, query: qal.SqlQuery) -> qal.SqlQuery:
        pass
