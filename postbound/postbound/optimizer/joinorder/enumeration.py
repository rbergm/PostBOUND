from __future__ import annotations

import abc

from postbound.qal import qal
from postbound.optimizer.bounds import joins as join_bounds, scans as scan_bounds, subqueries, stats
from postbound.optimizer import data


class JoinOrderOptimizer(abc.ABC):

    def __init__(self, name: str):
        self.name = name

    @abc.abstractmethod
    def optimize_join_order(self, query: qal.ImplicitSqlQuery) -> data.JoinTree | None:
        raise NotImplementedError


class UESJoinOrderOptimizer(JoinOrderOptimizer):
    def __init__(self, *, base_table_estimation: scan_bounds.BaseTableCardinalityEstimator,
                 join_estimation: join_bounds.JoinBoundCardinalityEstimator,
                 subquery_policy: subqueries.SubqueryGenerationPolicy,
                 stats_container: stats.UpperBoundsContainer) -> None:
        super().__init__("UES enumeration")
        self.base_table_estimation = base_table_estimation
        self.join_estimation = join_estimation
        self.subquery_policy = subquery_policy
        self.stats_container = stats_container


class EmptyJoinOrderOptimizer(JoinOrderOptimizer):
    def optimize_join_order(self, query: qal.ImplicitSqlQuery) -> data.JoinTree | None:
        return None
