from __future__ import annotations

import abc

from postbound.qal import base, qal, predicates
from postbound.optimizer import data
from postbound.optimizer.bounds import stats


class SubqueryGenerationPolicy(abc.ABC):
    def __init__(self, name: str) -> None:
        self.name = name

    @abc.abstractmethod
    def setup_for_query(self, query: qal.ImplicitSqlQuery, stats_container: stats.StatisticsContainer) -> None:
        raise NotImplementedError

    @abc.abstractmethod
    def generate_subquery_for(self, join: predicates.AbstractPredicate, join_graph: data.JoinGraph) -> bool:
        raise NotImplementedError

    @abc.abstractmethod
    def describe(self) -> dict:
        raise NotImplementedError


class LinearSubqueryGenerationPolicy(SubqueryGenerationPolicy):
    def __init__(self):
        super().__init__("Linear subquery policy")

    def setup_for_query(self, query: qal.ImplicitSqlQuery, stats_container: stats.StatisticsContainer) -> None:
        pass

    def generate_subquery_for(self, join: predicates.AbstractPredicate, join_graph: data.JoinGraph) -> bool:
        return False

    def describe(self) -> dict:
        return {"name": "linear"}


class UESSubqueryGenerationPolicy(SubqueryGenerationPolicy):
    def __init__(self):
        super().__init__("UES subquery policy")
        self.query: qal.SqlQuery | None = None
        self.stats_container: stats.StatisticsContainer | None = None

    def setup_for_query(self, query: qal.ImplicitSqlQuery, stats_container: stats.StatisticsContainer) -> None:
        self.query = query
        self.stats_container = stats_container

    def generate_subquery_for(self, join: predicates.AbstractPredicate, join_graph: data.JoinGraph) -> bool:
        if join_graph.count_consumed_tables() < 2:
            return False

        joined_table: base.TableReference | None = None
        for table in join.tables():
            if join_graph.is_free_table(table):
                joined_table = table
                break

        return self.stats_container.upper_bounds[joined_table] < self.stats_container.base_table_estimates[joined_table]

    def describe(self) -> dict:
        return {"name": "defensive"}
