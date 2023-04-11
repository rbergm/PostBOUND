"""Contains interfaces to influence the structure of a join tree."""
from __future__ import annotations

import abc
from typing import Optional

from postbound.qal import base, qal, predicates
from postbound.optimizer import data, validation
from postbound.optimizer.bounds import stats


class SubqueryGenerationPolicy(abc.ABC):
    """This policy allows to adapt the structure of the resulting join tree during join order optimization.

    It is intended to customize when branches should be created in a join tree and to compare different such
    strategies.

    Although the terminology here is subquery-related, the policy is actually more general and concerned with branches
    in the join tree. Whether these are achieved via subqueries during query execution is merely a technical detail
    (albeit a useful analogy).
    """

    def __init__(self, name: str) -> None:
        self.name = name

    @abc.abstractmethod
    def setup_for_query(self, query: qal.ImplicitSqlQuery, stats_container: stats.StatisticsContainer) -> None:
        """Enables the setup of internal data structures to enable decisions about the given query."""
        raise NotImplementedError

    @abc.abstractmethod
    def generate_subquery_for(self, join: predicates.AbstractPredicate, join_graph: data.JoinGraph) -> bool:
        """Decides whether the given join should be executed in a subquery.

        Notice that the supplied join predicate really is the one that should be executed in a subquery, it is not the
        predicate that should be used to merge the result of two subqueries.
        """
        raise NotImplementedError

    @abc.abstractmethod
    def describe(self) -> dict:
        """Provides a representation of the branching strategy."""
        raise NotImplementedError

    def pre_check(self) -> Optional[validation.OptimizationPreCheck]:
        """Provides requirements that an input query has to satisfy in order for the policy to work properly."""
        return None


class LinearSubqueryGenerationPolicy(SubqueryGenerationPolicy):
    """Subquery strategy that leaves all join paths linear, i.e. does not generate subqueries at all."""

    def __init__(self):
        super().__init__("Linear subquery policy")

    def setup_for_query(self, query: qal.ImplicitSqlQuery, stats_container: stats.StatisticsContainer) -> None:
        pass

    def generate_subquery_for(self, join: predicates.AbstractPredicate, join_graph: data.JoinGraph) -> bool:
        return False

    def describe(self) -> dict:
        return {"name": "linear"}


class UESSubqueryGenerationPolicy(SubqueryGenerationPolicy):
    """The subquery strategy as proposed in the UES paper.

    In short, this strategy generates subqueries if they guarantee a reduction of the upper bounds of the higher-level
    join.

    See Hertzschuch et al.: "Simplicity Done Right for Join Ordering", CIDR'2021 for details.
    """

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

        stats = self.stats_container
        return stats.upper_bounds[joined_table] < stats.base_table_estimates[joined_table]

    def describe(self) -> dict:
        return {"name": "defensive"}
