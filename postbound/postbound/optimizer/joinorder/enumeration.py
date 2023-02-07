from __future__ import annotations

import abc
import operator
import copy

import numpy as np

from postbound.qal import qal, predicates as preds
from postbound.db import db
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
                 stats_container: stats.UpperBoundsContainer,
                 database: db.Database) -> None:
        super().__init__("UES enumeration")
        self.base_table_estimation = base_table_estimation
        self.join_estimation = join_estimation
        self.subquery_policy = subquery_policy
        self.stats_container = stats_container
        self.database = database

    def optimize_join_order(self, query: qal.ImplicitSqlQuery) -> data.JoinTree | None:
        if len(list(query.tables())) <= 2:
            return None

        self.base_table_estimation.setup_for_query(query)
        self.join_estimation.setup_for_query(query)
        self.subquery_policy.setup_for_query(query)
        self.stats_container.setup_for_query(query)

        join_graph = data.JoinGraph(query, self.database.schema())

        if join_graph.contains_cross_products():
            # cross-product query is reduced to multiple independent optimization passes
            optimized_components = []
            for component in join_graph.join_components():
                optimized_component = self._clone().optimize_join_order(component.query)
                if not optimized_component:
                    raise JoinOrderOptimizationError(component.query)
                optimized_components.append(optimized_component)

            # insert cross-products such that the smaller partitions are joined first
            sorted(optimized_components, key=operator.attrgetter("upper_bound"))
            final_join_tree = data.JoinTree.cross_product_of(optimized_components)
        elif join_graph.contains_free_n_m_joins():
            final_join_tree = self._default_ues_optimizer(query, join_graph)
        else:
            final_join_tree = self._star_query_optimizer(join_graph)

        return final_join_tree

    def _default_ues_optimizer(self, query: qal.SqlQuery, join_graph: data.JoinGraph) -> data.JoinTree:
        join_tree = data.JoinTree()

        while join_graph.contains_free_n_m_joins():

            # Update the current upper bounds
            lowest_min_bound = np.inf
            lowest_bound_table = None
            for candidate_join in join_graph.available_join_paths():
                candidate_table = candidate_join.target_table
                filter_estimate = self.stats_container.base_table_estimates[candidate_table]
                pk_fk_bounds = [self.join_estimation.estimate_for(join_path.join_condition, join_graph) for join_path
                                in join_graph.available_pk_fk_joins_for(candidate_table)]
                candidate_min_bound = min([filter_estimate] + pk_fk_bounds)
                self.stats_container.upper_bounds[candidate_table] = candidate_min_bound

                if candidate_min_bound < lowest_min_bound:
                    lowest_min_bound = candidate_min_bound
                    lowest_bound_table = candidate_table

            if join_tree.is_empty():
                filter_predicate = preds.CompoundPredicate.create_and(
                    query.predicates().filters_for(lowest_bound_table))
                join_tree = data.JoinTree.for_base_table(lowest_bound_table, lowest_min_bound, filter_predicate)

        assert not join_graph.contains_free_tables()

        return join_tree

    def _star_query_optimizer(self, join_graph: data.JoinGraph) -> data.JoinTree:
        pass

    def _clone(self) -> UESJoinOrderOptimizer:
        return UESJoinOrderOptimizer(base_table_estimation=copy.copy(self.base_table_estimation),
                                     join_estimation=copy.copy(self.join_estimation),
                                     subquery_policy=copy.copy(self.subquery_policy),
                                     stats_container=copy.copy(self.stats_container),
                                     database=self.database)


class EmptyJoinOrderOptimizer(JoinOrderOptimizer):
    def optimize_join_order(self, query: qal.ImplicitSqlQuery) -> data.JoinTree | None:
        return None


class JoinOrderOptimizationError(RuntimeError):
    def __init__(self, query: qal.SqlQuery, message: str = "") -> None:
        super().__init__(f"Join order optimization failed for query {query}" if not message else message)
        self.query = query
