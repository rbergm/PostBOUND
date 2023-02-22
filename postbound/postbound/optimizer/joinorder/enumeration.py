from __future__ import annotations

import abc
import operator
import copy
from typing import Iterable

import numpy as np

from postbound.qal import qal, base, predicates as preds
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
                 stats_container: stats.StatisticsContainer,
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
        self.stats_container.setup_for_query(query, self.base_table_estimation)
        self.join_estimation.setup_for_query(query, self.stats_container)
        self.subquery_policy.setup_for_query(query, self.stats_container)

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
            final_join_tree = data.JoinTree.cross_product_of(*optimized_components)
        elif join_graph.contains_free_n_m_joins():
            final_join_tree = self._default_ues_optimizer(query, join_graph)
        else:
            final_join_tree = self._star_query_optimizer(join_graph)

        return final_join_tree

    def _default_ues_optimizer(self, query: qal.SqlQuery, join_graph: data.JoinGraph) -> data.JoinTree:
        join_tree = data.JoinTree()

        while join_graph.contains_free_n_m_joins():

            # Update the current upper bounds
            lowest_bound = np.inf
            lowest_bound_table = None
            for candidate_join in join_graph.available_join_paths():
                candidate_table = candidate_join.target_table
                filter_estimate = self.stats_container.base_table_estimates[candidate_table]
                pk_fk_bounds = [self.join_estimation.estimate_for(join_path.join_condition, join_graph) for join_path
                                in join_graph.available_pk_fk_joins_for(candidate_table)]
                candidate_min_bound = min([filter_estimate] + pk_fk_bounds)
                self.stats_container.upper_bounds[candidate_table] = candidate_min_bound

                if candidate_min_bound < lowest_bound:
                    lowest_bound = candidate_min_bound
                    lowest_bound_table = candidate_table

            if join_tree.is_empty():
                filter_pred = preds.CompoundPredicate.create_and(query.predicates().filters_for(lowest_bound_table))
                join_tree = data.JoinTree.for_base_table(lowest_bound_table, lowest_bound, filter_pred)
                join_graph.mark_joined(lowest_bound_table)
                self.stats_container.upper_bounds[join_tree] = lowest_bound
                pk_joins = join_graph.available_deep_pk_join_paths_for(lowest_bound_table,
                                                                       self._table_base_cardinality_ordering)
                for pk_join in pk_joins:
                    target_table = pk_join.target_table
                    base_cardinality = self.stats_container.base_table_estimates[target_table]
                    filter_pred = preds.CompoundPredicate.create_and(query.predicates().filters_for(target_table))
                    join_bound = self.join_estimation.estimate_for(pk_join.join_condition, join_graph)
                    join_graph.mark_joined(target_table, pk_join.join_condition)
                    join_tree = join_tree.join_with_base_table(pk_join.target_table, base_cardinality=base_cardinality,
                                                               base_filter_predicate=filter_pred,
                                                               join_predicate=pk_join.join_condition,
                                                               join_bound=join_bound, n_m_join=False)

            selected_candidate: data.JoinPath | None = None
            lowest_bound = np.inf
            for candidate_join in join_graph.available_join_paths():
                candidate_bound = self.join_estimation.estimate_for(candidate_join.join_condition, join_graph)
                if candidate_bound < lowest_bound:
                    selected_candidate = candidate_join
                    lowest_bound = candidate_bound

            direct_pk_joins = join_graph.available_pk_fk_joins_for(selected_candidate.target_table)
            create_subquery = any(self.subquery_policy.generate_subquery_for(pk_join.join_condition, join_graph)
                                  for pk_join in direct_pk_joins)
            all_pk_joins = join_graph.available_deep_pk_join_paths_for(selected_candidate.target_table)
            candidate_table = selected_candidate.target_table
            candidate_filters = preds.CompoundPredicate.create_and(query.predicates().filters_for(candidate_table))
            candidate_base_cardinality = self.stats_container.base_table_estimates[candidate_table]
            if create_subquery:
                subquery_tree = data.JoinTree.for_base_table(candidate_table, candidate_base_cardinality,
                                                             candidate_filters)
                join_graph.mark_joined(candidate_table)
                self._insert_pk_joins(query, all_pk_joins, subquery_tree, join_graph)
                join_tree = join_tree.join_with_subquery(subquery_tree, selected_candidate.join_condition, lowest_bound,
                                                         n_m_table=candidate_table)
                self.stats_container.upper_bounds[join_tree] = lowest_bound
            else:
                join_tree = join_tree.join_with_base_table(candidate_table, base_cardinality=candidate_base_cardinality,
                                                           join_predicate=selected_candidate.join_condition,
                                                           join_bound=lowest_bound,
                                                           base_filter_predicate=candidate_filters)
                join_graph.mark_joined(candidate_table, selected_candidate.join_condition)
                self.stats_container.upper_bounds[join_tree] = lowest_bound
                join_tree = self._insert_pk_joins(query, all_pk_joins, join_tree, join_graph)

        if join_graph.contains_free_tables():
            raise AssertionError("Join graph still has free tables remaining!")
        return join_tree

    def _star_query_optimizer(self, join_graph: data.JoinGraph) -> data.JoinTree:
        # TODO: implementation
        pass

    def _table_base_cardinality_ordering(self, table: base.TableReference, join_edge: dict) -> int:
        return self.stats_container.base_table_estimates[table]

    def _insert_pk_joins(self, query: qal.SqlQuery, pk_joins: Iterable[data.JoinPath],
                         join_tree: data.JoinTree, join_graph: data.JoinGraph) -> data.JoinTree:
        for pk_join in pk_joins:
            pk_table = pk_join.target_table
            pk_filters = preds.CompoundPredicate.create_and(query.predicates().filters_for(pk_table))
            pk_join_bound = self.join_estimation.estimate_for(pk_join.join_condition, join_graph)
            pk_base_cardinality = self.stats_container.base_table_estimates[pk_table]
            join_tree = join_tree.join_with_base_table(pk_table, base_cardinality=pk_base_cardinality,
                                                       join_predicate=pk_join.join_condition,
                                                       join_bound=pk_join_bound,
                                                       base_filter_predicate=pk_filters,
                                                       n_m_join=False)
            join_graph.mark_joined(pk_table, pk_join.join_condition)
            self.stats_container.upper_bounds[join_tree] = pk_join_bound
        return join_tree

    def _clone(self) -> UESJoinOrderOptimizer:
        return UESJoinOrderOptimizer(base_table_estimation=copy.copy(self.base_table_estimation),
                                     join_estimation=copy.copy(self.join_estimation),
                                     subquery_policy=copy.copy(self.subquery_policy),
                                     stats_container=copy.copy(self.stats_container),
                                     database=self.database)


class EmptyJoinOrderOptimizer(JoinOrderOptimizer):
    def __init__(self) -> None:
        super().__init__("empty")

    def optimize_join_order(self, query: qal.ImplicitSqlQuery) -> data.JoinTree | None:
        return None


class JoinOrderOptimizationError(RuntimeError):
    def __init__(self, query: qal.SqlQuery, message: str = "") -> None:
        super().__init__(f"Join order optimization failed for query {query}" if not message else message)
        self.query = query
