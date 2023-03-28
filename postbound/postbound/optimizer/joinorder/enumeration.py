from __future__ import annotations

import abc
import operator
import copy
from typing import Iterable

import numpy as np

from postbound.qal import qal, base, predicates as preds
from postbound.db import db
from postbound.optimizer import data, validation
from postbound.optimizer.bounds import joins as join_bounds, scans as scan_bounds, stats
from postbound.optimizer.joinorder import subqueries


class JoinOrderOptimizer(abc.ABC):

    def __init__(self, name: str):
        self.name = name

    @abc.abstractmethod
    def optimize_join_order(self, query: qal.ImplicitSqlQuery) -> data.JoinTree | None:
        raise NotImplementedError

    @abc.abstractmethod
    def describe(self) -> dict:
        raise NotImplementedError

    def pre_check(self) -> validation.OptimizationPreCheck:
        return validation.EmptyPreCheck()


def _fetch_filters(query: qal.SqlQuery, table: base.TableReference) -> preds.AbstractPredicate | None:
    all_filters = query.predicates().filters_for(table)
    predicate = preds.CompoundPredicate.create_and(all_filters) if all_filters else None
    return predicate


class UESJoinOrderOptimizer(JoinOrderOptimizer):
    def __init__(self, *, base_table_estimation: scan_bounds.BaseTableCardinalityEstimator,
                 join_estimation: join_bounds.JoinBoundCardinalityEstimator,
                 subquery_policy: subqueries.SubqueryGenerationPolicy,
                 stats_container: stats.StatisticsContainer,
                 database: db.Database, verbose: bool = False) -> None:
        super().__init__("UES enumeration")
        self.base_table_estimation = base_table_estimation
        self.join_estimation = join_estimation
        self.subquery_policy = subquery_policy
        self.stats_container = stats_container
        self.database = database
        self._logging_enabled = verbose

    def optimize_join_order(self, query: qal.ImplicitSqlQuery) -> data.JoinTree | None:
        if len(query.tables()) <= 2:
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
                # FIXME: join components might consist of single tables!
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
            final_join_tree = self._star_query_optimizer(query, join_graph)

        return final_join_tree

    def describe(self) -> dict:
        return {
            "name": "ues",
            "settings": {
                "base_table_estimation": self.base_table_estimation.describe(),
                "join_estimation": self.join_estimation.describe(),
                "subqueries": self.subquery_policy.describe(),
                "statistics": self.stats_container.describe()
            }
        }

    def pre_check(self) -> validation.OptimizationPreCheck:
        specified_checks = [check for check in [self.base_table_estimation.pre_check(),
                                                self.join_estimation.pre_check(),
                                                self.subquery_policy.pre_check()]
                            if check]
        specified_checks.append(validation.UESOptimizationPreCheck())
        return validation.merge_checks(specified_checks)

    def _default_ues_optimizer(self, query: qal.SqlQuery, join_graph: data.JoinGraph) -> data.JoinTree:
        join_tree = data.JoinTree()

        while join_graph.contains_free_n_m_joins():

            # Update the current upper bounds
            lowest_bound = np.inf
            lowest_bound_table = None
            for candidate_join in join_graph.available_n_m_join_paths():
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
                filter_pred = _fetch_filters(query, lowest_bound_table)
                join_tree = data.JoinTree.for_base_table(lowest_bound_table, lowest_bound, filter_pred)
                join_graph.mark_joined(lowest_bound_table)
                self.stats_container.upper_bounds[join_tree] = lowest_bound
                pk_joins = join_graph.available_deep_pk_join_paths_for(lowest_bound_table,
                                                                       self._table_base_cardinality_ordering)
                for pk_join in pk_joins:
                    target_table = pk_join.target_table
                    base_cardinality = self.stats_container.base_table_estimates[target_table]
                    filter_pred = _fetch_filters(query, target_table)
                    join_bound = self.join_estimation.estimate_for(pk_join.join_condition, join_graph)
                    join_graph.mark_joined(target_table, pk_join.join_condition)
                    join_tree = join_tree.join_with_base_table(pk_join.target_table, base_cardinality=base_cardinality,
                                                               base_filter_predicate=filter_pred,
                                                               join_predicate=pk_join.join_condition,
                                                               join_bound=join_bound, n_m_join=False)
                self._log_optimization_progress("Initial table selection", lowest_bound_table, pk_joins)
                continue

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
            candidate_table = selected_candidate.target_table
            all_pk_joins = join_graph.available_deep_pk_join_paths_for(candidate_table)
            candidate_filters = _fetch_filters(query, candidate_table)
            candidate_base_cardinality = self.stats_container.base_table_estimates[candidate_table]
            self._log_optimization_progress("n:m join", candidate_table, all_pk_joins,
                                            join_condition=selected_candidate.join_condition,
                                            subquery_join=create_subquery)
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

    def _star_query_optimizer(self, query: qal.ImplicitSqlQuery, join_graph: data.JoinGraph) -> data.JoinTree:
        # initial table / join selection
        lowest_bound = np.inf
        lowest_bound_join = None
        for candidate_join in join_graph.available_join_paths():
            current_bound = self.join_estimation.estimate_for(candidate_join.join_condition, join_graph)
            if current_bound < lowest_bound:
                lowest_bound = current_bound
                lowest_bound_join = candidate_join

        start_table, target_table = lowest_bound_join.start_table, lowest_bound_join.target_table
        start_filters = _fetch_filters(query, start_table)
        join_tree = data.JoinTree.for_base_table(start_table, self.stats_container.base_table_estimates[start_table],
                                                 start_filters)
        join_graph.mark_joined(start_table)
        join_tree = self._apply_pk_fk_join(query, lowest_bound_join, join_bound=lowest_bound, join_graph=join_graph,
                                           current_join_tree=join_tree)

        # join partner selection
        while join_graph.contains_free_tables():
            lowest_bound = np.inf
            lowest_bound_join = None
            for candidate_join in join_graph.available_join_paths():
                current_bound = self.join_estimation.estimate_for(candidate_join.join_condition, join_graph)
                if current_bound < lowest_bound:
                    lowest_bound = current_bound
                    lowest_bound_join = candidate_join

            join_tree = self._apply_pk_fk_join(query, lowest_bound_join, join_bound=lowest_bound, join_graph=join_graph,
                                               current_join_tree=join_tree)

        return join_tree

    def _table_base_cardinality_ordering(self, table: base.TableReference, join_edge: dict) -> int:
        return self.stats_container.base_table_estimates[table]

    def _apply_pk_fk_join(self, query: qal.SqlQuery, pk_fk_join: data.JoinPath, *, join_bound: int,
                          join_graph: data.JoinGraph, current_join_tree: data.JoinTree) -> data.JoinTree:
        target_table = pk_fk_join.target_table
        target_filters = _fetch_filters(query, target_table)
        target_cardinality = self.stats_container.base_table_estimates[target_table]
        updated_join_tree = current_join_tree.join_with_base_table(target_table,
                                                                   join_predicate=pk_fk_join.join_condition,
                                                                   base_cardinality=target_cardinality,
                                                                   join_bound=join_bound,
                                                                   n_m_join=False,
                                                                   base_filter_predicate=target_filters)
        join_graph.mark_joined(target_table, pk_fk_join.join_condition)
        self.stats_container.upper_bounds[updated_join_tree] = join_bound
        return updated_join_tree

    def _insert_pk_joins(self, query: qal.SqlQuery, pk_joins: Iterable[data.JoinPath],
                         join_tree: data.JoinTree, join_graph: data.JoinGraph) -> data.JoinTree:
        for pk_join in pk_joins:
            pk_table = pk_join.target_table
            if not join_graph.is_free_table(pk_table):
                continue
            pk_filters = _fetch_filters(query, pk_table)
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

    def _log_optimization_progress(self, phase: str, candidate_table: base.TableReference,
                                   pk_joins: Iterable[data.JoinPath], *,
                                   join_condition: preds.AbstractPredicate | None = None,
                                   subquery_join: bool | None = None) -> None:
        # TODO: use proper logging
        if not self._logging_enabled:
            return
        log_components = [phase, "::", str(candidate_table), "with PK joins", str(pk_joins)]
        if join_condition:
            log_components.extend(["on condition", str(join_condition)])
        if subquery_join is not None:
            log_components.append("with subquery" if subquery_join else "without subquery")
        log_message = " ".join(log_components)
        print(log_message)


class EmptyJoinOrderOptimizer(JoinOrderOptimizer):
    def __init__(self) -> None:
        super().__init__("empty")

    def optimize_join_order(self, query: qal.ImplicitSqlQuery) -> data.JoinTree | None:
        return None

    def describe(self) -> dict:
        return {"name": "no_ordering"}


class JoinOrderOptimizationError(RuntimeError):
    def __init__(self, query: qal.SqlQuery, message: str = "") -> None:
        super().__init__(f"Join order optimization failed for query {query}" if not message else message)
        self.query = query
