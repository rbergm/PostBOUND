from __future__ import annotations

import abc
import copy
import math
import operator
import typing
from collections.abc import Iterable
from typing import Generic, Optional

import numpy as np

from postbound.db import db
from postbound.qal import base, qal, predicates
from postbound.optimizer import joingraph, jointree, physops, stages, validation
from postbound.optimizer.policies import cardinalities as card_policy, jointree as tree_policy
from postbound.util import collections as collection_utils, dicts as dict_utils

# TODO: adjust documentation

ColumnType = typing.TypeVar("ColumnType")
"""The type of the columns for which statistics are generated."""

StatsType = typing.TypeVar("StatsType")
"""The type of the actual statistics that are stored."""

MaxFrequency = typing.NewType("MaxFrequency", int)
"""Type alias for maximum frequency statistics of columns (which are just an integer).

The maximum frequency of a column is the maximum number of occurrences of a column value within that column.

For example, consider a column `R.a` with values `[a, b, a, a, b, c]`. In this case, maximum column frequency for `R.a`
is 3.
"""

MostCommonElements = typing.NewType("MostCommonElements", list[tuple[Generic[ColumnType], int]])
"""Type alias for top-k lists statistics. The top-k list is generic over the actual column type."""


class StatisticsContainer(abc.ABC, Generic[StatsType]):
    """The statistics container eases the management of the statistics lifecycle.

    It provides means to store different kinds of statistics and can take care of their update automatically. Each
    statistics container instance is intended for one specific query and has to be initialized for that query using the
    `setup_for_query` method.

    A container can store the following statistics:

    - `base_table_estimates` are intended for tables that are not part of the intermediate result, yet. These estimates
    approximate the number of rows that are returned when scanning the table.
    - `upper_bounds` contain the cardinality estimates for intermediate results of the input query. Inserting new
    bounds into that attribute can result in an update of the column statistics.
    - `attribute_frequencies` contain the current statistics value for individual columns. This is the main data
    structure that has to be maintained during the query optimization process to update the column statistics once
    they become part of an intermediate result (and get changed as part of the join process).

    The update of the intermediate attribute frequencies can be turned off by setting `disable_auto_updates` to
    `False` when creating the statistics container.

    A statistics container is abstract to enable a tailored implementation of the loading and updating procedures for
    different statistics types.


    Warning: the current implementation of the statistics container is quite tailored to the UES join enumeration
    algorithm. This can become problematic to underlying assumptions of when and how a statistics update will happen.
    For many scenarios, it will be better to overwrite the `trigger_trigger_frequency_update` method or study
    its current implementation carefully.
    """

    def __init__(self) -> None:
        self.base_table_estimates: dict[base.TableReference, int] = {}
        self.upper_bounds: dict[base.TableReference | jointree.LogicalJoinTree, int] = {}
        self.attribute_frequencies: dict[base.ColumnReference, StatsType] = {}

        self.query: qal.SqlQuery | None = None
        self.base_table_estimator: card_policy.BaseTableCardinalityEstimator | None = None

    def setup_for_query(self, query: qal.SqlQuery,
                        base_table_estimator: card_policy.BaseTableCardinalityEstimator) -> None:
        """Initializes the internal data of the statistics container for th given query.

        The `base_table_estimator` is used to inflate the `base_table_estimates` using the tables that are contained
        in the query. It is assumed that the estimator has to be set up already.
        """
        self.query = query
        self.base_table_estimator = base_table_estimator

        self._inflate_base_table_estimates()
        self._inflate_attribute_frequencies()

    def join_bounds(self) -> dict[jointree.LogicalJoinTree, int]:
        """Provides the cardinality estimates of all join trees that are currently stored in the container."""
        return {join_tree: bound for join_tree, bound in self.upper_bounds.items()
                if isinstance(join_tree, jointree.LogicalJoinTree)}

    def trigger_frequency_update(self, join_tree: jointree.LogicalJoinTree, joined_table: base.TableReference,
                                 join_condition: predicates.AbstractPredicate) -> None:
        """Updates the `attribute_frequencies` according to the supplied join tree.

        This method is usually executed automatically when new join trees are inserted into the upper bounds.

        The frequency update uses a delta-based approach, i.e. it figures out the new join based on the current
        contents of the upper bounds (especially the stored join trees) and the supplied join tree. Therefore, it is
        important to store all intermediate join trees in the upper bounds to ensure the correct operation of the
        frequency update.

        The update procedure distinguishes between two different types of column statistics and uses different
        (and statistics-dependent) update methods for each: partner columns and third-party columns.

        Partner columns are those columns from the intermediate query result, that are directly involved in the join
        predicate, i.e. they are a join partner for some column of the newly joined table. On the other hand, third
        party columns are part of the intermediate result, but not directly involved in the join. In order to update
        them, some sort of correlation info is usually required.
        """
        partner_columns = join_condition.join_partners_of(joined_table)
        third_party_columns = set(col for col in join_tree.join_columns()
                                  if col.table != joined_table and col not in partner_columns)

        for col1, col2 in join_condition.join_partners():
            joined_column, partner_column = (col1, col2) if col1.table == joined_table else (col2, col1)
            self._update_partner_column_frequency(joined_column, partner_column)

        joined_columns_frequencies = {joined_col: self.attribute_frequencies[joined_col] for joined_col
                                      in join_condition.columns_of(joined_table)}
        lowest_joined_column_frequency = dict_utils.argmin(joined_columns_frequencies)
        for third_party_column in third_party_columns:
            self._update_third_party_column_frequency(lowest_joined_column_frequency, third_party_column)

    @abc.abstractmethod
    def describe(self) -> dict:
        """
        Provides a representation of the statistics container and its configuration (but not its current contents).
        """
        raise NotImplementedError

    def _inflate_base_table_estimates(self):
        """Retrieves the base table estimate for each table in the query."""
        for table in self.query.tables():
            table_estimate = self.base_table_estimator.estimate_for(table)
            self.base_table_estimates[table] = table_estimate

    @abc.abstractmethod
    def _inflate_attribute_frequencies(self):
        """Loads the attribute frequency/statistics for all required columns.

        Which statistics to load and for which columns this is required is completely up to the specific statistics
        container.
        """
        raise NotImplementedError

    @abc.abstractmethod
    def _update_partner_column_frequency(self, joined_column: base.ColumnReference,
                                         partner_column: base.ColumnReference) -> None:
        """Performs the frequency update for the partner column.

        This implies that there has been a join between the joined column and the partner column, where the partner
        column is already part of the intermediate result. Likewise, the joined column has just become part of the
        intermediate result as of this join.
        """
        raise NotImplementedError

    @abc.abstractmethod
    def _update_third_party_column_frequency(self, joined_column: base.ColumnReference,
                                             third_party_column: base.ColumnReference) -> None:
        """Performs the frequency update for the third party column (see `trigger_frequency_update`).

        This implies that there has been a join between the joined column and some other column from the intermediate
        result. The third party columns was part of the intermediate result already, but not directly involved in the
        join. The joined column has just become part of the intermediate result as of this join.
        """
        raise NotImplementedError


class MaxFrequencyStatsContainer(StatisticsContainer[MaxFrequency]):
    """Statistics container that stores the maximum frequency of the join columns.

    See `MaxFrequency` for more details on this statistic. The frequency updates happen pessimistically.
    """

    def __init__(self, database_stats: db.DatabaseStatistics):
        super().__init__()
        self.database_stats = database_stats

    def describe(self) -> dict:
        return {"name": "max_column_frequency"}

    def _inflate_attribute_frequencies(self):
        referenced_columns = set()
        for join_predicate in self.query.predicates().joins():
            referenced_columns |= join_predicate.columns()

        for column in referenced_columns:
            top1_list = self.database_stats.most_common_values(column, k=1)
            mcv_value, mcv_frequency = collection_utils.simplify(top1_list)
            self.attribute_frequencies[column] = mcv_frequency

    def _update_partner_column_frequency(self, joined_column: base.ColumnReference,
                                         partner_column: base.ColumnReference) -> None:
        joined_frequency = self.attribute_frequencies[joined_column]
        partner_frequency = self.attribute_frequencies[partner_column]
        self.attribute_frequencies[joined_column] *= partner_frequency
        self.attribute_frequencies[partner_column] *= joined_frequency

    def _update_third_party_column_frequency(self, joined_column: base.ColumnReference,
                                             third_party_column: base.ColumnReference) -> None:
        self.attribute_frequencies[third_party_column] *= self.attribute_frequencies[joined_column]


class UESJoinBoundEstimator(card_policy.JoinBoundCardinalityEstimator):
    """Join cardinality estimator that produces upper bounds according to the formula described in the UES paper.

    This requires maximum frequency statistics over the join columns in order to work properly.

    See Hertzschuch et al.: "Simplicity Done Right for Join Ordering", CIDR'2021 for details.
    """

    def __init__(self) -> None:
        super().__init__("UES join estimator")
        self.query: qal.ImplicitSqlQuery | None = None
        self.stats_container: StatisticsContainer[MaxFrequency] | None = None

    def setup_for_query(self, query: qal.SqlQuery) -> None:
        self.query = query

    def setup_for_stats(self, stats_container: StatisticsContainer[MaxFrequency]) -> None:
        self.stats_container = stats_container

    def estimate_for(self, join_edge: predicates.AbstractPredicate, join_graph: joingraph.JoinGraph) -> int:
        current_min_bound = np.inf

        for base_predicate in join_edge.base_predicates():
            first_col, second_col = collection_utils.simplify(base_predicate.join_partners())
            if join_graph.is_pk_fk_join(first_col.table, second_col.table):
                join_bound = self._estimate_pk_fk_join(first_col, second_col)
            elif join_graph.is_pk_fk_join(second_col.table, first_col.table):
                join_bound = self._estimate_pk_fk_join(second_col, first_col)
            else:
                join_bound = self._estimate_n_m_join(first_col, second_col)

            if join_bound < current_min_bound:
                current_min_bound = join_bound

        return current_min_bound

    def describe(self) -> dict:
        return {"name": "ues"}

    def pre_check(self) -> Optional[validation.OptimizationPreCheck]:
        # TODO: the UES check is slightly too restrictive here.
        # It suffices to check that there are only conjunctive equi joins.
        return validation.UESOptimizationPreCheck()

    def _estimate_pk_fk_join(self, fk_column: base.ColumnReference, pk_column: base.ColumnReference) -> int:
        """Estimation formula for primary key/foreign key joins."""
        pk_cardinality = self.stats_container.base_table_estimates[pk_column.table]
        fk_frequency = self.stats_container.attribute_frequencies[fk_column]
        return math.ceil(fk_frequency * pk_cardinality)

    def _estimate_n_m_join(self, first_column: base.ColumnReference, second_column: base.ColumnReference) -> int:
        """Estimation formula for n:m joins."""
        first_bound, second_bound = self._fetch_bound(first_column), self._fetch_bound(second_column)
        first_freq = self.stats_container.attribute_frequencies[first_column]
        second_freq = self.stats_container.attribute_frequencies[second_column]

        if any(var == 0 for var in [first_bound, second_bound, first_freq, second_freq]):
            return 0

        first_distinct_vals = first_bound / first_freq
        second_distinct_vals = second_bound / second_freq

        n_m_bound = min(first_distinct_vals, second_distinct_vals) * first_freq * second_freq
        return math.ceil(n_m_bound)

    def _fetch_bound(self, column: base.ColumnReference) -> int:
        """Provides the appropriate table bound (based on upper bound or base table estimate) for the given column."""
        table = column.table
        return (self.stats_container.upper_bounds[table] if table in self.stats_container.upper_bounds
                else self.stats_container.base_table_estimates[table])


class UESSubqueryGenerationPolicy(tree_policy.BranchGenerationPolicy):
    """The subquery strategy as proposed in the UES paper.

    In short, this strategy generates subqueries if they guarantee a reduction of the upper bounds of the higher-level
    join.

    See Hertzschuch et al.: "Simplicity Done Right for Join Ordering", CIDR'2021 for details.
    """

    def __init__(self):
        super().__init__("UES subquery policy")
        self.query: qal.SqlQuery | None = None
        self.stats_container: StatisticsContainer | None = None

    def setup_for_query(self, query: qal.SqlQuery) -> None:
        self.query = query

    def setup_for_stats_container(self, stats_container: StatisticsContainer) -> None:
        self.stats_container = stats_container

    def generate_subquery_for(self, join: predicates.AbstractPredicate, join_graph: joingraph.JoinGraph) -> bool:
        if join_graph.count_consumed_tables() < 2:
            return False

        joined_table: base.TableReference | None = None
        for table in join.tables():
            if join_graph.is_free_table(table):
                joined_table = table
                break

        stats_container = self.stats_container
        return stats_container.upper_bounds[joined_table] < stats_container.base_table_estimates[joined_table]

    def describe(self) -> dict:
        return {"name": "defensive"}


class UESJoinOrderOptimizer(stages.JoinOrderOptimization):
    """Implementation of the UES join order algorithm.

    See Hertzschuch et al.: "Simplicity Done Right for Join Ordering", CIDR'2021 for a formal introduction into the
    algorithm

    The actual implementation used here expands upon the original algorithm in several ways:

    - conjunctive join predicates are supported by using the smallest bound of all components of the conjunction
    - queries with cross products are optimized on a per-connected-component basis. The final join order is created
    such that components with smaller upper bounds are joined first
    - star queries that only consist of primary key/foreign key joins are optimized using a derived algorithm:
    the smallest bound pk/fk join is used as the first join. Afterwards, the smallest bound tables are inserted
    greedily
    - paths of pk/fk joins are still inserted greedily, but their subtree is iterated using a BFS based on the base
    table estimates
    """

    def __init__(self, *, base_table_estimation: Optional[card_policy.BaseTableCardinalityEstimator] = None,
                 join_estimation: Optional[card_policy.JoinBoundCardinalityEstimator] = None,
                 subquery_policy: Optional[tree_policy.BranchGenerationPolicy] = None,
                 stats_container: Optional[StatisticsContainer] = None,
                 database: Optional[db.Database] = None, verbose: bool = False) -> None:
        super().__init__()
        self.database = database if database else db.DatabasePool().get_instance().current_database()
        self.base_table_estimation = (base_table_estimation if base_table_estimation
                                      else card_policy.NativeCardinalityEstimator(self.database))
        self.join_estimation = join_estimation if join_estimation else UESJoinBoundEstimator()
        self.subquery_policy = subquery_policy if subquery_policy else UESSubqueryGenerationPolicy()
        self.stats_container = (stats_container if stats_container
                                else MaxFrequencyStatsContainer(self.database.statistics()))
        self._logging_enabled = verbose

    def optimize_join_order(self, query: qal.SqlQuery
                            ) -> Optional[jointree.LogicalJoinTree]:
        if not isinstance(query, qal.ImplicitSqlQuery):
            raise ValueError("UES optimization only works for implicit queries for now")
        if len(query.tables()) < 2:
            return None

        self.base_table_estimation.setup_for_query(query)
        self.stats_container.setup_for_query(query, self.base_table_estimation)
        self.join_estimation.setup_for_query(query)
        self.join_estimation.setup_for_stats(self.stats_container)
        self.subquery_policy.setup_for_query(query)
        self.subquery_policy.setup_for_stats_container(self.stats_container)

        join_graph = joingraph.JoinGraph(query, self.database.schema())

        if len(query.tables()) == 2:
            final_join_tree = self._binary_join_optimization(query, join_graph)
        elif join_graph.contains_cross_products():
            # cross-product query is reduced to multiple independent optimization passes
            optimized_components = []
            for component in join_graph.join_components():
                # FIXME: join components might consist of single tables!
                optimized_component = self._clone().optimize_join_order(component.query)
                if not optimized_component:
                    raise stages.JoinOrderOptimizationError(component.query)
                optimized_components.append(optimized_component)

            # insert cross-products such that the smaller partitions are joined first
            sorted(optimized_components, key=operator.attrgetter("upper_bound"))
            merger = jointree.logical_join_tree_annotation_merger
            final_join_tree = jointree.LogicalJoinTree.cross_product_of(*optimized_components,
                                                                        annotation_supplier=merger)
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

    def _default_ues_optimizer(self, query: qal.SqlQuery, join_graph: joingraph.JoinGraph) -> jointree.LogicalJoinTree:
        """Implementation of our take on the UES algorithm for n:m joined queries."""
        join_tree = jointree.LogicalJoinTree()

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
                filter_pred = query.predicates().filters_for(lowest_bound_table)
                annotation = jointree.LogicalBaseTableMetadata(filter_pred, lowest_bound)
                join_tree = jointree.LogicalJoinTree.for_base_table(lowest_bound_table, annotation)
                join_graph.mark_joined(lowest_bound_table)
                self.stats_container.upper_bounds[join_tree] = lowest_bound
                pk_joins = join_graph.available_deep_pk_join_paths_for(lowest_bound_table,
                                                                       self._table_base_cardinality_ordering)
                for pk_join in pk_joins:
                    target_table = pk_join.target_table
                    base_cardinality = self.stats_container.base_table_estimates[target_table]
                    filter_pred = query.predicates().filters_for(target_table)
                    join_bound = self.join_estimation.estimate_for(pk_join.join_condition, join_graph)
                    join_graph.mark_joined(target_table, pk_join.join_condition)
                    base_annotation = jointree.LogicalBaseTableMetadata(filter_pred, base_cardinality)
                    join_annotation = jointree.LogicalJoinMetadata(pk_join.join_condition, join_bound)
                    join_tree = join_tree.join_with_base_table(pk_join.target_table, base_annotation, join_annotation)
                self._log_optimization_progress("Initial table selection", lowest_bound_table, pk_joins)
                continue

            selected_candidate: joingraph.JoinPath | None = None
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
            candidate_filters = query.predicates().filters_for(candidate_table)
            candidate_base_cardinality = self.stats_container.base_table_estimates[candidate_table]
            join_annotation = jointree.LogicalJoinMetadata(selected_candidate.join_condition, lowest_bound)
            base_annotation = jointree.LogicalBaseTableMetadata(candidate_filters, candidate_base_cardinality)
            self._log_optimization_progress("n:m join", candidate_table, all_pk_joins,
                                            join_condition=selected_candidate.join_condition,
                                            subquery_join=create_subquery)
            if create_subquery:
                subquery_tree = jointree.LogicalJoinTree.for_base_table(candidate_table, base_annotation)
                join_graph.mark_joined(candidate_table)
                subquery_tree = self._insert_pk_joins(query, all_pk_joins, subquery_tree, join_graph)

                join_tree = join_tree.join_with_subquery(subquery_tree, join_annotation)
                self.stats_container.upper_bounds[join_tree] = lowest_bound
            else:

                join_tree = join_tree.join_with_base_table(candidate_table, base_annotation, join_annotation)
                join_graph.mark_joined(candidate_table, selected_candidate.join_condition)
                self.stats_container.upper_bounds[join_tree] = lowest_bound
                join_tree = self._insert_pk_joins(query, all_pk_joins, join_tree, join_graph)

            self.stats_container.trigger_frequency_update(join_tree, candidate_table,
                                                          selected_candidate.join_condition)

        if join_graph.contains_free_tables():
            raise AssertionError("Join graph still has free tables remaining!")
        return join_tree

    def _binary_join_optimization(self, query: qal.ImplicitSqlQuery,
                                  join_graph: joingraph.JoinGraph) -> jointree.LogicalJoinTree:
        """Simplified "optimization" for a query with a single join. Makes the smaller table the outer relation."""
        table1, table2 = query.tables()
        table1_smaller = self.stats_container.base_table_estimates[table1] < self.stats_container.base_table_estimates[
            table2]
        small_table, large_table = (table1, table2) if table1_smaller else (table2, table1)

        large_card = self.stats_container.base_table_estimates[large_table]
        small_card = self.stats_container.base_table_estimates[small_table]

        large_filter = query.predicates().filters_for(large_table)
        small_filter = query.predicates().filters_for(small_table)

        join_predicate = query.predicates().joins_between(large_table, small_table)
        join_bound = self.join_estimation.estimate_for(join_predicate, join_graph)

        base_annotation = jointree.LogicalBaseTableMetadata(large_filter, large_card)
        join_tree = jointree.LogicalJoinTree.for_base_table(large_table, base_annotation)
        partner_annotation = jointree.LogicalBaseTableMetadata(small_filter, small_card)
        join_annotation = jointree.LogicalJoinMetadata(join_predicate, join_bound)
        join_tree = join_tree.join_with_base_table(small_table, partner_annotation, join_annotation)
        return join_tree

    def _star_query_optimizer(self, query: qal.ImplicitSqlQuery,
                              join_graph: joingraph.JoinGraph) -> jointree.LogicalJoinTree:
        """UES-inspired algorithm for star queries (i.e. queries with pk/fk joins only)"""
        # initial table / join selection
        lowest_bound = np.inf
        lowest_bound_join = None
        for candidate_join in join_graph.available_join_paths():
            current_bound = self.join_estimation.estimate_for(candidate_join.join_condition, join_graph)
            if current_bound < lowest_bound:
                lowest_bound = current_bound
                lowest_bound_join = candidate_join

        start_table = lowest_bound_join.start_table
        start_filters = query.predicates().filters_for(start_table)
        start_annotation = jointree.LogicalBaseTableMetadata(start_filters,
                                                             self.stats_container.base_table_estimates[start_table])
        join_tree = jointree.LogicalJoinTree.for_base_table(start_table, start_annotation)
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

            join_tree = self._apply_pk_fk_join(query, lowest_bound_join, join_bound=lowest_bound,
                                               join_graph=join_graph, current_join_tree=join_tree)

        return join_tree

    def _table_base_cardinality_ordering(self, table: base.TableReference, join_edge: dict) -> int:
        """Utility method to enable an ordering of base tables according to their base table estimates."""
        return self.stats_container.base_table_estimates[table]

    def _apply_pk_fk_join(self, query: qal.SqlQuery, pk_fk_join: joingraph.JoinPath, *,
                          join_bound: int, join_graph: joingraph.JoinGraph,
                          current_join_tree: jointree.LogicalJoinTree) -> jointree.LogicalJoinTree:
        """Includes the given  pk/fk join into the join tree, taking care of all necessary updates."""
        target_table = pk_fk_join.target_table
        target_filters = query.predicates().filters_for(target_table)
        target_cardinality = self.stats_container.base_table_estimates[target_table]
        base_annotation = jointree.LogicalBaseTableMetadata(target_filters, target_cardinality)
        join_annotation = jointree.LogicalJoinMetadata(pk_fk_join.join_condition, join_bound)
        updated_join_tree = current_join_tree.join_with_base_table(target_table, base_annotation, join_annotation)
        join_graph.mark_joined(target_table, pk_fk_join.join_condition)
        self.stats_container.upper_bounds[updated_join_tree] = join_bound
        return updated_join_tree

    def _insert_pk_joins(self, query: qal.SqlQuery, pk_joins: Iterable[joingraph.JoinPath],
                         join_tree: jointree.LogicalJoinTree,
                         join_graph: joingraph.JoinGraph) -> jointree.LogicalJoinTree:
        """Generalization of `_apply_pk_fk_join` to multiple join paths."""
        # TODO: refactor in terms of _apply_pk_fk_join
        for pk_join in pk_joins:
            pk_table = pk_join.target_table
            if not join_graph.is_free_table(pk_table):
                continue
            pk_filters = query.predicates().filters_for(pk_table)
            pk_join_bound = self.join_estimation.estimate_for(pk_join.join_condition, join_graph)
            pk_base_cardinality = self.stats_container.base_table_estimates[pk_table]
            base_annotation = jointree.LogicalBaseTableMetadata(pk_filters, pk_base_cardinality)
            join_annotation = jointree.LogicalJoinMetadata(pk_join.join_condition, pk_join_bound)
            join_tree = join_tree.join_with_base_table(pk_table, base_annotation, join_annotation)
            join_graph.mark_joined(pk_table, pk_join.join_condition)
            self.stats_container.upper_bounds[join_tree] = pk_join_bound
        return join_tree

    def _clone(self) -> UESJoinOrderOptimizer:
        """Creates a new join order optimizer with the same settings as this one."""
        return UESJoinOrderOptimizer(base_table_estimation=copy.copy(self.base_table_estimation),
                                     join_estimation=copy.copy(self.join_estimation),
                                     subquery_policy=copy.copy(self.subquery_policy),
                                     stats_container=copy.copy(self.stats_container),
                                     database=self.database)

    def _log_optimization_progress(self, phase: str, candidate_table: base.TableReference,
                                   pk_joins: Iterable[joingraph.JoinPath], *,
                                   join_condition: predicates.AbstractPredicate | None = None,
                                   subquery_join: bool | None = None) -> None:
        """Logs the current optimization state."""
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


class UESOperatorSelection(stages.PhysicalOperatorSelection):
    """Implementation of the operator selection for the UES algorithm.

    UES is actually not concerned with operator selection and focuses exclusively on join orders. Therefore, this
    strategy simply disables nested loop joins since they can typically lead to performance degradation. Essentially
    this enforces the usage of hash joins for the vast majority of joins in a typical database system since they
    provide the most robust behavior.

    See Hertzschuch et al.: "Simplicity Done Right for Join Ordering", CIDR'2021 for more details.
    """

    def __init__(self, database: db.Database) -> None:
        super().__init__()
        self.database = database

    def _apply_selection(self, query: qal.SqlQuery,
                         join_order: Optional[jointree.LogicalJoinTree | jointree.PhysicalQueryPlan]
                         ) -> physops.PhysicalOperatorAssignment:
        if isinstance(join_order, jointree.PhysicalQueryPlan):
            assignment = join_order.physical_operators().clone()
        else:
            assignment = physops.PhysicalOperatorAssignment()

        if self.database.hinting().supports_hint(physops.JoinOperators.NestedLoopJoin):
            assignment.set_operator_enabled_globally(physops.JoinOperators.NestedLoopJoin, False,
                                                     overwrite_fine_grained_selection=True)
        return assignment

    def _description(self) -> dict:
        return {"name": "ues"}
