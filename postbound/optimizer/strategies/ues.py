"""Generalized implementation of the UES join order optimization algorihtm [1]_.

Our implementation differs from the original algorithm in a number of ways, most importantly by making policies more explicit
and enabling their variation. More specifically, we enable variation of the following parts of the algorithm:

- estimation strategy for the cardinality of (filtered) base tables
- estimation strategy for the cardinality of joins, thereby enabling the usage of different statistics. For example, this
  enables top-k list statistics instead of only using the maximum value frequency
- deciding when to generate subqueries for primary key/foreign key joins

Additionally, our implementation has a stricter treatment of chains of primary key/foreign key joins. Consider a join of the
form A ⋈ B ⋈ C. Here, A ⋈ B is primary key/foreign key join with A acting as the foreign key partner and B acting as the
primary key partner. At the same time, B ⋈ C is also a primary key/foreign key join, but this time B acts as the foreign key
partner and C is the primary key partner. The original implementation did not specify how such situations should be handled and
multiple possible approaches exist (e.g. treating the entire join sequence as one large primary key/foreign key join or
invalidating the second join once the primary key/foreign key join between A and B has been performed). Our implementation can
use the first strategy (the join is treated as one large primary key/foreign key join and the subquery contains all the
related tables) but defaults to the second one.

References
----------

.. [1] A. Hertzschuch et al.: "Simplicity Done Right for Join Ordering", CIDR'2021
"""

from __future__ import annotations

import abc
import copy
import math
import operator
import typing
from collections.abc import Iterable
from typing import Generic, Optional

import numpy as np

from ... import util
from ..._core import Cardinality, ColumnReference, JoinOperator, TableReference
from ..._hints import PhysicalOperatorAssignment
from ..._jointree import JoinTree, LogicalJoinTree
from ..._stages import (
    JoinOrderOptimization,
    JoinOrderOptimizationError,
    OptimizationPreCheck,
    PhysicalOperatorSelection,
)
from ..._validation import (
    DependentSubqueryPreCheck,
    EquiJoinPreCheck,
    ImplicitQueryPreCheck,
    SetOperationsPreCheck,
    VirtualTablesPreCheck,
    merge_checks,
)
from ...db._db import Database, DatabasePool, DatabaseStatistics
from ...qal._qal import AbstractPredicate, ImplicitSqlQuery, SqlQuery
from .._joingraph import JoinGraph, JoinPath
from ..policies import cardinalities as cardpol
from ..policies import jointree as treepol

ColumnType = typing.TypeVar("ColumnType")
"""The type of the columns for which statistics are generated."""

StatsType = typing.TypeVar("StatsType")
"""The type of the actual statistics that are stored, e.g. single values or frequency lists."""

MaxFrequency = typing.NewType("MaxFrequency", int)
"""Type alias for maximum frequency statistics of columns (which are just integer values).

The maximum frequency of a column is the maximum number of occurrences of a column value within that column.

For example, consider a column R.a with values ``[a, b, a, a, b, c]``. In this case, maximum column frequency for R.a is 3.
"""

MostCommonElements = typing.NewType("MostCommonElements", list[tuple[ColumnType, int]])
"""Type alias for top-k lists statistics. The top-k list is generic over the actual column type."""


class StatisticsContainer(abc.ABC, Generic[StatsType]):
    """The statistics container eases the management of the statistics lifecycle.

    It provides means to store different kinds of statistics as attributes and can take care of their update automatically.
    Each statistics container instance is intended for one specific query and has to be initialized for that query using the
    `setup_for_query` method.

    A statistics container is abstract to enable a tailored implementation of the loading and updating procedures for
    different statistics types.

    Attributes
    ----------
    base_table_estimates : dict[TableReference, int]
        These statistics are intended for tables that are not part of the intermediate result, yet. The estimates approximate
        the number of rows that are returned when scanning the table.
    upper_bounds : dict[TableReference | jointree.LogicalJoinTree, int]
        These statistics contain the cardinality estimates for intermediate results of the input query. Inserting new bounds
        can result in an update of the column statistics.
    attribute_frequencies : dict[ColumnReference, StatsType]
        This statistic contains the current statistics value for individual columns. This is the main data structure that has
        to be maintained during the query optimization process to update the column statistics once they become part of an
        intermediate result (and get changed as part of the join process).
    query : Optional[SqlQuery]
        Stores the query that this container is created for
    """

    def __init__(self) -> None:
        self.base_table_estimates: dict[TableReference, int] = {}
        self.upper_bounds: dict[TableReference | LogicalJoinTree, int] = {}
        self.attribute_frequencies: dict[ColumnReference, StatsType] = {}
        self.query: Optional[SqlQuery] = None

    def setup_for_query(
        self,
        query: SqlQuery,
        base_table_estimator: cardpol.BaseTableCardinalityEstimator,
    ) -> None:
        """Initializes the internal data of the statistics container for a specific query.

        Parameters
        ----------
        query : SqlQuery
            The query that
        base_table_estimator : card_policy.BaseTableCardinalityEstimator
            Estimator to inflate the `base_table_estimates` for all tables that are contained in the query. The estimator has
            to set-up properly.
        """
        self._reset_containers()
        self.query = query
        self._inflate_base_table_estimates(base_table_estimator)
        self._inflate_attribute_frequencies()

    def join_bounds(self) -> dict[LogicalJoinTree, int]:
        """Provides the cardinality estimates of all join trees that are currently stored in the container.

        Returns
        -------
        dict[jointree.LogicalJoinTree, int]
            The bounds for all intermediate results
        """
        return {
            join_tree: bound
            for join_tree, bound in self.upper_bounds.items()
            if isinstance(join_tree, JoinTree)
        }

    def trigger_frequency_update(
        self,
        join_tree: LogicalJoinTree,
        joined_table: TableReference,
        join_condition: AbstractPredicate,
    ) -> None:
        """Updates the `attribute_frequencies` according to a new n:m join.

        The update procedure distinguishes between two different types of column statistics and uses different
        (and statistics-dependent) update methods for each: partner columns and third-party columns.

        Partner columns are those columns from the intermediate query result, that are directly involved in the join
        predicate, i.e. they are a join partner for some column of the newly joined table. On the other hand, third
        party columns are part of the intermediate result, but not directly involved in the join. In order to update
        them, some sort of correlation info is usually required.

        The precise update semantics depend on the specific statistic type. Hence, the updates are performed via abstract
        methods.

        Parameters
        ----------
        join_tree : jointree.LogicalJoinTree
            A join order that indicates the last join that was performed. This is the join that is used to infer the necessary
            updates.
        joined_table : TableReference
            The actual table that was joined. Remember that UES performs either primary key/foreign key joins, or joins with
            exactly one n:m table join partner. In the first case, no frequency updates are necessary since cardinalities may
            never increase when the foreign key is already part of an intermediate result. In the second case, there is exactly
            one partner table that is denoted by this parameter.
        join_condition : AbstractPredicate
            The predicate that was used for the join. This is required to determine the columns that were directly involved in
            the join. These columns have to be updated in a different way compared to other columns in the intermediate result.
        """
        partner_tables = join_tree.tables() - {joined_table}

        third_party_columns: set[ColumnReference] = set()
        for third_party_table in partner_tables:
            potential_partners = partner_tables - {third_party_table}
            join_pred = self.query.predicates().joins_between(
                third_party_table, potential_partners
            )
            if not join_pred:
                continue
            third_party_columns |= join_pred.columns()

        for col1, col2 in join_condition.join_partners():
            joined_column, partner_column = (
                (col1, col2) if col1.table == joined_table else (col2, col1)
            )
            self._update_partner_column_frequency(joined_column, partner_column)

        joined_columns_frequencies = {
            joined_col: self.attribute_frequencies[joined_col]
            for joined_col in join_condition.columns_of(joined_table)
        }
        lowest_joined_column_frequency = util.argmin(joined_columns_frequencies)
        for third_party_column in third_party_columns:
            self._update_third_party_column_frequency(
                lowest_joined_column_frequency, third_party_column
            )

    @abc.abstractmethod
    def describe(self) -> dict:
        """Generates a JSON-serializable description of the specific container, including the actual statistics type.

        Returns
        -------
        dict
            The description
        """
        raise NotImplementedError

    def _reset_containers(self) -> None:
        """Drops all currently stored statistics. This is a necessary preparation step when a new input query is encoutered."""
        self.base_table_estimates = {}
        self.upper_bounds = {}
        self.attribute_frequencies = {}
        self.query = None

    def _inflate_base_table_estimates(
        self, base_table_estimator: cardpol.BaseTableCardinalityEstimator
    ):
        """Retrieves the base table estimate for each table in the current query.

        Parameters
        ----------
        base_table_estimator : card_policy.BaseTableCardinalityEstimator
            The strategy that should be used to obtain the estimate.
        """
        for table in self.query.tables():
            table_estimate = base_table_estimator.estimate_for(table)
            self.base_table_estimates[table] = table_estimate

    @abc.abstractmethod
    def _inflate_attribute_frequencies(self):
        """Loads the attribute frequency statistics for all required columns.

        The precise statistics that have to be loaded, as well as the columns that require loading of statistics is completely
        up to the specific statistics container.
        """
        raise NotImplementedError

    @abc.abstractmethod
    def _update_partner_column_frequency(
        self, joined_column: ColumnReference, partner_column: ColumnReference
    ) -> None:
        """Performs the frequency update for a partner column.

        This implies that there is a join between the joined column and the partner column, and the partner column is already
        part of the intermediate result. Likewise, the joined column has just become part of the intermediate result as of this
        join.

        Parameters
        ----------
        joined_column : ColumnReference
            A column that is already part of the intermediate result
        partner_column : ColumnReference
            A column of the relation that has just been joined with the current intermedaite result
        """
        raise NotImplementedError

    @abc.abstractmethod
    def _update_third_party_column_frequency(
        self, joined_column: ColumnReference, third_party_column: ColumnReference
    ) -> None:
        """Performs the frequency update for a third party column (see `trigger_frequency_update`).

        This implies that there is a join between the joined column and some other column from the intermediate result. The
        third party columns is already part of the intermediate result, but not directly involved in the join. The joined
        column has just become part of the intermediate result as of this join.

        Parameters
        ----------
        joined_column : ColumnReference
            A column that is already part of the intermediate result
        third_party_column : ColumnReference
            A column of the relation that has just been joined with the current intermediate result
        """
        raise NotImplementedError


class MaxFrequencyStatsContainer(StatisticsContainer[MaxFrequency]):
    """Statistics container that stores the maximum frequency of the join columns.

    The frequency updates happen pessimistically, which means that each column frequency of the intermediate result is
    multiplied by the maximum frequency of the partner column. This ensures that no under-estimation is possible, but
    over-estimates the true frequencies by a very large margin. However, in order to close this gap, correlation information is
    required.

    See Also
    --------
    MaxFrequency
    """

    def __init__(self, database_stats: DatabaseStatistics):
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
            if not top1_list:
                mcv_frequency = self._uniform_frequency(column)
            else:
                _, mcv_frequency = util.simplify(top1_list)
            self.attribute_frequencies[column] = mcv_frequency

    def _update_partner_column_frequency(
        self, joined_column: ColumnReference, partner_column: ColumnReference
    ) -> None:
        joined_frequency = self.attribute_frequencies[joined_column]
        partner_frequency = self.attribute_frequencies[partner_column]
        self.attribute_frequencies[joined_column] *= partner_frequency
        self.attribute_frequencies[partner_column] *= joined_frequency

    def _update_third_party_column_frequency(
        self, joined_column: ColumnReference, third_party_column: ColumnReference
    ) -> None:
        self.attribute_frequencies[third_party_column] *= self.attribute_frequencies[
            joined_column
        ]

    def _uniform_frequency(self, column: ColumnReference) -> float:
        """Calculates the value frequency for a column, assuming that all values are uniformly distributed.

        Parameters
        ----------
        column : ColumnReference
            The column to calculate for

        Returns
        -------
        float
            The estimated frequency of all column values
        """
        n_tuples = self.database_stats.total_rows(column.table)
        n_tuples = 1 if n_tuples is None else n_tuples
        n_distinct = self.database_stats.distinct_values(column)
        n_distinct = 1 if n_distinct is None else n_distinct
        return n_tuples / n_distinct


class UESJoinBoundEstimator(cardpol.JoinCardinalityEstimator):
    """Implementation of the UES formula to calculate upper bounds of join cardinalities.

    The formula distinuishes two cases: n:m joins are estimated according to the maximum frequencies of the join columns.
    Primary key/foreign key joins are estimated according to the cardinality of the foreign key column. The calculation also
    accounts for conjunctive join predicates, but is still limited to equi joins.
    """

    def __init__(self) -> None:
        super().__init__("UES join estimator")
        self.query: ImplicitSqlQuery | None = None
        self.stats_container: StatisticsContainer[MaxFrequency] | None = None

    def setup_for_query(self, query: SqlQuery) -> None:
        self.query = query

    def setup_for_stats(
        self, stats_container: StatisticsContainer[MaxFrequency]
    ) -> None:
        """Configures the statistics container that contains the actual frequencies and cardinalities to use.

        Parameters
        ----------
        stats_container : StatisticsContainer[MaxFrequency]
            The statistics to use
        """
        self.stats_container = stats_container

    def estimate_for(
        self, join_edge: AbstractPredicate, join_graph: JoinGraph
    ) -> Cardinality:
        current_min_bound = np.inf

        for base_predicate in join_edge.base_predicates():
            first_col, second_col = util.simplify(base_predicate.join_partners())
            if join_graph.is_pk_fk_join(first_col.table, second_col.table):
                join_bound = self._estimate_pk_fk_join(first_col, second_col)
            elif join_graph.is_pk_fk_join(second_col.table, first_col.table):
                join_bound = self._estimate_pk_fk_join(second_col, first_col)
            else:
                join_bound = self._estimate_n_m_join(first_col, second_col)

            if join_bound < current_min_bound:
                current_min_bound = join_bound

        return Cardinality(current_min_bound)

    def describe(self) -> dict:
        return {"name": "ues"}

    def pre_check(self) -> Optional[OptimizationPreCheck]:
        # TODO: the UES check is slightly too restrictive here.
        # It suffices to check that there are only conjunctive equi joins.
        return UESOptimizationPreCheck  # this is a pre-generated check instance, don't call () it here!

    def _estimate_pk_fk_join(
        self, fk_column: ColumnReference, pk_column: ColumnReference
    ) -> Cardinality:
        """Estimation formula for primary key/foreign key joins.

        Parameters
        ----------
        fk_column : ColumnReference
            The foreign key column
        pk_column : ColumnReference
            The primary key column

        Returns
        -------
        Cardinality
            An upper bound of the primary key/foreign key join cardinality.
        """
        pk_cardinality = Cardinality(
            self.stats_container.base_table_estimates[pk_column.table]
        )
        fk_frequency = self.stats_container.attribute_frequencies[fk_column]
        return math.ceil(fk_frequency * pk_cardinality)

    def _estimate_n_m_join(
        self, first_column: ColumnReference, second_column: ColumnReference
    ) -> Cardinality:
        """Estimation formula for n:m joins.

        Parameters
        ----------
        first_column : ColumnReference
            The join column from the first partner
        second_column : ColumnReference
            The join column from the second partner

        Returns
        -------
        Cardinality
            An upper bound of the n:m join cardinality.
        """
        first_bound, second_bound = (
            self._fetch_bound(first_column),
            self._fetch_bound(second_column),
        )
        first_freq = self.stats_container.attribute_frequencies[first_column]
        second_freq = self.stats_container.attribute_frequencies[second_column]

        if any(
            var == 0 for var in [first_bound, second_bound, first_freq, second_freq]
        ):
            return 0

        first_distinct_vals = first_bound / first_freq
        second_distinct_vals = second_bound / second_freq

        n_m_bound = (
            min(first_distinct_vals, second_distinct_vals) * first_freq * second_freq
        )
        return Cardinality(math.ceil(n_m_bound))

    def _fetch_bound(self, column: ColumnReference) -> Cardinality:
        """Provides the appropriate table bound (based on upper bound or base table estimate) for the given column.

        This is a utility method to work with the statistics container in a more convenient way, since the container can store
        the table cardinality at two different places: as a base table estimate, or as a intermediate estimate for base tables
        that can be filtered via a primary key/foreign key join.

        Parameters
        ----------
        column : ColumnReference
            The column for which the upper bound of the corresponding base table should be loaded.

        Returns
        -------
        Cardinality
            An upper bound on the cardinality of the table
        """
        table = column.table
        card = (
            self.stats_container.upper_bounds[table]
            if table in self.stats_container.upper_bounds
            else self.stats_container.base_table_estimates[table]
        )
        return Cardinality(card)


class UESSubqueryGenerationPolicy(treepol.BranchGenerationPolicy):
    """Implementation of the UES policy to decide when to insert branches into the join order.

    In short, the policy generates subqueries whenever they guarantee a reduction of the upper bound of the n:m join partner
    table.
    """

    def __init__(self):
        super().__init__("UES subquery policy")
        self.query: SqlQuery | None = None
        self.stats_container: StatisticsContainer | None = None

    def setup_for_query(self, query: SqlQuery) -> None:
        self.query = query

    def setup_for_stats_container(self, stats_container: StatisticsContainer) -> None:
        """Configures the statistics container that contains the actual frequencies and bounds to use.

        Parameters
        ----------
        stats_container : StatisticsContainer[MaxFrequency]
            The statistics to use
        """
        self.stats_container = stats_container

    def generate_subquery_for(
        self, join: AbstractPredicate, join_graph: JoinGraph
    ) -> bool:
        if join_graph.count_consumed_tables() < 2:
            return False

        stats_container = self.stats_container
        for first_col, second_col in join.join_partners():
            first_tab, second_tab = first_col.table, second_col.table
            if join_graph.is_pk_fk_join(first_tab, second_tab):
                joined_table = first_tab
            elif join_graph.is_pk_fk_join(second_tab, first_tab):
                joined_table = second_tab
            else:
                continue

            generate_subquery = (
                stats_container.upper_bounds[joined_table]
                < stats_container.base_table_estimates[joined_table]
            )
            if generate_subquery:
                return True

        return False

    def describe(self) -> dict:
        return {"name": "defensive"}


class UESJoinOrderOptimizer(JoinOrderOptimization):
    """Implementation of the UES join order algorithm.

    Our implementation expands upon the original algorithm in a number of ways. These are used to enable a variation of
    different policies during optimization, and to apply the algorithm to a larger set of queries. See the module documentation
    for details.

    Parameters
    ----------
    base_table_estimation : Optional[card_policy.BaseTableCardinalityEstimator], optional
        A strategy to estimate the cardinalities/bounds of (filtered) base tables. Defaults to a native estimation by the
        optimizer of the `database`.
    join_estimation : Optional[card_policy.JoinBoundCardinalityEstimator], optional
        A strategy to estimate the upper bounds of intermediate joins. Defaults to the `UESJoinBoundEstimator`.
    subquery_policy : Optional[tree_policy.BranchGenerationPolicy], optional
        A strategy to determine when to insert subqueries into the resulting join tree. Defaults to the
        `UESSubqueryGenerationPolicy`.
    stats_container : Optional[StatisticsContainer], optional
        The statistics used to calcualte the different upper bounds. These have to be compatible with the `join_estimation`.
        Defaults to a `MaxFrequencyStatsContainer`.
    pull_eager_pk_tables : bool, optional
        How to deal with chains of primary key/foreign key joins (joins where the primary key table for one join acts as the
        foreign key in another join). By default, only the first primary key/foreign key join is used as a filter. If eager
        pulling is enabled, the subsequent primary key filters are also included.
    database : Optional[db.Database], optional
        The database whose statistics should be used. The database has to be configured appropriately already (e.g. regarding
        the usage of emulated statistics). If this parameter is omitted, it is inferred from the `db.DatabasePool`.
    verbose : bool, optional
        Whether to log internal progress and bound statistics. This is off by default.

    References
    ----------

    .. A. Hertzschuch et al.: "Simplicity Done Right for Join Ordering", CIDR'2021
    """

    def __init__(
        self,
        *,
        base_table_estimation: Optional[cardpol.BaseTableCardinalityEstimator] = None,
        join_estimation: Optional[cardpol.JoinCardinalityEstimator] = None,
        subquery_policy: Optional[treepol.BranchGenerationPolicy] = None,
        stats_container: Optional[StatisticsContainer] = None,
        pull_eager_pk_tables: bool = False,
        database: Optional[Database] = None,
        verbose: bool = False,
    ) -> None:
        super().__init__()
        self.database = (
            database if database else DatabasePool().get_instance().current_database()
        )
        self.base_table_estimation = (
            base_table_estimation
            if base_table_estimation
            else cardpol.NativeCardinalityEstimator(self.database)
        )
        self.join_estimation = (
            join_estimation if join_estimation else UESJoinBoundEstimator()
        )
        self.subquery_policy = (
            subquery_policy if subquery_policy else UESSubqueryGenerationPolicy()
        )
        self.stats_container = (
            stats_container
            if stats_container
            else MaxFrequencyStatsContainer(self.database.statistics())
        )
        self._pull_eager_pk_tables = pull_eager_pk_tables
        self._logging_enabled = verbose

    def optimize_join_order(self, query: SqlQuery) -> Optional[LogicalJoinTree]:
        if not isinstance(query, ImplicitSqlQuery):
            raise ValueError("UES optimization only works for implicit queries for now")
        if len(query.tables()) < 2:
            return None

        self.base_table_estimation.setup_for_query(query)
        self.stats_container.setup_for_query(query, self.base_table_estimation)
        self.join_estimation.setup_for_query(query)
        self.join_estimation.setup_for_stats(self.stats_container)
        self.subquery_policy.setup_for_query(query)
        if "setup_for_stats_container" in dir(self.subquery_policy):
            self.subquery_policy.setup_for_stats_container(self.stats_container)

        join_graph = JoinGraph(query, self.database.schema())

        if len(query.tables()) == 2:
            final_join_tree = self._binary_join_optimization(query, join_graph)
        elif join_graph.contains_cross_products():
            # cross-product query is reduced to multiple independent optimization passes
            optimized_components: list[LogicalJoinTree] = []
            for component in join_graph.join_components():
                # FIXME: join components might consist of single tables!
                optimized_component = self._clone().optimize_join_order(component.query)
                if not optimized_component:
                    raise JoinOrderOptimizationError(component.query)
                optimized_components.append(optimized_component)

            # insert cross-products such that the smaller partitions are joined first
            sorted(optimized_components, key=operator.attrgetter("annotation"))
            final_join_tree, *remaining_joins = optimized_components
            for remaining_join in remaining_joins:
                output_cardinality = (
                    final_join_tree.annotation * remaining_join.annotation
                )
                final_join_tree = final_join_tree.join_with(
                    remaining_join, annotation=output_cardinality
                )
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
                "statistics": self.stats_container.describe(),
            },
        }

    def pre_check(self) -> OptimizationPreCheck:
        specified_checks = [
            check
            for check in [
                self.base_table_estimation.pre_check(),
                self.join_estimation.pre_check(),
                self.subquery_policy.pre_check(),
            ]
            if check
        ]
        specified_checks.append(UESOptimizationPreCheck)
        return merge_checks(specified_checks)

    def _default_ues_optimizer(
        self, query: SqlQuery, join_graph: JoinGraph
    ) -> LogicalJoinTree:
        """Implementation of our take on the UES algorithm for queries with n:m joins.

        Parameters
        ----------
        query : SqlQuery
            The query to optimize.
        join_graph : JoinGraph
            The join graph of the input query. This structure is mutated during the algorithm.

        Returns
        -------
        LogicalJoinTree
            The resulting join tree

        Raises
        ------
        AssertionError
            If the iterative construction failed. This indicates a bug in the implementation of the algorithm, not a mistake by
            the user.
        """
        self._log_information("Using default UES optimizer")
        join_tree = LogicalJoinTree.empty()

        while join_graph.contains_free_n_m_joins():
            # Update the current upper bounds
            lowest_bound = np.inf
            lowest_bound_table = None
            for candidate_join in join_graph.available_n_m_join_paths(
                both_directions_on_initial=True
            ):
                candidate_table = candidate_join.target_table
                filter_estimate = self.stats_container.base_table_estimates[
                    candidate_table
                ]
                pk_fk_bounds = [
                    self.join_estimation.estimate_for(
                        join_path.join_condition, join_graph
                    )
                    for join_path in join_graph.available_pk_fk_joins_for(
                        candidate_table
                    )
                ]
                candidate_min_bound = min([filter_estimate] + pk_fk_bounds)
                self.stats_container.upper_bounds[candidate_table] = candidate_min_bound

                if candidate_min_bound < lowest_bound:
                    lowest_bound = candidate_min_bound
                    lowest_bound_table = candidate_table
            self._log_information(
                ".. Current bounds: "
                + util.dicts.stringify(self.stats_container.upper_bounds)
            )

            if join_tree.is_empty():
                join_tree = LogicalJoinTree.scan(
                    lowest_bound_table, annotation=lowest_bound
                )
                join_graph.mark_joined(lowest_bound_table)
                self.stats_container.upper_bounds[join_tree] = lowest_bound
                pk_joins = join_graph.available_deep_pk_join_paths_for(
                    lowest_bound_table, self._table_base_cardinality_ordering
                )
                for pk_join in pk_joins:
                    target_table = pk_join.target_table
                    base_cardinality = self.stats_container.base_table_estimates[
                        target_table
                    ]
                    join_bound = self.join_estimation.estimate_for(
                        pk_join.join_condition, join_graph
                    )
                    join_graph.mark_joined(target_table, pk_join.join_condition)
                    join_tree = join_tree.join_with(
                        pk_join.target_table,
                        annotation=join_bound,
                        partner_annotation=base_cardinality,
                    )
                self._log_optimization_progress(
                    "Initial table selection", lowest_bound_table, pk_joins
                )
                continue

            selected_candidate: JoinPath | None = None
            lowest_bound = np.inf
            bounds_log: dict[JoinPath, float] = {}
            for candidate_join in join_graph.available_n_m_join_paths():
                candidate_bound = self.join_estimation.estimate_for(
                    candidate_join.join_condition, join_graph
                )
                bounds_log[candidate_join] = candidate_bound
                if candidate_bound < lowest_bound:
                    selected_candidate = candidate_join
                    lowest_bound = candidate_bound
            self._log_information(f".. n:m join bounds: {bounds_log}")

            direct_pk_joins = join_graph.available_pk_fk_joins_for(
                selected_candidate.target_table
            )
            create_subquery = any(
                self.subquery_policy.generate_subquery_for(
                    pk_join.join_condition, join_graph
                )
                for pk_join in direct_pk_joins
            )
            candidate_table = selected_candidate.target_table
            all_pk_joins = (
                join_graph.available_deep_pk_join_paths_for(candidate_table)
                if self._pull_eager_pk_tables
                else join_graph.available_pk_fk_joins_for(candidate_table)
            )
            candidate_base_cardinality = self.stats_container.base_table_estimates[
                candidate_table
            ]
            self._log_optimization_progress(
                "n:m join",
                candidate_table,
                all_pk_joins,
                join_condition=selected_candidate.join_condition,
                subquery_join=create_subquery,
            )
            if create_subquery:
                subquery_tree = JoinTree.scan(
                    candidate_table, annotation=candidate_base_cardinality
                )
                join_graph.mark_joined(candidate_table)
                subquery_tree = self._insert_pk_joins(
                    query, all_pk_joins, subquery_tree, join_graph
                )

                join_tree = join_tree.join_with(subquery_tree, annotation=lowest_bound)
                self.stats_container.upper_bounds[join_tree] = lowest_bound

            else:
                join_tree = join_tree.join_with(
                    candidate_table,
                    annotation=lowest_bound,
                    partner_annotation=candidate_base_cardinality,
                )
                join_graph.mark_joined(
                    candidate_table, selected_candidate.join_condition
                )
                self.stats_container.upper_bounds[join_tree] = lowest_bound
                join_tree = self._insert_pk_joins(
                    query, all_pk_joins, join_tree, join_graph
                )

            self.stats_container.trigger_frequency_update(
                join_tree, candidate_table, selected_candidate.join_condition
            )

        if join_graph.contains_free_tables():
            raise AssertionError("Join graph still has free tables remaining!")
        return join_tree

    def _binary_join_optimization(
        self, query: ImplicitSqlQuery, join_graph: JoinGraph
    ) -> LogicalJoinTree:
        """Specialized optimization algorithm for queries with just a single join.

        The algorithm can still be meaningful to determine the inner and outer relation for the only join that has to be
        performed. Furthermore, this algorithm can be used for smaller partitions of queries with cross products.

        The algorithm is inspired by UES and uses the table with the smaller upper bound as the outer relation.

        Parameters
        ----------
        query : ImplicitSqlQuery
            The query to optimize
        join_graph : joingraph.JoinGraph
            The join graph of the query. This structure is mutated during the algorithm.

        Returns
        -------
        LogicalJoinTree
            The resulting join tree
        """
        table1, table2 = query.tables()
        table1_smaller = (
            self.stats_container.base_table_estimates[table1]
            < self.stats_container.base_table_estimates[table2]
        )
        small_table, large_table = (
            (table1, table2) if table1_smaller else (table2, table1)
        )

        large_card = self.stats_container.base_table_estimates[large_table]
        small_card = self.stats_container.base_table_estimates[small_table]

        join_predicate = query.predicates().joins_between(large_table, small_table)
        join_bound = self.join_estimation.estimate_for(join_predicate, join_graph)

        join_tree = LogicalJoinTree.scan(large_table, annotation=large_card)
        join_tree = join_tree.join_with(
            small_table, annotation=join_bound, partner_annotation=small_card
        )
        return join_tree

    def _star_query_optimizer(
        self, query: ImplicitSqlQuery, join_graph: JoinGraph
    ) -> LogicalJoinTree:
        """Join ordering algorithm for star queries (i.e. queries which only consist of primary key/foreign key joins).

        The algorithm is inspired by UES and always tries to insert the table next that guarantees the smallest upper bound.

        Parameters
        ----------
        query : ImplicitSqlQuery
            The query to optimize
        join_graph : JoinGraph
            The join graph of the input query. This structure is mutated during the algorithm.

        Returns
        -------
        LogicalJoinTree
            The resulting join tree
        """
        self._log_information("Using star query optimizer")
        # initial table / join selection
        lowest_bound = np.inf
        lowest_bound_join = None
        for candidate_join in join_graph.available_join_paths():
            current_bound = self.join_estimation.estimate_for(
                candidate_join.join_condition, join_graph
            )
            if current_bound < lowest_bound:
                lowest_bound = current_bound
                lowest_bound_join = candidate_join

        start_table = lowest_bound_join.start_table
        start_card = self.stats_container.base_table_estimates[start_table]
        join_tree = LogicalJoinTree.scan(start_table, annotation=start_card)
        join_graph.mark_joined(start_table)
        join_tree = self._apply_pk_fk_join(
            query,
            lowest_bound_join,
            join_bound=lowest_bound,
            join_graph=join_graph,
            current_join_tree=join_tree,
        )

        # join partner selection
        while join_graph.contains_free_tables():
            lowest_bound = np.inf
            lowest_bound_join = None
            for candidate_join in join_graph.available_join_paths():
                current_bound = self.join_estimation.estimate_for(
                    candidate_join.join_condition, join_graph
                )
                if current_bound < lowest_bound:
                    lowest_bound = current_bound
                    lowest_bound_join = candidate_join

            join_tree = self._apply_pk_fk_join(
                query,
                lowest_bound_join,
                join_bound=lowest_bound,
                join_graph=join_graph,
                current_join_tree=join_tree,
            )

        return join_tree

    def _table_base_cardinality_ordering(
        self, table: TableReference, join_edge: dict
    ) -> int:
        """Utility method to impose an ordering of multiple primary key tables for a foreign key join.

        The actual ordering sorts the primary key tables according to their upper bounds and is used internally by the join
        graph.

        Parameters
        ----------
        table : TableReference
            The table for which the cardinality should be retrieved.
        join_edge : dict
            The edge of the join graph that describes the current join. This is ignored by the calculation and only required
            to satisfy the interface required by the join graph.

        Returns
        -------
        int
            A order index based on the cardinality estimate of the table

        See Also
        --------
        joingraph.JoinGraph.available_deep_pk_join_paths_for
        """
        return self.stats_container.base_table_estimates[table]

    def _apply_pk_fk_join(
        self,
        query: SqlQuery,
        pk_fk_join: JoinPath,
        *,
        join_bound: int,
        join_graph: JoinGraph,
        current_join_tree: LogicalJoinTree,
    ) -> LogicalJoinTree:
        """Includes a specific pk/fk join into a join tree, taking care of all necessary updates.

        Parameters
        ----------
        query : SqlQuery
            The query that is being optimized
        pk_fk_join : JoinPath
            The actual join that should be performed
        join_bound : int
            The calculated upper bound of the join
        join_graph : JoinGraph
            The join graph of the query. This structure is mutated as part of the update
        current_join_tree : LogicalJoinTree
            The join order that has been determined so far

        Returns
        -------
        LogicalJoinTree
            An updated join tree that includes the given join as the last (i.e. top-most) join.
        """
        target_table = pk_fk_join.target_table
        target_cardinality = self.stats_container.base_table_estimates[target_table]
        updated_join_tree = current_join_tree.join_with(
            target_table, annotation=join_bound, partner_annotation=target_cardinality
        )
        join_graph.mark_joined(target_table, pk_fk_join.join_condition)
        self.stats_container.upper_bounds[updated_join_tree] = join_bound
        return updated_join_tree

    def _insert_pk_joins(
        self,
        query: SqlQuery,
        pk_joins: Iterable[JoinPath],
        join_tree: LogicalJoinTree,
        join_graph: JoinGraph,
    ) -> LogicalJoinTree:
        """Generalization of `_apply_pk_fk_join` to multiple join paths.

        Parameters
        ----------
        query : SqlQuery
            The query that is being optimized
        pk_joins : Iterable[joingraph.JoinPath]
            The joins that should be included in the join tree, in the order in which they are inserted
        join_tree : jointree.LogicalJoinTree
            The join order that has been determined so far
        join_graph : joingraph.JoinGraph
            The join graph of the query. This structure is mutated as part of the update

        Returns
        -------
        jointree.LogicalJoinTree
            An updated join tree that includes all of the join paths. Join paths that appear earlier in the iterable are
            inserted deeper within the tree.
        """
        # TODO: refactor in terms of _apply_pk_fk_join
        for pk_join in pk_joins:
            pk_table = pk_join.target_table
            if not join_graph.is_free_table(pk_table):
                continue
            pk_join_bound = self.join_estimation.estimate_for(
                pk_join.join_condition, join_graph
            )
            pk_base_cardinality = self.stats_container.base_table_estimates[pk_table]
            join_tree = join_tree.join_with(
                pk_table,
                annotation=pk_join_bound,
                partner_annotation=pk_base_cardinality,
            )
            join_graph.mark_joined(pk_table, pk_join.join_condition)
            self.stats_container.upper_bounds[join_tree] = pk_join_bound
        return join_tree

    def _clone(self) -> UESJoinOrderOptimizer:
        """Creates a new join order optimizer with the same settings as this one.

        Returns
        -------
        UESJoinOrderOptimizer
            The cloned optimizer
        """
        return UESJoinOrderOptimizer(
            base_table_estimation=copy.copy(self.base_table_estimation),
            join_estimation=copy.copy(self.join_estimation),
            subquery_policy=copy.copy(self.subquery_policy),
            stats_container=copy.copy(self.stats_container),
            database=self.database,
        )

    def _log_information(self, info: str) -> None:
        """Displays arbitrary information.

        The current implementation of this methods writes to *stdout* directly. If logging is disabled, no information is
        printed.

        Parameters
        ----------
        info : str
            The information to display
        """
        if self._logging_enabled:
            print(info)

    def _log_optimization_progress(
        self,
        phase: str,
        candidate_table: TableReference,
        pk_joins: Iterable[JoinPath],
        *,
        join_condition: AbstractPredicate | None = None,
        subquery_join: bool | None = None,
    ) -> None:
        """Displays the current optimizer state.

        The current implementation of this method writes to *stdout* directly. If logging is disabled, no information is
        printed.

        Parameters
        ----------
        phase : str
            The phase of the UES algorithm, e.g. initial table selection of n:m join execution
        candidate_table : TableReference
            The table that is considered as the next join partner
        pk_joins : Iterable[JoinPath]
            Primary key joins that should be applied to the candidate table
        join_condition : AbstractPredicate | None, optional
            The join condition that was used to find the candidate table. Can be ``None`` to omit this information, e.g. when
            it is not applicable for the current phase.
        subquery_join : bool | None, optional
            Whether the primary key tables should be joined before the actual n:m join. Can be ``None`` to omit this
            information, e.g. when it is not applicable for the current phase.
        """
        # TODO: use proper logging instead of print() calls
        if not self._logging_enabled:
            return
        log_components = [
            phase,
            "::",
            str(candidate_table),
            "with PK joins",
            str(pk_joins),
        ]
        if join_condition:
            log_components.extend(["on condition", str(join_condition)])
        if subquery_join is not None:
            log_components.append(
                "with subquery" if subquery_join else "without subquery"
            )
        log_message = " ".join(log_components)
        print(log_message)


class UESOperatorSelection(PhysicalOperatorSelection):
    """Implementation of the operator selection used in the UES algorithm.

    UES is actually not concerned with operator selection and focuses exclusively on join orders. Therefore, this
    strategy simply disables nested loop joins since they can typically lead to performance degradation. Essentially
    this enforces the usage of hash joins for the vast majority of joins in a typical database system because they
    provide the most robust behavior.

    Parameters
    ----------
    database : Database
        The target database on which the optimized query should be executed. This parameter enables a graceful fallback in case
        the database does not support a nested-loop join in the first place. If this situation occurs, nothing is disabled.

    Notes
    -----
    Although the UES join order optimizer never produces physical query plans and is only concerned with logical join trees,
    this selection algorithm handles physical plans gracefully by retaining all former operator assignments that do not
    contradict the no-nested-loop join rule.
    """

    def __init__(self, database: Database) -> None:
        super().__init__()
        self.database = database

    def select_physical_operators(
        self, query: SqlQuery, join_order: Optional[JoinTree]
    ) -> PhysicalOperatorAssignment:
        assignment = PhysicalOperatorAssignment()
        if self.database.hinting().supports_hint(JoinOperator.NestedLoopJoin):
            assignment.set_operator_enabled_globally(
                JoinOperator.NestedLoopJoin,
                False,
                overwrite_fine_grained_selection=True,
            )
        return assignment

    def describe(self) -> dict:
        return {"name": "ues"}


UESOptimizationPreCheck = merge_checks(
    ImplicitQueryPreCheck(),
    EquiJoinPreCheck(),
    DependentSubqueryPreCheck(),
    SetOperationsPreCheck(),
    VirtualTablesPreCheck(),
)
"""Check for all query features that UES does (not) support.

This check asserts that the following criteria are met:

- the input query is an *implicit* SQL query (see qal for details)
- all join predicates are binary equi joins
- there are no dependent subqueries
- there are no virtual tables, including no CTEs
"""
