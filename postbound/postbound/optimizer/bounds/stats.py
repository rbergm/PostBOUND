"""Provides data structures to store statistics for cardinality estimates and upper bounds.

Note that the wording for most of the code assumes the usage of upper bounds. However, normal cardinality estimates
can be stored just as well and the implementation is not actually limited to upper bounds. The wording is due to
PostBOUND's primary focus on the implementation and evaluation of defensive (i.e. mostly upper bound-based)
optimization techniques. Therefore, upper bounds and cardinality estimates can be treated as interchangeable terms in
this module.

The data structures themselves are generic and should work for a wide variety of statistics, e.g. single scalar values,
top-k lists or histograms should work just as well as entirely custom types.
"""
from __future__ import annotations

import abc
import typing
from typing import Generic

from postbound.db import db
from postbound.qal import base, qal, predicates
from postbound.optimizer import jointree
from postbound.optimizer.bounds import scans
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
        self.base_table_estimator: scans.BaseTableCardinalityEstimator | None = None

    def setup_for_query(self, query: qal.SqlQuery, base_table_estimator: scans.BaseTableCardinalityEstimator) -> None:
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
