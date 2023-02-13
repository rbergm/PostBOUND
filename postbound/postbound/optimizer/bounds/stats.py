from __future__ import annotations

import abc
import typing
import collections
from typing import Generic

from postbound.db import db
from postbound.qal import base, qal
from postbound.optimizer import data
from postbound.optimizer.bounds import scans
from postbound.util import dicts as dict_utils

ColumnType = typing.TypeVar("ColumnType")
StatsType = typing.TypeVar("StatsType")
MaxFrequency = typing.NewType("MaxFrequency", int)
MostCommonElements = typing.NewType("MostCommonElements", list[tuple[Generic[ColumnType], int]])


class UpperBoundsContainer(collections.UserDict[base.TableReference | data.JoinTree, int]):
    def __init__(self, stats_container: StatisticsContainer):
        super().__init__()
        self._stats_container = stats_container

    def __setitem__(self, key: base.TableReference | data.JoinTree, value: int) -> None:
        self.data[key] = value
        if isinstance(key, data.JoinTree) and key.root.is_join_node():
            self._stats_container.trigger_frequency_update(key)


class StatisticsContainer(abc.ABC, Generic[StatsType]):

    def __init__(self) -> None:
        self.base_table_estimates: dict[base.TableReference, int] = {}
        self.upper_bounds: UpperBoundsContainer = UpperBoundsContainer(self)
        self.attribute_frequencies: dict[base.ColumnReference, StatsType] = {}

        self.query: qal.SqlQuery | None = None
        self.base_table_estimator: scans.BaseTableCardinalityEstimator | None = None

    def setup_for_query(self, query: qal.SqlQuery, base_table_estimator: scans.BaseTableCardinalityEstimator) -> None:
        self.query = query
        self.base_table_estimator = base_table_estimator

        self._inflate_base_table_estimates()
        self._inflate_attribute_frequencies()

    def join_bounds(self) -> dict[data.JoinTree, int]:
        return {join_tree: bound for join_tree, bound in self.upper_bounds.items()
                if isinstance(join_tree, data.JoinTree)}

    def trigger_frequency_update(self, join_tree: data.JoinTree) -> None:
        if join_tree.is_empty():
            return

        root_node = join_tree.root
        if not root_node or not isinstance(root_node, data.JoinNode):
            raise ValueError(f"Expected join node, but was '{root_node}'")

        if not root_node.n_m_join:
            return

        joined_table = root_node.n_m_joined_table
        partner_columns = root_node.join_condition.join_partners_of(joined_table)
        third_party_columns = set(col for col in join_tree.columns()
                                  if col.table != joined_table and col not in partner_columns)

        for col1, col2 in root_node.join_condition.join_partners():
            joined_column, partner_column = (col1, col2) if col1.table == joined_table else (col2, col1)
            self._update_partner_column_frequency(joined_column, partner_column)

        joined_columns_frequencies = {joined_col: self.attribute_frequencies[joined_col] for joined_col
                                      in root_node.join_condition.columns_of(joined_table)}
        lowest_joined_column_frequency = dict_utils.argmin(joined_columns_frequencies)
        for third_party_column in third_party_columns:
            self._update_third_party_column_frequency(lowest_joined_column_frequency, third_party_column)

    def _inflate_base_table_estimates(self):
        for table in self.query.tables():
            table_estimate = self.base_table_estimator.estimate_for(table)
            self.base_table_estimates[table] = table_estimate

    @abc.abstractmethod
    def _inflate_attribute_frequencies(self):
        raise NotImplementedError

    @abc.abstractmethod
    def _update_partner_column_frequency(self, joined_column: base.ColumnReference,
                                         partner_column: base.ColumnReference) -> None:
        raise NotImplementedError

    @abc.abstractmethod
    def _update_third_party_column_frequency(self, joined_column: base.ColumnReference,
                                             third_party_column: base.ColumnReference) -> None:
        raise NotImplementedError


class MaxFrequencyStatsContainer(StatisticsContainer[MaxFrequency]):
    def __init__(self, database_stats: db.DatabaseStatistics):
        super().__init__()
        self.database_stats = database_stats

    def _inflate_attribute_frequencies(self):
        pass

    def _update_partner_column_frequency(self, joined_column: base.ColumnReference,
                                         partner_column: base.ColumnReference) -> None:
        joined_frequency = self.attribute_frequencies[joined_column]
        partner_frequency = self.attribute_frequencies[partner_column]
        self.attribute_frequencies[joined_column] *= partner_frequency
        self.attribute_frequencies[partner_column] *= joined_frequency

    def _update_third_party_column_frequency(self, joined_column: base.ColumnReference,
                                             third_party_column: base.ColumnReference) -> None:
        self.attribute_frequencies[third_party_column] *= self.attribute_frequencies[joined_column]
