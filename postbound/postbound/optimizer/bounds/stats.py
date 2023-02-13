from __future__ import annotations

import abc
import typing
import collections
from typing import Generic

from postbound.qal import base, qal
from postbound.optimizer import data
from postbound.optimizer.bounds import scans

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
            self._stats_container.trigger_frequency_update()


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

    @abc.abstractmethod
    def trigger_frequency_update(self):
        # TODO: we need to figure out the n:m joined table, get the frequency of the join attribute and apply this
        # frequency increase the other frequencies
        raise NotImplementedError

    def _inflate_base_table_estimates(self):
        for table in self.query.tables():
            table_estimate = self.base_table_estimator.estimate_for(table)
            self.base_table_estimates[table] = table_estimate

    @abc.abstractmethod
    def _inflate_attribute_frequencies(self):
        raise NotImplementedError
