from __future__ import annotations

import abc
from typing import Iterable

from postbound.qal import base, qal, predicates
from postbound.optimizer import data
from postbound.optimizer.bounds import stats


class SubqueryGenerationPolicy(abc.ABC):
    def __init__(self, name: str) -> None:
        self.name = name

    @abc.abstractmethod
    def setup_for_query(self, query: qal.ImplicitSqlQuery) -> None:
        raise NotImplementedError

    @abc.abstractmethod
    def generate_subquery_for(self, join: predicates.AbstractPredicate, join_graph: data.JoinGraph,
                              stats_container: stats.UpperBoundsContainer) -> bool:
        raise NotImplementedError


class LinearSubqueryGenerationPolicy(SubqueryGenerationPolicy):

    def setup_for_query(self, query: qal.ImplicitSqlQuery) -> None:
        pass

    def generate_subquery_for(self, join: predicates.AbstractPredicate, join_graph: data.JoinGraph,
                              stats_container: stats.UpperBoundsContainer) -> bool:
        return False
