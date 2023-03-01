from __future__ import annotations

import abc

from postbound.qal import qal
from postbound.optimizer import data
from postbound.optimizer.physops import operators


class HintProvider(abc.ABC):
    @abc.abstractmethod
    def adapt_query(self, query: qal.ImplicitSqlQuery, join_order: data.JoinTree | None,
                    physical_operators: operators.PhysicalOperatorAssignment | None) -> qal.SqlQuery:
        raise NotImplementedError
