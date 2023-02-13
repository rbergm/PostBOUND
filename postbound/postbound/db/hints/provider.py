from __future__ import annotations

import abc

from postbound.qal import qal
from postbound.optimizer import data
from postbound.optimizer.physops import operators


class HintProvider(abc.ABC):
    @abc.abstractmethod
    def adapt_query(self, query: qal.SqlQuery, join_order: data.JoinTree,
                    physical_operators: operators.PhysicalOperatorAssignment) -> qal.SqlQuery:
        raise NotImplementedError


class PostgresHintProvider(HintProvider):
    pass
