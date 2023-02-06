from __future__ import annotations

import abc

from postbound.qal import base, qal


class BaseTableCardinalityEstimator(abc.ABC):

    def __init__(self, name: str) -> None:
        self.name = name

    @abc.abstractmethod
    def setup_for_query(self, query: qal.ImplicitSqlQuery) -> None:
        raise NotImplementedError

    @abc.abstractmethod
    def estimate_for(self, table: base.TableReference) -> int:
        raise NotImplementedError

    def __getitem__(self, item: base.TableReference) -> int:
        return self.estimate_for(item)
