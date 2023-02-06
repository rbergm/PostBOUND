from __future__ import annotations

import abc

from postbound.qal import qal


class OptimizationPreCheck(abc.ABC):

    @abc.abstractmethod
    def check_supported_query(self, query: qal.SqlQuery) -> bool:
        raise NotImplementedError


class UESOptimizationPreCheck(OptimizationPreCheck):
    pass


class EmptyPreCheck(OptimizationPreCheck):
    def check_supported_query(self, query: qal.SqlQuery) -> bool:
        return True


class UnsupportedQueryError(RuntimeError):
    def __init__(self, query: qal.SqlQuery) -> None:
        super().__init__(f"Query contains unsupported features: {query}")
        self.query = query
