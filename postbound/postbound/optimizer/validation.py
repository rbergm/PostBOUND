from __future__ import annotations

import abc

from postbound.qal import qal


class OptimizationPreCheck(abc.ABC):

    @abc.abstractmethod
    def check_supported_query(self, query: qal.ImplicitSqlQuery) -> bool:
        raise NotImplementedError


class UESOptimizationPreCheck(OptimizationPreCheck):
    pass
