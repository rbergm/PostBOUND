from __future__ import annotations

import abc
from typing import Iterable

from postbound.qal import base, qal


class JoinBoundCardinalityEstimator(abc.ABC):
    def __init__(self, name: str) -> None:
        self.name = name

    @abc.abstractmethod
    def setup_for_query(self, query: qal.ImplicitSqlQuery) -> None:
        raise NotImplementedError

    @abc.abstractmethod
    def estimate_for(self, join: Iterable[base.TableReference]) -> int:
        raise NotImplementedError
