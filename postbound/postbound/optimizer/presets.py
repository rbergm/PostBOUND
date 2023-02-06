from __future__ import annotations

import abc

from postbound.optimizer.joinorder import enumeration
from postbound.optimizer.physops import selection


class OptimizationSettings(abc.ABC):
    @abc.abstractmethod
    def build_join_order_optimizer(self) -> enumeration.JoinOrderOptimizer:
        raise NotImplementedError

    @abc.abstractmethod
    def build_physical_operator_selection(self) -> selection.PhysicalOperatorSelection:
        raise NotImplementedError


class UESOptimizationSettings(OptimizationSettings):
    pass


def fetch(key: str) -> OptimizationSettings:
    if key.upper() == "UES":
        return UESOptimizationSettings()
    else:
        raise ValueError(f"Unknown presets for key '{key}'")
