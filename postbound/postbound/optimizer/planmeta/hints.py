from __future__ import annotations

import enum
from typing import Iterable

from postbound.qal import base


class PlanParameterization:
    def __init__(self) -> None:
        self.cardinality_hints: dict[frozenset[base.TableReference], int] = {}
        self.parallel_worker_hints: dict[frozenset[base.TableReference], int] = {}

    def add_cardinality_hint(self, tables: Iterable[base.TableReference], cardinality: int) -> None:
        self.cardinality_hints[frozenset(tables)] = cardinality

    def add_parallelization_hint(self, tables: Iterable[base.TableReference], num_workers: int) -> None:
        self.parallel_worker_hints[frozenset(tables)] = num_workers


class HintType(enum.Enum):
    JoinOrderHint = "Join order"
    JoinDirectionHint = "Join direction"
    ParallelizationHint = "Par. workers"
    CardinalityHint = "Cardinality"
