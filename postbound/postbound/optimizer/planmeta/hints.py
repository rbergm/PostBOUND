"""Provides the central data structure that stores the actual plan parameters."""
from __future__ import annotations

import enum
from typing import Iterable

from postbound.qal import base


class PlanParameterization:
    """The `PlanParameterization` stores the parameters that are assigned to different parts of the plan.

    Currently, two types of parameters are supported:

    - `cardinality_hints` provide specific cardinality estimates for individual joins or tables that overwrite the
    estimation of the native database system
    - `parallel_worker_hints` indicate how many worker processes should be used to execute individual joins or table
    scans (assuming that the selected operator can be parallelized)
    """

    def __init__(self) -> None:
        self.cardinality_hints: dict[frozenset[base.TableReference], int] = {}
        self.parallel_worker_hints: dict[frozenset[base.TableReference], int] = {}

    def add_cardinality_hint(self, tables: Iterable[base.TableReference], cardinality: int) -> None:
        """Assigns the given cardinality hint to the (join of) tables."""
        self.cardinality_hints[frozenset(tables)] = cardinality

    def add_parallelization_hint(self, tables: Iterable[base.TableReference], num_workers: int) -> None:
        """Assigns the given number of parallel works to the (join of) tables."""
        self.parallel_worker_hints[frozenset(tables)] = num_workers


class HintType(enum.Enum):
    """Contains all hint types that are supported by PostBOUND (or at least should be supported in the future)."""
    JoinOrderHint = "Join order"
    JoinDirectionHint = "Join direction"
    ParallelizationHint = "Par. workers"
    CardinalityHint = "Cardinality"
