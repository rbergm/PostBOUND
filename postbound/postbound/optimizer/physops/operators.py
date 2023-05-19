"""Provides the central data structure that stores the physical operator assignment."""
from __future__ import annotations

import enum
import math
import typing
from collections.abc import Collection
from dataclasses import dataclass
from typing import Iterable

from postbound.qal import base


class ScanOperators(enum.Enum):
    """The scan operators supported by PostBOUND.

    These can differ from the scan operators that are actually available in the selected target database system.
    """
    SequentialScan = "Seq. Scan"
    IndexScan = "Idx. Scan"
    IndexOnlyScan = "Idx-only Scan"
    BitmapScan = "Bitmap Scan"


class JoinOperators(enum.Enum):
    """The join operators supported by PostBOUND.

    These can differ from the join operators that are actually available in the selected target database system.
    """
    NestedLoopJoin = "NLJ"
    HashJoin = "Hash Join"
    SortMergeJoin = "Sort-Merge Join"
    IndexNestedLoopJoin = "Idx. NLJ"


@dataclass(frozen=True)
class ScanOperatorAssignment:
    operator: ScanOperators
    table: base.TableReference
    parallel_workers: float | int = math.nan

    def inspect(self) -> str:
        return f"USING {self.operator}" if self.operator else ""

    def __repr__(self) -> str:
        return str(self)

    def __str__(self) -> str:
        return f"{self.operator.value}({self.table})"


class JoinOperatorAssignment:
    def __init__(self, operator: JoinOperators, join: Collection[base.TableReference], *,
                 parallel_workers: float | int = math.nan) -> None:
        if not join:
            raise ValueError("Joined tables must be given")
        self._operator = operator
        self._join = frozenset(join)
        self._parallel_workers = parallel_workers

        self._hash_val = hash((self._operator, self._join, self._parallel_workers))

    @property
    def operator(self) -> JoinOperators:
        return self._operator

    @property
    def join(self) -> frozenset[base.TableReference]:
        return self._join

    @property
    def parallel_workers(self) -> float | int:
        return self._parallel_workers

    def inspect(self) -> str:
        return f"USING {self.operator}" if self.operator else ""

    def is_directional(self) -> bool:
        return False

    def __hash__(self) -> int:
        return self._hash_val

    def __eq__(self, other: object) -> bool:
        return (isinstance(other, type(self))
                and self._operator == other._operator
                and self._join == other._join
                and self._parallel_workers == other._parallel_workers)

    def __repr__(self) -> str:
        return str(self)

    def __str__(self) -> str:
        join_str = ", ".join(str(tab) for tab in self.join)
        return f"{self.operator.value}({join_str})"


class DirectionalJoinOperatorAssignment(JoinOperatorAssignment):
    def __init__(self, operator: JoinOperators, inner: Collection[base.TableReference],
                 outer: Collection[base.TableReference], *, parallel_workers: float | int = math.nan) -> None:
        if not inner or not outer:
            raise ValueError("Both inner and outer relations must be given")
        self._inner = frozenset(inner)
        self._outer = frozenset(outer)
        super().__init__(operator, self._inner | self._outer, parallel_workers=parallel_workers)

    @property
    def inner(self) -> frozenset[base.TableReference]:
        return self._inner

    @property
    def outer(self) -> frozenset[base.TableReference]:
        return self._outer

    def is_directional(self) -> bool:
        return True

    __hash__ = JoinOperatorAssignment.__hash__

    def __eq__(self, other: object) -> bool:
        return (isinstance(other, type(self))
                and self._inner == other._inner
                and self._outer == other._outer
                and super().__eq__(other))


PhysicalOperator = typing.Union[ScanOperators, JoinOperators]
"""All physical operators supported by PostBOUND.

These can differ from the operators that are actually available in the selected target database system.
"""


class PhysicalOperatorAssignment:
    """The `PhysicalOperatorAssignment` stores the operators that should be used for specific joins or scans.

    The assignment can happen at different levels:

    - `global_settings` enable or disable specific operators for the entire query
    - `join_operators` and `scan_operators` are concerned with specific (joins of) base tables. These assignments
    overwrite the global settings, i.e. it is possible to assign a nested loop join to a specific set of tables, but
    disable NLJ globally. In this case, only the specified join will be executed as an NLJ and other algorithms are
    used for all other joins

    The basic assumption here is that for all joins and scans that have no assignment, the database system that handles
    the actual query execution should determine the best operators by itself.

    The assignment enables `__getitem__` access tries to determine the requested setting in an intelligent way, i.e.
    supplying a single base table will provide the associated scan operator, supplying an iterable of base tables
    the join operator and supplying an operator will return the global setting.
    """

    def __init__(self) -> None:
        self.global_settings: dict[ScanOperators | JoinOperators, bool] = {}
        self.join_operators: dict[frozenset[base.TableReference], JoinOperatorAssignment] = {}
        self.scan_operators: dict[base.TableReference, ScanOperatorAssignment] = {}

    def set_operator_enabled_globally(self, operator: ScanOperators | JoinOperators, enabled: bool, *,
                                      overwrite_fine_grained_selection: bool = False) -> None:
        """Enables or disables an operator for all joins/scans in the query.

        If `overwrite_overwrite_fine_grained_selection` is `True`, this also drops all assignments to individual
        scans/joins that contradict the global setting.
        """
        self.global_settings[operator] = enabled

        if not overwrite_fine_grained_selection or enabled:
            return

        # at this point we know that we should disable a scan or join operator that was potentially set for
        # individual joins or tables
        if isinstance(operator, ScanOperators):
            self.scan_operators = {table: current_setting for table, current_setting in self.scan_operators.items()
                                   if current_setting != operator}
        elif isinstance(operator, JoinOperators):
            self.join_operators = {join: current_setting for join, current_setting in self.join_operators.items()
                                   if current_setting != operator}

    def set_join_operator(self, join_operator: JoinOperatorAssignment) -> None:
        """Enforces the specified operator for the join that consists of the given tables."""
        self.join_operators[join_operator.join] = join_operator

    def set_scan_operator(self, scan_operator: ScanOperatorAssignment) -> None:
        """Enforces the specified scan operator for the given base table."""
        self.scan_operators[scan_operator.table] = scan_operator

    def merge_with(self, other_assignment: PhysicalOperatorAssignment) -> PhysicalOperatorAssignment:
        """Combines the current optimization settings with the settings from the `other_assignment`.

        In case of contradicting assignments, the `other_settings` take precedence.
        """
        merged_assignment = PhysicalOperatorAssignment()
        merged_assignment.global_settings = self.global_settings | other_assignment.global_settings
        merged_assignment.join_operators = self.join_operators | other_assignment.join_operators
        merged_assignment.scan_operators = self.scan_operators | other_assignment.scan_operators
        return merged_assignment

    def global_settings_only(self) -> PhysicalOperatorAssignment:
        """Provides only those settings of the assignment that affect all operators."""
        global_assignment = PhysicalOperatorAssignment()
        global_assignment.global_settings = dict(self.global_settings)
        return global_assignment

    def clone(self) -> PhysicalOperatorAssignment:
        """Provides a copy of the current settings."""
        cloned_assignment = PhysicalOperatorAssignment()
        cloned_assignment.global_settings = dict(self.global_settings)
        cloned_assignment.join_operators = dict(self.join_operators)
        cloned_assignment.scan_operators = dict(self.scan_operators)
        return cloned_assignment

    def __getitem__(self, item: base.TableReference | Iterable[base.TableReference] | ScanOperators | JoinOperators
                    ) -> ScanOperatorAssignment | JoinOperatorAssignment | bool | None:
        if isinstance(item, ScanOperators) or isinstance(item, JoinOperators):
            return self.global_settings.get(item, None)
        elif isinstance(item, base.TableReference):
            return self.scan_operators.get(item, None)
        elif isinstance(item, Iterable):
            return self.join_operators.get(frozenset(item), None)
        else:
            return None
