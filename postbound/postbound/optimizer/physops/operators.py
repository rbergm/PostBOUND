from __future__ import annotations

import enum
import typing
from typing import Any, Iterable

from postbound.qal import base, qal


class ScanOperators(enum.Enum):
    SequentialScan = "Seq. Scan"
    IndexScan = "Idx. Scan"
    IndexOnlyScan = "Idx-only Scan"


class JoinOperators(enum.Enum):
    NestedLoopJoin = "NLJ"
    HashJoin = "Hash Join"
    SortMergeJoin = "Sort-Merge Join"
    IndexNestedLoopJoin = "Idx. NLJ"


PhysicalOperator = typing.Union[ScanOperators, JoinOperators]


class PhysicalOperatorAssignment:
    def __init__(self, query: qal.SqlQuery) -> None:
        self.query = query
        self.global_settings: dict[ScanOperators | JoinOperators, bool] = {}
        self.join_operators: dict[frozenset[base.TableReference], JoinOperators] = {}
        self.scan_operators: dict[base.TableReference, ScanOperators] = {}
        self.system_specific_settings: dict[str, Any] = {}

    def set_operator_enabled_globally(self, operator: ScanOperators | JoinOperators, enabled: bool, *,
                                      overwrite_fine_grained_selection: bool = False) -> None:
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

    def set_join_operator(self, tables: Iterable[base.TableReference], operator: JoinOperators) -> None:
        self.join_operators[frozenset(tables)] = operator

    def set_scan_operator(self, base_table: base.TableReference, operator: ScanOperators) -> None:
        self.scan_operators[base_table] = operator

    def set_system_settings(self, setting_name: str = "", setting_value: Any = None, **kwargs) -> None:
        if setting_name and kwargs:
            raise ValueError("Only setting or kwargs can be supplied")

        if setting_name:
            self.system_specific_settings[setting_name] = setting_value
        else:
            self.system_specific_settings |= kwargs

    def merge_with(self, other_assignment: PhysicalOperatorAssignment) -> PhysicalOperatorAssignment:
        merged_assignment = PhysicalOperatorAssignment(self.query)
        merged_assignment.global_settings = self.global_settings | other_assignment.global_settings
        merged_assignment.join_operators = self.join_operators | other_assignment.join_operators
        merged_assignment.scan_operators = self.scan_operators | other_assignment.scan_operators
        merged_assignment.system_specific_settings = (self.system_specific_settings
                                                      | other_assignment.system_specific_settings)
        return merged_assignment

    def clone(self) -> PhysicalOperatorAssignment:
        cloned_assignment = PhysicalOperatorAssignment(self.query)
        cloned_assignment.global_settings = dict(self.global_settings)
        cloned_assignment.join_operators = dict(self.join_operators)
        cloned_assignment.scan_operators = dict(self.scan_operators)
        cloned_assignment.system_specific_settings = dict(self.system_specific_settings)
        return cloned_assignment

    def __getitem__(self, item: base.TableReference | Iterable[base.TableReference] | ScanOperators | JoinOperators
                    ) -> ScanOperators | JoinOperators | bool | None:
        if isinstance(item, ScanOperators) or isinstance(item, JoinOperators):
            return self.global_settings.get(item, None)
        elif isinstance(item, base.TableReference):
            return self.scan_operators.get(item, None)
        elif isinstance(item, Iterable):
            return self.join_operators.get(frozenset(item), None)
        else:
            raise ValueError("Unknown item type: " + str(item))
