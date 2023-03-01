from __future__ import annotations

import enum
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


class PhysicalOperatorAssignment:
    def __init__(self, query: qal.SqlQuery) -> None:
        self.query = query
        self.global_settings: dict[ScanOperators | JoinOperators, bool] = {}
        self.join_operators: dict[frozenset[base.TableReference], JoinOperators] = {}
        self.scan_operators: dict[base.TableReference, ScanOperators] = {}
        self.system_specific_settings: dict[str, Any] = {}

    def set_operator_enabled_globally(self, operator: ScanOperators | JoinOperators, enabled: bool):
        self.global_settings[operator] = enabled

    def set_join_operator(self, tables: Iterable[base.TableReference], operator: JoinOperators):
        self.join_operators[frozenset(tables)] = operator

    def set_scan_operator(self, base_table: base.TableReference, operator: ScanOperators):
        self.scan_operators[base_table] = operator

    def set_system_settings(self, setting_name: str = "", setting_value: Any = None, **kwargs):
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
