from __future__ import annotations

import enum
from typing import Iterable

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

    def set_operator_enabled_globally(self, operator: ScanOperators | JoinOperators, enabled: bool):
        self.global_settings[operator] = enabled

    def set_join_operator(self, tables: Iterable[base.TableReference], operator: JoinOperators):
        self.join_operators[frozenset(tables)] = operator

    def set_scan_operator(self, base_table: base.TableReference, operator: ScanOperators):
        self.scan_operators[base_table] = operator

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
