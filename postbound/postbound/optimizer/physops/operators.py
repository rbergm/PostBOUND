"""Provides the central data structure that stores the physical operator assignment."""
from __future__ import annotations

import enum
import typing
from typing import Any, Iterable

from postbound.qal import base, qal


class ScanOperators(enum.Enum):
    """The scan operators supported by PostBOUND.

    These can differ from the scan operators that are actually available in the selected target database system.
    """
    SequentialScan = "Seq. Scan"
    IndexScan = "Idx. Scan"
    IndexOnlyScan = "Idx-only Scan"


class JoinOperators(enum.Enum):
    """The join operators supported by PostBOUND.

    These can differ from the join operators that are actually available in the selected target database system.
    """
    NestedLoopJoin = "NLJ"
    HashJoin = "Hash Join"
    SortMergeJoin = "Sort-Merge Join"
    IndexNestedLoopJoin = "Idx. NLJ"


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
    - `system_specific_settings` do not fit in any of the above categories and only work for a known target database
    system. These should be used sparingly since they defeat the purpose of optimization algorithms that are
    independent of specific database systems. For example, they could modify the assignment strategy of the native
    database system. Their value also depends on the specific setting.

    The basic assumption here is that for all joins and scans that have no assignment, the database system that handles
    the actual query execution should determine the best operators by itself.

    The assignment enables `__getitem__` access tries to determine the requested setting in an intelligent way, i.e.
    supplying a single base table will provide the associated scan operator, supplying an iterable of base tables
    the join operator and supplying an operator will return the global setting.
    """

    def __init__(self, query: qal.SqlQuery) -> None:
        self.query = query
        self.global_settings: dict[ScanOperators | JoinOperators, bool] = {}
        self.join_operators: dict[frozenset[base.TableReference], JoinOperators] = {}
        self.scan_operators: dict[base.TableReference, ScanOperators] = {}
        self.system_specific_settings: dict[str, Any] = {}

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

    def set_join_operator(self, tables: Iterable[base.TableReference], operator: JoinOperators) -> None:
        """Enforces the specified operator for the join that consists of the given tables."""
        self.join_operators[frozenset(tables)] = operator

    def set_scan_operator(self, base_table: base.TableReference, operator: ScanOperators) -> None:
        """Enforces the specified scan operator for the given base table."""
        self.scan_operators[base_table] = operator

    def set_system_settings(self, setting_name: str = "", setting_value: Any = None, **kwargs) -> None:
        """Stores the given system setting.

        This may happen in one of two ways: giving the setting name and value as two different parameters, or combining
        their assignment in the keyword parameters. While the first is limited to a single parameter, the second can
        be used to assign an arbitrary number of settings. However, this is limited to setting names that form valid
        keyword names.

        Example usage with separate setting name and value: `set_system_settings("join_collapse_limit", 1)`
        Example usage with kwargs: `set_system_settings(join_collapse_limit=1, jit=False)`
        Both examples are specific to Postgres (see https://www.postgresql.org/docs/current/runtime-config-query.html).
        """
        # TODO: system settings should be moved to the plan parameters
        if setting_name and kwargs:
            raise ValueError("Only setting or kwargs can be supplied")
        elif not setting_name and not kwargs:
            raise ValueError("setting_name or kwargs required!")

        if setting_name:
            self.system_specific_settings[setting_name] = setting_value
        else:
            self.system_specific_settings |= kwargs

    def merge_with(self, other_assignment: PhysicalOperatorAssignment) -> PhysicalOperatorAssignment:
        """Combines the current optimization settings with the settings from the `other_assignment`.

        In case of contradicting assignments, the `other_settings` take precedence.
        """
        merged_assignment = PhysicalOperatorAssignment(self.query)
        merged_assignment.global_settings = self.global_settings | other_assignment.global_settings
        merged_assignment.join_operators = self.join_operators | other_assignment.join_operators
        merged_assignment.scan_operators = self.scan_operators | other_assignment.scan_operators
        merged_assignment.system_specific_settings = (self.system_specific_settings
                                                      | other_assignment.system_specific_settings)
        return merged_assignment

    def clone(self) -> PhysicalOperatorAssignment:
        """Provides a copy of the current settings."""
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
