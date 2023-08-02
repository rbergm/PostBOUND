"""Models physical operators and their assignments to elements of query plans."""
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

    These can differ from the scan operators that are actually available in the selected target database system. The individual
    operators are chosen because they are supported by a wide variety of database systems and they are sufficiently different
    from each other.
    """
    SequentialScan = "Seq. Scan"
    IndexScan = "Idx. Scan"
    IndexOnlyScan = "Idx-only Scan"
    BitmapScan = "Bitmap Scan"


class JoinOperators(enum.Enum):
    """The join operators supported by PostBOUND.

    These can differ from the join operators that are actually available in the selected target database system. The individual
    operators are chosen because they are supported by a wide variety of database systems and they are sufficiently different
    from each other.
    """
    NestedLoopJoin = "NLJ"
    HashJoin = "Hash Join"
    SortMergeJoin = "Sort-Merge Join"
    IndexNestedLoopJoin = "Idx. NLJ"


@dataclass(frozen=True)
class ScanOperatorAssignment:
    """Models the selection of a scan operator to a specific base table.

    Attributes
    -------
    operator : ScanOperators
        The selected operator
    table : base.TableReference
        The table that is scanned using the operator
    parallel_workers : float | int
        The number of parallel processes that should be used to execute the scan. Can be set to 1 to indicate sequential
        operation. Defaults to NaN to indicate that no choice has been made.
    """
    operator: ScanOperators
    table: base.TableReference
    parallel_workers: float | int = math.nan

    def inspect(self) -> str:
        """Provides the scan as a natural string.

        Returns
        -------
        str
            A string representation of the assignment
        """
        return f"USING {self.operator}" if self.operator else ""

    def __repr__(self) -> str:
        return str(self)

    def __str__(self) -> str:
        return f"{self.operator.value}({self.table})"


class JoinOperatorAssignment:
    """Models the selection of a join operator for a specific join of tables.

    Each join is identified by all base tables that are involved in the join. The assignment to intermediate results does not
    matter here. For example, a join between R ⨝ S and T is expressed as R, S, T even though the actual join combined an
    intermediate result with as base table.

    A more verbose model is provided by the `DirectionalJoinOperatorAssignment`. In addition to the joined tables, that model
    also distinguishes between inner and outer relation of the join.

    Parameters
    ----------
    operator : JoinOperators
        The selected operator
    join : Collection[base.TableReference]
        The base tables that are joined using the operator
    parallel_workers : float | int, optional
        The number of parallel processes that should be used to execute the join. Can be set to 1 to indicate sequential
        operation. Defaults to NaN to indicate that no choice has been made.

    Raises
    ------
    ValueError
        If `join` contains less than 2 tables
    """

    def __init__(self, operator: JoinOperators, join: Collection[base.TableReference], *,
                 parallel_workers: float | int = math.nan) -> None:
        if len(join) < 2:
            raise ValueError("At least 2 join tables must be given")
        self._operator = operator
        self._join = frozenset(join)
        self._parallel_workers = parallel_workers

        self._hash_val = hash((self._operator, self._join, self._parallel_workers))

    @property
    def operator(self) -> JoinOperators:
        """Get the operator that was selected for the join

        Returns
        -------
        JoinOperators
            The operator
        """
        return self._operator

    @property
    def join(self) -> frozenset[base.TableReference]:
        """Get the tables that are joined together.

        For joins of more than 2 base tables this usually means that the join combines an intermediate result with a base table
        or another intermediate result. These two cases are not distinguished by the assignment and have to be detected
        through other information, e.g. the join tree.

        The more verbose model of a `DirectionalJoinOperatorAssignment` also distinguishes between inner and outer relations.

        Returns
        -------
        frozenset[base.TableReference]
            The tables that are joined together
        """
        return self._join

    @property
    def parallel_workers(self) -> float | int:
        """Get the number of parallel processes that should be used in the join.

        "Processes" does not necessarily mean system processes. The database system can also choose to use threads or other
        means of parallelization. This is not restricted by the join assignment.

        Returns
        -------
        float | int
            The number processes to use. Can be 1 to indicate sequential processing or NaN to indicate that no choice has been
            made.
        """
        return self._parallel_workers

    def inspect(self) -> str:
        """Provides this assignment as a natural string.

        Returns
        -------
        str
            A string representation of the assignment.
        """
        return f"USING {self.operator}" if self.operator else ""

    def is_directional(self) -> bool:
        """Checks, whether this assignment contains directional information, i.e. regarding inner and outer relation.

        Returns
        -------
        bool
            Whether the assignment explicitly denotes which relation should be the inner relationship and which relation should
            be the outer relationship
        """
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
    """A more verbose model of join operators.

    The directional assignment does not only represent the relations that should be joined together, but also denotes which
    role they should play for the join. More specifically, the directional assignment provides the *inner* and *outer* relation
    of the join. The precise semantics of this distinction depends on the specific join operator and is also used
    inconsistently between different database systems. In PostBOUND we use the following definitions:

    - for nested-loop joins the outer relation corresponds to the outer loop and the inner relation is the inner loop. As a
      special case for index nested-loop joins ths inner relation is the one that is probed via an index
    - for hash joins the outer relation is the one that is aggregated in a hash table and the inner relation is the one that
      is probed against that table
    - for sort-merge joins the assignment does not matter

    Parameters
    ----------
    operator : JoinOperators
        The selected operator
    inner : Collection[base.TableReference]
        The tables that form the inner relation of the join
    outer : Collection[base.TableReference]
        The tables that form the outer relation of the join
    parallel_workers : float | int, optional
        The number of parallel processes that should be used to execute the join. Can be set to 1 to indicate sequential
        operation. Defaults to NaN to indicate that no choice has been made.

    Raises
    ------
    ValueError
        If either `inner` or `outer` is empty.
    """
    def __init__(self, operator: JoinOperators, inner: Collection[base.TableReference],
                 outer: Collection[base.TableReference], *, parallel_workers: float | int = math.nan) -> None:
        if not inner or not outer:
            raise ValueError("Both inner and outer relations must be given")
        self._inner = frozenset(inner)
        self._outer = frozenset(outer)
        super().__init__(operator, self._inner | self._outer, parallel_workers=parallel_workers)

    @property
    def inner(self) -> frozenset[base.TableReference]:
        """Get the inner relation of the join.

        Returns
        -------
        frozenset[base.TableReference]
            The tables of the inner relation
        """
        return self._inner

    @property
    def outer(self) -> frozenset[base.TableReference]:
        """Get the outer relation of the join.

        Returns
        -------
        frozenset[base.TableReference]
            The tables of the outer relation
        """
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
"""Supertype to model all physical operators supported by PostBOUND.

These can differ from the operators that are actually available in the selected target database system.
"""


class PhysicalOperatorAssignment:
    """The physical operator assignment stores the operators that should be used for specific joins or scans.

    The assignment can happen at different levels:

    - `global_settings` enable or disable specific operators for the entire query
    - `join_operators` and `scan_operators` are concerned with specific (joins of) base tables. These assignments overwrite the
      global settings, i.e. it is possible to assign a nested loop join to a specific set of tables, but disable NLJ globally.
      In this case, only the specified join will be executed as an NLJ and other algorithms are used for all other joins

    The basic assumption here is that for all joins and scans that have no assignment, the database system should determine the
    best operators by itself.

    Although it is allowed to modify the different dictionaries directly, the more high-level methods should be used instead.
    This ensures that all potential (future) invariants are maintained.

    The assignment enables ``__getitem__`` access tries to determine the requested setting in an intelligent way, i.e.
    supplying a single base table will provide the associated scan operator, supplying an iterable of base tables the join
    operator and supplying an operator will return the global setting. If no item is found, ``None`` will be returned.

    Attributes
    ----------
    global_settings : dict[ScanOperators | JoinOperators, bool]
        Contains the global settings. Each operator is mapped to whether it is enable for the entire query or not. If an
        operator is not present in the dictionary, the default setting of the database system is used.
    join_operators : dict[frozenset[base.TableReference], JoinOperatorAssignment]
        Contains the join operators that should be used for individual joins. All joins are identified by the base tables that
        they combine. If a join does not appear in this dictionary, the database system has to choose an appropriate operator
        (perhaps while considering the `global_settings`).
    scan_operators : dict[base.TableReference, ScanOperatorAssignment]
        Contains the scan operators that should be used for individual base table scans. Each scan is identified by the table
        that should be scanned. If a table does not appear in this dictionary, the database system has to choose an appropriate
        operator (perhaps while considering the `global_settings`).
    """

    def __init__(self) -> None:
        self.global_settings: dict[ScanOperators | JoinOperators, bool] = {}
        self.join_operators: dict[frozenset[base.TableReference], JoinOperatorAssignment] = {}
        self.scan_operators: dict[base.TableReference, ScanOperatorAssignment] = {}

    def get_globally_enabled_operators(self, include_by_default: bool = True) -> frozenset[PhysicalOperator]:
        """Provides all operators that are enabled globally.

        This differs from just calling ``assignment.global_settings`` directly, since all operators are checked, not just the
        operators that appear in the global settings dictionary.

        Parameters
        ----------
        include_by_default : bool, optional
            The behaviour for operators that do not have a global setting set. If enabled, such operators are assumed to be
            enabled and are hence included in the set.

        Returns
        -------
        frozenset[PhysicalOperator]
            The enabled scan and join operators. If no global setting is available for an operator `include_by_default`
            determines the appropriate action.
        """
        enabled_scan_ops = [scan_op for scan_op in ScanOperators if self.global_settings.get(scan_op, include_by_default)]
        enabled_join_ops = [join_op for join_op in JoinOperators if self.global_settings.get(join_op, include_by_default)]
        return frozenset(enabled_scan_ops + enabled_join_ops)

    def set_operator_enabled_globally(self, operator: ScanOperators | JoinOperators, enabled: bool, *,
                                      overwrite_fine_grained_selection: bool = False) -> None:
        """Enables or disables an operator for all joins/scans in the query.

        Parameters
        ----------
        operator : ScanOperators | JoinOperators
            The operator to configure
        enabled : bool
            Whether the database system is allowed to choose the operator
        overwrite_fine_grained_selection : bool, optional
            How to deal with assignments of the same operator to individual scans or joins. If ``True`` all such assignments
            that contradict the setting are removed. For example, consider a situation where nested-loop joins should be
            disabled globally, but a specific join has already been assigned to be executed with an NLJ. In this case, setting
            `overwrite_fine_grained_selection` removes the assignment for the specific join. This is off by default, to enable
            the per-node selection to overwrite global settings.
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
        """Enforces a specific join operator for the join that consists of the contained tables.

        This overwrites all previous assignments for the same join. Global settings are left unmodified since per-join settings
        overwrite them anyway.

        Parameters
        ----------
        join_operator : JoinOperatorAssignment
            The join operator
        """
        self.join_operators[join_operator.join] = join_operator

    def set_scan_operator(self, scan_operator: ScanOperatorAssignment) -> None:
        """Enforces a specific scan operator for the contained base table.

        This overwrites all previous assignments for the same table. Global settings are left unmodified since per-table
        settings overwrite them anyway.

        Parameters
        ----------
        scan_operator : ScanOperatorAssignment
            The scan operator
        """
        self.scan_operators[scan_operator.table] = scan_operator

    def merge_with(self, other_assignment: PhysicalOperatorAssignment) -> PhysicalOperatorAssignment:
        """Combines the current assignment with additional operators.

        In case of assignments to the same operators, the settings from the other assignment take precedence. None of the input
        assignments are modified.

        Parameters
        ----------
        other_assignment : PhysicalOperatorAssignment
            The assignment to combine with the current assignment

        Returns
        -------
        PhysicalOperatorAssignment
            The combined assignment
        """
        merged_assignment = PhysicalOperatorAssignment()
        merged_assignment.global_settings = self.global_settings | other_assignment.global_settings
        merged_assignment.join_operators = self.join_operators | other_assignment.join_operators
        merged_assignment.scan_operators = self.scan_operators | other_assignment.scan_operators
        return merged_assignment

    def global_settings_only(self) -> PhysicalOperatorAssignment:
        """Provides an assignment that only contains the global settings.

        Changes to the global settings of the derived assignment are not reflected in this assignment and vice-versa.

        Returns
        -------
        PhysicalOperatorAssignment
            An assignment of the global settings
        """
        global_assignment = PhysicalOperatorAssignment()
        global_assignment.global_settings = dict(self.global_settings)
        return global_assignment

    def clone(self) -> PhysicalOperatorAssignment:
        """Provides a copy of the current settings.

        Changes to the copy are not reflected back on this assignment and vice-versa.

        Returns
        -------
        PhysicalOperatorAssignment
            The copy
        """
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

    def __repr__(self) -> str:
        return str(self)

    def __str__(self) -> str:
        global_str = ", ".join(f"{op.value}: {enabled}" for op, enabled in self.global_settings.items())
        scans_str = ", ".join(
            f"{scan.table.identifier()}: {scan.operator.value}" for scan in self.scan_operators.values())
        joins_keys = ((join, "⨝".join(tab.identifier() for tab in join.join)) for join in self.join_operators.values())
        joins_str = ", ".join(f"{key}: {join.operator.value}" for join, key in joins_keys)
        return f"global=[{global_str}] scans=[{scans_str}] joins=[{joins_str}]"
