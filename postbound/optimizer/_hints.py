
from __future__ import annotations

import math
from collections.abc import Collection, Iterable
from dataclasses import dataclass
from enum import Enum
from typing import Any, overload

from .. import util
from .._core import ScanOperators, JoinOperators, PhysicalOperator
from ..qal import parser, TableReference, SqlExpression
from ..util import jsondict


class ScanOperatorAssignment:
    """Models the selection of a scan operator to a specific base table.

    Attributes
    -------
    operator : ScanOperators
        The selected operator
    table : TableReference
        The table that is scanned using the operator
    parallel_workers : float | int
        The number of parallel processes that should be used to execute the scan. Can be set to 1 to indicate sequential
        operation. Defaults to NaN to indicate that no choice has been made.
    """
    def __init__(self, operator: ScanOperators, table: TableReference, parallel_workers: float | int = math.nan) -> None:
        self._operator = operator
        self._table = table
        self._parallel_workers = parallel_workers
        self._hash_val = hash((self._operator, self._table, self._parallel_workers))

    @property
    def operator(self) -> ScanOperators:
        """Get the assigned operator.

        Returns
        -------
        ScanOperators
            The operator
        """
        return self._operator

    @property
    def table(self) -> TableReference:
        """Get the table being scanned.

        Returns
        -------
        TableReference
            The table
        """
        return self._table

    @property
    def parallel_workers(self) -> int | float:
        """Get the number of parallel workers used for the scan.

        This number designates the total number of parallel processes. It can be 1 to indicate sequential operation, or even
        *NaN* if it is unknown.

        Returns
        -------
        int | float
            The number of workers
        """
        return self._parallel_workers

    def inspect(self) -> str:
        """Provides the scan as a natural string.

        Returns
        -------
        str
            A string representation of the assignment
        """
        return f"USING {self.operator}" if self.operator else ""

    def __json__(self) -> jsondict:
        return {"operator": self.operator.value, "table": self.table, "parallel_workers": self.parallel_workers}

    def __hash__(self) -> int:
        return self._hash_val

    def __eq__(self, other: object) -> bool:
        return (isinstance(other, type(self))
                and self.operator == other.operator
                and self.table == other.table
                and self.parallel_workers == other.parallel_workers)

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
    join : Collection[TableReference]
        The base tables that are joined using the operator
    parallel_workers : float | int, optional
        The number of parallel processes that should be used to execute the join. Can be set to 1 to indicate sequential
        operation. Defaults to NaN to indicate that no choice has been made.

    Raises
    ------
    ValueError
        If `join` contains less than 2 tables
    """

    def __init__(self, operator: JoinOperators, join: Collection[TableReference], *,
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
    def join(self) -> frozenset[TableReference]:
        """Get the tables that are joined together.

        For joins of more than 2 base tables this usually means that the join combines an intermediate result with a base table
        or another intermediate result. These two cases are not distinguished by the assignment and have to be detected
        through other information, e.g. the join tree.

        The more verbose model of a `DirectionalJoinOperatorAssignment` also distinguishes between inner and outer relations.

        Returns
        -------
        frozenset[TableReference]
            The tables that are joined together
        """
        return self._join

    @property
    def parallel_workers(self) -> float | int:
        """Get the number of parallel processes that should be used in the join.

        "Processes" does not necessarily mean "system processes". The database system can also choose to use threads or other
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

    def __json__(self) -> jsondict:
        return {"directional": self.is_directional(), "operator": self.operator.value, "join": self.join,
                "parallel_workers": self.parallel_workers}

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
    inner : Collection[TableReference]
        The tables that form the inner relation of the join
    outer : Collection[TableReference]
        The tables that form the outer relation of the join
    parallel_workers : float | int, optional
        The number of parallel processes that should be used to execute the join. Can be set to 1 to indicate sequential
        operation. Defaults to NaN to indicate that no choice has been made.

    Raises
    ------
    ValueError
        If either `inner` or `outer` is empty.
    """
    def __init__(self, operator: JoinOperators, inner: Collection[TableReference],
                 outer: Collection[TableReference], *, parallel_workers: float | int = math.nan) -> None:
        if not inner or not outer:
            raise ValueError("Both inner and outer relations must be given")
        self._inner = frozenset(inner)
        self._outer = frozenset(outer)
        super().__init__(operator, self._inner | self._outer, parallel_workers=parallel_workers)

    @property
    def inner(self) -> frozenset[TableReference]:
        """Get the inner relation of the join.

        Returns
        -------
        frozenset[TableReference]
            The tables of the inner relation
        """
        return self._inner

    @property
    def outer(self) -> frozenset[TableReference]:
        """Get the outer relation of the join.

        Returns
        -------
        frozenset[TableReference]
            The tables of the outer relation
        """
        return self._outer

    def is_directional(self) -> bool:
        return True

    def __json__(self) -> jsondict:
        return {"directional": True, "operator": self.operator, "inner": self.inner, "outer": self.outer,
                "parallel_workers": self.parallel_workers}

    __hash__ = JoinOperatorAssignment.__hash__

    def __eq__(self, other: object) -> bool:
        return (isinstance(other, type(self))
                and self._inner == other._inner
                and self._outer == other._outer
                and super().__eq__(other))


@overload
def read_operator_json(json_data: str) -> ScanOperators | JoinOperators:
    """Reconstructs a physical operator from its JSON representation.

    Parameters
    ----------
    json_data : str
        The JSON data

    Returns
    -------
    ScanOperators | JoinOperators
        The corresponding operator
    """
    pass


@overload
def read_operator_json(json_data: dict) -> ScanOperatorAssignment | JoinOperatorAssignment:
    """Reconstructs a physical operator assignment from its JSON representation.

    Parameters
    ----------
    json_data : dict
        The JSON data

    Returns
    -------
    ScanOperatorAssignment | JoinOperatorAssignment
        The parsed assignment. Whether it is a scan or join assignment is inferred from the JSON dictionary.
    """
    pass


def read_operator_json(json_data: dict | str) -> PhysicalOperator | ScanOperatorAssignment | JoinOperatorAssignment:
    """Reads a physical operator assignment from a JSON dictionary.

    The precise type of return value is determined based on the supplied argument: a string parameter will provide a plain
    operator whereas each dictionary is assumed to describe an operator assignment.

    Parameters
    ----------
    json_data : dict | str
        The JSON dictionary to read from

    Returns
    -------
    ScanOperators | JoinOperators | ScanOperatorAssignment | JoinOperatorAssignment
        The parsed assignment. Whether it is a scan or join assignment is inferred from the JSON dictionary.

    Raises
    ------
    ValueError
        If the JSON dictionary does not contain a valid assignment
    """
    if isinstance(json_data, str):
        if json_data in {op.value for op in ScanOperators}:
            return ScanOperators(json_data)
        elif json_data in {op.value for op in JoinOperators}:
            return JoinOperators(json_data)
        else:
            raise ValueError(f"Unknown physical operator: '{json_data}'")

    parallel_workers = json_data.get("parallel_workers", math.nan)

    if "table" in json_data:
        parsed_table = parser.load_table_json(json_data["table"])
        scan_operator = ScanOperators(json_data["operator"])
        return ScanOperatorAssignment(scan_operator, parsed_table, parallel_workers)
    elif "join" not in json_data and not ("inner" in json_data and "outer" in json_data):
        raise ValueError(f"Malformed operator JSON: either 'table' or 'join' must be given: '{json_data}'")

    directional = json_data["directional"]
    join_operator = JoinOperators(json_data["operator"])
    if directional:
        inner = [parser.load_table_json(tab) for tab in json_data["inner"]]
        outer = [parser.load_table_json(tab) for tab in json_data["outer"]]
        return DirectionalJoinOperatorAssignment(join_operator, inner, outer, parallel_workers=parallel_workers)

    joined_tables = [parser.load_table_json(tab) for tab in json_data["join"]]
    return JoinOperatorAssignment(join_operator, joined_tables, parallel_workers=parallel_workers)


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

    The assignment enables ``__getitem__`` access and tries to determine the requested setting in an intelligent way, i.e.
    supplying a single base table will provide the associated scan operator, supplying an iterable of base tables the join
    operator and supplying an operator will return the global setting. If no item is found, ``None`` will be returned.

    Attributes
    ----------
    global_settings : dict[ScanOperators | JoinOperators, bool]
        Contains the global settings. Each operator is mapped to whether it is enable for the entire query or not. If an
        operator is not present in the dictionary, the default setting of the database system is used.
    join_operators : dict[frozenset[TableReference], JoinOperatorAssignment]
        Contains the join operators that should be used for individual joins. All joins are identified by the base tables that
        they combine. If a join does not appear in this dictionary, the database system has to choose an appropriate operator
        (perhaps while considering the `global_settings`).
    scan_operators : dict[TableReference, ScanOperatorAssignment]
        Contains the scan operators that should be used for individual base table scans. Each scan is identified by the table
        that should be scanned. If a table does not appear in this dictionary, the database system has to choose an appropriate
        operator (perhaps while considering the `global_settings`).
    """

    def __init__(self) -> None:
        self.global_settings: dict[ScanOperators | JoinOperators, bool] = {}
        self.join_operators: dict[frozenset[TableReference], JoinOperatorAssignment] = {}
        self.scan_operators: dict[TableReference, ScanOperatorAssignment] = {}

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

    def __bool__(self) -> bool:
        return bool(self.global_settings) or bool(self.join_operators) or bool(self.scan_operators)

    def __getitem__(self, item: TableReference | Iterable[TableReference] | ScanOperators | JoinOperators
                    ) -> ScanOperatorAssignment | JoinOperatorAssignment | bool | None:
        if isinstance(item, ScanOperators) or isinstance(item, JoinOperators):
            return self.global_settings.get(item, None)
        elif isinstance(item, TableReference):
            return self.scan_operators.get(item, None)
        elif isinstance(item, Iterable):
            return self.join_operators.get(frozenset(item), None)
        else:
            return None

    def __hash__(self) -> int:
        return hash((util.hash_dict(self.global_settings),
                     util.hash_dict(self.scan_operators),
                     util.hash_dict(self.join_operators)))

    def __eq__(self, other: object) -> bool:
        return (isinstance(other, type(self))
                and self.global_settings == other.global_settings
                and self.scan_operators == other.scan_operators
                and self.join_operators == other.join_operators)

    def __repr__(self) -> str:
        return str(self)

    def __str__(self) -> str:
        global_str = ", ".join(f"{op.value}: {enabled}" for op, enabled in self.global_settings.items())
        scans_str = ", ".join(
            f"{scan.table.identifier()}: {scan.operator.value}" for scan in self.scan_operators.values())
        joins_keys = ((join, " ⨝ ".join(tab.identifier() for tab in join.join)) for join in self.join_operators.values())
        joins_str = ", ".join(f"{key}: {join.operator.value}" for join, key in joins_keys)
        return f"global=[{global_str}] scans=[{scans_str}] joins=[{joins_str}]"


class PlanParameterization:
    """The plan parameterization stores metadata that is assigned to different parts of the plan.

    Currently, three types of parameters are supported:

    - `cardinality_hints` provide specific cardinality estimates for individual joins or tables. These can be used to overwrite
      the estimation of the native database system
    - `parallel_worker_hints` indicate how many worker processes should be used to execute individual joins or table
      scans (assuming that the selected operator can be parallelized). Notice that this can also be indicated as part of the
      `physops` module
    - `system_specific_settings` can be used to enable or disable specific optimization or execution features of the target
      database. For example, they can be used to disable parallel execution or switch to another cardinality estimation method.
      Such settings should be used sparingly since they defeat the purpose of optimization algorithms that are independent of
      specific database systems.

    Although it is allowed to modify the different dictionaries directly, the more high-level methods should be used instead.
    This ensures that all potential (future) invariants are maintained.

    Attributes
    ----------
    cardinality_hints : dict[frozenset[TableReference], int | float]
        Contains the cardinalities for individual joins and scans. This is always the cardinality that is emitted by a specific
        operator. All joins are identified by the base tables that they combine. Keys of single tables correpond to scans.
    paralell_worker_hints : dict[frozenset[TableReference], int]
        Contains the number of parallel processes that should be used to execute a join or scan. All joins are identified by
        the base tables that they combine. Keys of single tables correpond to scans. "Processes" does not necessarily mean
        "system processes". The database system can also choose to use threads or other means of parallelization. This is not
        restricted by the join assignment.
    system_specific_settings : dict[str, Any]
        Contains the settings for the target database system. The keys and values, as well as their usage depend entirely on
        the system. For example, in Postgres a setting like *enable_geqo = 'off'* can be used to disable the genetic optimizer.
        During query execution, this is applied as preparatory statement before the actual query is executed.
    """

    def __init__(self) -> None:
        self.cardinality_hints: dict[frozenset[TableReference], int | float] = {}
        self.parallel_worker_hints: dict[frozenset[TableReference], int] = {}
        self.system_specific_settings: dict[str, Any] = {}

    def add_cardinality_hint(self, tables: Iterable[TableReference], cardinality: int | float) -> None:
        """Assigns a specific cardinality hint to a (join of) tables.

        Parameters
        ----------
        tables : Iterable[TableReference]
            The tables for which the hint is generated. This can be an iterable of a single table, which denotes a scan hint.
        cardinality : int | float
            The estimated or known cardinality.
        """
        self.cardinality_hints[frozenset(tables)] = cardinality

    def add_parallelization_hint(self, tables: Iterable[TableReference], num_workers: int) -> None:
        """Assigns a specific number of parallel workers to a (join of) tables.

        How these workers are implemented depends on the database system. They could become actual system processes, threads,
        etc.

        Parameters
        ----------
        tables : Iterable[TableReference]
            The tables for which the hint is generated. This can be an iterable of a single table, which denotes a scan hint.
        num_workers : int
            The desired number of worker processes. This denotes the total number of processes, not an additional amount. For
            some database systems this is an important distinction since one operator node will always be created. This node
            is then responsible for spawning the workers, but can also take part in the actual calculation. To prevent one-off
            errors, we standardize this number to denote the total number of workers that take part in the calculation.
        """
        self.parallel_worker_hints[frozenset(tables)] = num_workers

    def set_system_settings(self, setting_name: str = "", setting_value: Any = None, **kwargs) -> None:
        """Stores a specific system setting.

        This may happen in one of two ways: giving the setting name and value as two different parameters, or combining their
        assignment in the keyword parameters. While the first is limited to a single parameter, the second can be used to
        assign an arbitrary number of settings. However, this is limited to setting names that form valid keyword names.

        Parameters
        ----------
        setting_name : str, optional
            The name of the setting when using the separate key/value assignment mode. Defaults to an empty string to enable
            the integrated keyword parameter mode.
        setting_value : Any, optional
            The setting's value when using the separate key/value assignment mode. Defaults to ``None`` to enable the
            integrated keyword parameter mode.
        **kwargs
            The key/value pairs in the integrated keyword parameter mode.

        Raises
        ------
        ValueError
            If both the `setting_name` as well as keyword arguments are given
        ValueError
            If neither the `setting_name` nor keyword arguments are given

        Examples
        --------
        Using the separate setting name and value syntax: ``set_system_settings("join_collapse_limit", 1)``
        Using the kwargs syntax: ``set_system_settings(join_collapse_limit=1, jit=False)``
        Both examples are specific to Postgres (see https://www.postgresql.org/docs/current/runtime-config-query.html).
        """
        if setting_name and kwargs:
            raise ValueError("Only setting or kwargs can be supplied")
        elif not setting_name and not kwargs:
            raise ValueError("setting_name or kwargs required!")

        if setting_name:
            self.system_specific_settings[setting_name] = setting_value
        else:
            self.system_specific_settings |= kwargs

    def merge_with(self, other_parameters: PlanParameterization) -> PlanParameterization:
        """Combines the current parameters with additional hints.

        In case of assignments to the same hints, the values from the other parameters take precedence. None of the input
        parameterizations are modified.

        Parameters
        ----------
        other_parameters : PlanParameterization
            The parameterization to combine with the current parameterization

        Returns
        -------
        PlanParameterization
            The merged parameters
        """
        merged_params = PlanParameterization()
        merged_params.cardinality_hints = self.cardinality_hints | other_parameters.cardinality_hints
        merged_params.parallel_worker_hints = self.parallel_worker_hints | other_parameters.parallel_worker_hints
        merged_params.system_specific_settings = (self.system_specific_settings
                                                  | other_parameters.system_specific_settings)
        return merged_params

    def __repr__(self) -> str:
        return str(self)

    def __str__(self) -> str:
        return (f"PlanParams(cards={self.cardinality_hints}, "
                f"system specific={self.system_specific_settings}, par workers={self.parallel_worker_hints})")


class HintType(Enum):
    """Contains all hint types that are supported by PostBOUND.

    Notice that not all of these hints need to be represented in the `PlanParameterization`, since some of them concern other
    aspects such as the join order. Furthermore, not all database systems will support all operators. The availability of
    certain hints can be checked on the database system interface and should be handled as part of the optimization pre-checks.
    """
    LinearJoinOrder = "Join order"
    JoinDirection = "Join direction"
    BushyJoinOrder = "Bushy join order"
    Operator = "Physical operators"
    Parallelization = "Par. workers"
    Cardinality = "Cardinality"


@dataclass(frozen=True)
class SortKey:
    """Sort keys describe how the tuples in a relation are sorted.

    Attributes
    ----------
    column : SqlExpression
        The column that is used to sort the tuples. This will usually be a column reference, but can also be a more complex
        expression.
    ascending : bool
        Whether the sorting is ascending or descending. Defaults to ascending.
    """

    column: SqlExpression
    ascending: bool = True

    @staticmethod
    def of(column: SqlExpression, ascending: bool = True) -> SortKey:
        """Creates a new sort key.

        This is just a more expressive alias for the constructor.

        Parameters
        ----------
        column : SqlExpression
            The column that is used to sort the tuples. This will usually be a column reference, but can also be a more complex
            expression.
        ascending : bool, optional
            Whether the sorting is ascending or descending. Defaults to ascending.

        Returns
        -------
        SortKey
            The sort key
        """
        return SortKey(column, ascending)

    def __str__(self):
        if self.ascending:
            return str(self.column)
        return f"{self.column} DESC"
