from __future__ import annotations

import json
import math
from enum import Enum
from typing import Optional

from postbound._physops import (
    DirectionalJoinOperatorAssignment,
    JoinOperatorAssignment,
    ScanOperatorAssignment,
)

from .._core import (
    IntermediateOperator,
    JoinOperator,
    PhysicalOperator,
    ScanOperator,
)
from .._physops import PhysicalOperatorAssignment, PlanParameterization
from .._qep import PlanEstimates, PlanParams, QueryPlan
from ..qal import parser


def read_operator_json(
    json_data: dict | str,
) -> Optional[PhysicalOperator | ScanOperatorAssignment | JoinOperatorAssignment]:
    """Reads a physical operator assignment from a JSON dictionary.

    Parameters
    ----------
    json_data : dict | str
        Either the JSON dictionary, or a string encoding of the dictionary (which will be parsed by *json.loads*).

    Returns
    -------
    Optional[ScanOperators | JoinOperators | ScanOperatorAssignment | JoinOperatorAssignment]
        The parsed assignment. Whether it is a scan or join assignment is inferred from the JSON dictionary. If the input is
        empty or *None*, *None* is returned.
    """
    if not json_data:
        return None

    if isinstance(json_data, str):
        if json_data in {op.value for op in ScanOperator}:
            return ScanOperator(json_data)
        elif json_data in {op.value for op in JoinOperator}:
            return JoinOperator(json_data)
        elif json_data in {op.value for op in IntermediateOperator}:
            return IntermediateOperator(json_data)
        else:
            json_data = json.loads(json_data)

    parallel_workers = json_data.get("parallel_workers", math.nan)

    if "table" in json_data:
        parsed_table = parser.load_table_json(json_data["table"])
        scan_operator = ScanOperator(json_data["operator"])
        return ScanOperatorAssignment(scan_operator, parsed_table, parallel_workers)
    elif "join" not in json_data and not (
        "inner" in json_data and "outer" in json_data
    ):
        raise ValueError(
            f"Malformed operator JSON: either 'table' or 'join' must be given: '{json_data}'"
        )

    directional = json_data["directional"]
    join_operator = JoinOperator(json_data["operator"])
    if directional:
        inner = [parser.load_table_json(tab) for tab in json_data["inner"]]
        outer = [parser.load_table_json(tab) for tab in json_data["outer"]]
        return DirectionalJoinOperatorAssignment(
            join_operator, inner, outer, parallel_workers=parallel_workers
        )

    joined_tables = [parser.load_table_json(tab) for tab in json_data["join"]]
    return JoinOperatorAssignment(
        join_operator, joined_tables, parallel_workers=parallel_workers
    )


def operators_from_plan(
    query_plan: QueryPlan, *, include_workers: bool = False
) -> PhysicalOperatorAssignment:
    """Extracts the operator assignment from a whole query plan.

    Notice that this method only adds parallel workers to the assignment if explicitly told to, since this is generally
    better handled by the parameterization.
    """
    assignment = PhysicalOperatorAssignment()
    if not query_plan.operator and query_plan.input_node:
        return operators_from_plan(query_plan.input_node)

    workers = query_plan.parallel_workers if include_workers else math.nan
    match query_plan.operator:
        case ScanOperator():
            operator = ScanOperatorAssignment(
                query_plan.operator,
                query_plan.base_table,
                workers,
            )
            assignment.add(operator)
        case JoinOperator():
            operator = JoinOperatorAssignment(
                query_plan.operator,
                query_plan.tables(),
                parallel_workers=workers,
            )
            assignment.add(operator)
        case _:
            assignment.add(query_plan.operator, query_plan.tables())

    for child in query_plan.children:
        child_assignment = operators_from_plan(child)
        assignment = assignment.merge_with(child_assignment)
    return assignment


def read_operator_assignment_json(json_data: dict | str) -> PhysicalOperatorAssignment:
    """Loads an operator assignment from its JSON representation.

    Parameters
    ----------
    json_data : dict | str
        Either the JSON dictionary, or a string encoding of the dictionary (which will be parsed by *json.loads*).

    Returns
    -------
    PhysicalOperatorAssignment
        The assignment
    """
    json_data = json.loads(json_data) if isinstance(json_data, str) else json_data
    assignment = PhysicalOperatorAssignment()

    for hint in json_data.get("global_settings", []):
        enabled = hint["enabled"]
        match hint["kind"]:
            case "scan":
                assignment.global_settings[ScanOperator(hint["operator"])] = enabled
            case "join":
                assignment.global_settings[JoinOperator(hint["operator"])] = enabled
            case "intermediate":
                assignment.global_settings[IntermediateOperator(hint["operator"])] = (
                    enabled
                )
            case _:
                raise ValueError(f"Unknown operator kind: {hint['kind']}")

    for hint in json_data.get("scan_operators", []):
        parsed_table = parser.load_table_json(hint["table"])
        assignment.scan_operators[parsed_table] = ScanOperatorAssignment(
            ScanOperator(hint["operator"]), parsed_table
        )

    for hint in json_data.get("join_operators", []):
        parsed_tables = frozenset(
            parser.load_table_json(tab) for tab in hint["intermediate"]
        )
        assignment.join_operators[parsed_tables] = JoinOperatorAssignment(
            JoinOperator(hint["operator"]), parsed_tables
        )

    for hint in json_data.get("intermediate_operators", []):
        parsed_tables = frozenset(
            parser.load_table_json(tab) for tab in hint["intermediate"]
        )
        assignment.intermediate_operators[parsed_tables] = IntermediateOperator(
            hint["operator"]
        )

    return assignment


def read_plan_params_json(json_data: dict | str) -> PlanParameterization:
    """Loads a plan parameterization from its JSON representation.

    Parameters
    ----------
    json_data : dict | str
        Either the JSON dictionary, or a string encoding of the dictionary (which will be parsed by *json.loads*).

    Returns
    -------
    PlanParameterization
        The plan parameterization
    """
    json_data = json.loads(json_data) if isinstance(json_data, str) else json_data
    params = PlanParameterization()
    params.cardinalities = {
        frozenset(parser.load_table_json(tab)): card
        for tab, card in json_data.get("cardinality_hints", {}).items()
    }
    params.parallel_workers = {
        frozenset(parser.load_table_json(tab)): workers
        for tab, workers in json_data.get("parallel_worker_hints", {}).items()
    }
    return params


def update_plan(
    query_plan: QueryPlan,
    *,
    operators: Optional[PhysicalOperatorAssignment] = None,
    params: Optional[PlanParameterization] = None,
    simplify: bool = True,
) -> QueryPlan:
    """Assigns new operators and/or new estimates to a query plan, leaving the join order intact.

    Notice that this update method is not particularly smart and only operates on a per-node basis. This means that high-level
    functions that are composed of multiple operators might not be updated properly. For example, Postgres represents a hash
    join as a combination of a hash operator (which builds the actual hash table) and a follow-up hash join operator (which
    performs the probing). If the update changes the hash join to a different join, the hash operator will still exist, likely
    leading to an invalid query plan. To circumvent such problems, the query plan is by default simplified before processing.
    Simplification removes all auxiliary non-join and non-scan operators, thereby effectively only leaving those nodes with a
    corresponding operator. But, there is no free lunch and the simplification might also remove some other important
    operators, such as using hash-based or sort-based aggregation operators. Therefore, simplification can be disabled by
    setting the `simplify` parameter to *False*.

    Parameters
    ----------
    query_plan : QueryPlan
        The plan to update.
    operators : Optional[PhysicalOperatorAssignment], optional
        The new operators to use. This can be a partial assignment, in which case only the operators that are present in the
        new assignment are used and all others are left unchanged. If this parameter is not given, no operators are updated.
    params : Optional[PlanParameterization], optional
        The new parameters to use. This can be a partial assignment, in which case only the cardinalities/parallel workers in
        the new assignment are used and all others are left unchanged. If this parameter is not given, no parameters are
        updated.
    simplify : bool, optional
        Whether to simplify the query plan before updating it. For a detailed discussion, see the high-level documentatio of
        this method. Simplifications is enabled by default.

    Returns
    -------
    QueryPlan
        The updated query plan

    See Also
    --------
    QueryPlan.simplify
    """
    query_plan = query_plan.canonical() if simplify else query_plan

    updated_operator = (
        operators.get(query_plan.tables(), query_plan.operator)
        if operators
        else query_plan.operator
    )
    updated_card_est = (
        params.cardinalities.get(query_plan.tables(), query_plan.estimated_cardinality)
        if params
        else query_plan.estimated_cardinality
    )
    updated_workers = (
        params.parallel_workers.get(
            query_plan.tables(), query_plan.params.parallel_workers
        )
        if params
        else query_plan.params.parallel_workers
    )

    updated_params = PlanParams(
        **(query_plan.params.items() | {"parallel_workers": updated_workers})
    )
    updated_estimates = PlanEstimates(
        **(query_plan.estimates.items() | {"estimated_cardinality": updated_card_est})
    )
    updated_children = [
        update_plan(child, operators=operators, params=params)
        for child in query_plan.children
    ]

    return QueryPlan(
        query_plan.node_type,
        operator=updated_operator,
        children=updated_children,
        plan_params=updated_params,
        estimates=updated_estimates,
        measures=query_plan.measures,
        subplan=query_plan.subplan,
    )


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
