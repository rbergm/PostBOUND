from __future__ import annotations

import collections
import math
from collections.abc import Callable, Iterable, Sequence
from dataclasses import dataclass
from numbers import Number
from typing import Any, Literal, Optional

from . import util
from ._core import Cardinality, Cost, ScanOperators, JoinOperators, PhysicalOperator, TableReference, ColumnReference
from .qal import SqlExpression, AbstractPredicate
from .util import jsondict


TraversalDirection = Literal["inner", "outer"]


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


class PlanParams:

    def __init__(self, *, base_table: Optional[TableReference] = None,
                 filter_predicate: Optional[AbstractPredicate] = None,
                 sort_keys: Optional[Sequence[SortKey]] = None,
                 parallel_workers: Optional[int] = None,
                 index: Optional[str] = None,
                 **kwargs) -> None:
        self._params: dict[str, Any] = {
            "base_table": base_table,
            "filter_predicate": filter_predicate,
            "sort_keys": sort_keys,
            "parallel_workers": parallel_workers,
            "index": index,
            **kwargs
        }

    @property
    def base_table(self) -> Optional[TableReference]:
        return self._params["base_table"]

    @property
    def filter_predicate(self) -> Optional[AbstractPredicate]:
        return self._params["filter_predicate"]

    @property
    def sort_keys(self) -> Sequence[SortKey]:
        return self._params["sort_keys"]

    @property
    def parallel_workers(self) -> Optional[int]:
        return self._params["parallel_workers"]

    @property
    def index(self) -> Optional[str]:
        return self._params["index"]

    def tables(self) -> set[TableReference]:
        tables = set()
        if self.base_table:
            tables.add(self.base_table)
        if self.filter_predicate:
            tables |= self.filter_predicate.tables()
        return tables

    def columns(self) -> set[ColumnReference]:
        return self.filter_predicate.columns() if self.filter_predicate else set()

    def get(self, key: str, default: Any = None) -> Any:
        value = self._params.get(key, default)
        if isinstance(value, float) and math.isnan(value):
            return default
        return value

    def __getattribute__(self, name: str) -> Any:
        if hasattr(self, name):
            return object.__getattribute__(self, name)
        if name in self._params:
            return self._params[name]
        raise AttributeError(f"PlanParams object has no attribute '{name}'")

    def __getitem__(self, key: str) -> Any:
        return self._params[key]

    def __setitem__(self, key: str, value: Any) -> None:
        self._params[key] = value


class PlanEstimates:

    def __init__(self, *,
                 cardinality: Cardinality = math.nan,
                 cost: Cost = math.nan) -> None:
        self._params = {
            "cardinality": cardinality,
            "cost": cost
        }

    @property
    def cardinality(self) -> Cardinality:
        return self._params["cardinality"]

    @property
    def cost(self) -> Cost:
        return self._params["cost"]

    def get(self, key: str, default: Any = None) -> Any:
        value = self._params.get(key, default)
        if isinstance(value, float) and math.isnan(value):
            return default
        return value

    def __getattribute__(self, name: str) -> Any:
        if hasattr(self, name):
            return object.__getattribute__(self, name)
        if name in self._params:
            return self._params[name]
        raise AttributeError(f"PlanEstimates object has no attribute '{name}'")

    def __getitem__(self, key: str) -> Any:
        return self._params[key]


class PlanMeasures:

    def __init__(self, *,
                 cardinality: Cardinality = None,
                 execution_time: float = math.nan,
                 cache_hits: Optional[int] = None,
                 cache_misses: Optional[int] = None) -> None:
        self._params = {
            "cardinality": cardinality,
            "execution_time": execution_time,
            "cache_hits": cache_hits,
            "cache_misses": cache_misses
        }

    @property
    def cardinality(self) -> Cardinality:
        return self._params["cardinality"]

    @property
    def execution_time(self) -> float:
        return self._params["execution_time"]

    @property
    def cache_hits(self) -> Optional[int]:
        return self._params["cache_hits"]

    @property
    def cache_misses(self) -> Optional[int]:
        return self._params["cache_misses"]

    def get(self, key: str, default: Any = None) -> Any:
        value = self._params.get(key, default)
        if isinstance(value, float) and math.isnan(value):
            return default
        return value

    def __getattribute__(self, name: str) -> Any:
        if hasattr(self, name):
            return object.__getattribute__(self, name)
        if name in self._params:
            return self._params[name]
        raise AttributeError(f"PlanMeasures object has no attribute '{name}'")

    def __getitem__(self, key: str) -> Any:
        return self._params[key]

    def __bool__(self) -> bool:
        return any(not math.isnan(v) if isinstance(v, float) else (v is not None) for v in self._params.values())


@dataclass(frozen=True)
class Subplan:
    root: QueryPlan
    target_name: str = ""

    def tables(self) -> set[TableReference]:
        target_table = TableReference.create_virtual(self.target_name)
        return self.root.tables() | {target_table}


class QueryPlan:
    def __init__(self, node_type: str, *, operator: Optional[PhysicalOperator] = None,
                 input_node: Optional[QueryPlan] = None, children: Optional[Iterable[QueryPlan]] = None,
                 plan_params: Optional[PlanParams] = None, subplan: Optional[Subplan] = None,
                 estimates: Optional[PlanEstimates] = None, measures: Optional[PlanMeasures] = None,
                 base_table: Optional[TableReference] = None, filter_predicate: Optional[AbstractPredicate] = None,
                 parallel_workers: Optional[int] = None, index: Optional[str] = None,
                 sort_keys: Optional[Sequence[SortKey]] = None,
                 estimated_cardinality: Cardinality = math.nan, estimated_cost: Cost = math.nan,
                 actual_cardinality: Cardinality = math.nan, execution_time: float = math.nan,
                 cache_hits: Optional[int] = None, cache_misses: Optional[int] = None,
                 **kwargs) -> None:
        if not node_type:
            raise ValueError("Node type must be provided")

        custom_params = (base_table, filter_predicate, parallel_workers, index, sort_keys)
        has_custom_params = any(v is not None for v in custom_params) or bool(kwargs)
        if plan_params is not None and has_custom_params:
            raise ValueError("PlanParams and individual parameters/kwargs cannot be provided at the same time")
        if plan_params is None:
            plan_params = PlanParams(base_table=base_table, filter_predicate=filter_predicate,
                                     sort_keys=sort_keys, parallel_workers=parallel_workers, index=index, **kwargs)

        if estimates is not None and any(v is not None for v in (estimated_cardinality, estimated_cost)):
            raise ValueError("PlanEstimates and individual estimates cannot be provided at the same time")
        if estimates is None:
            estimates = PlanEstimates(cardinality=estimated_cardinality, cost=estimated_cost)

        if measures is not None and any(v is not None for v in (execution_time, cache_hits, cache_misses)):
            raise ValueError("PlanMeasures and individual measures cannot be provided at the same time")
        if measures is None:
            measures = PlanMeasures(execution_time=execution_time, cardinality=actual_cardinality,
                                    cache_hits=cache_hits, cache_misses=cache_misses)

        if len(children) > 2:
            raise ValueError("Query plan nodes can have at most two children")

        self._node_type = node_type
        self._operator = operator
        self._input_node = input_node
        self._children = tuple(children) if children else ()
        self._plan_params = plan_params
        self._estimates = estimates
        self._measures = measures
        self._subplan = subplan

    @property
    def node_type(self) -> str:
        return self._node_type

    @property
    def operator(self) -> Optional[PhysicalOperator]:
        return self._operator

    @property
    def input_node(self) -> Optional[QueryPlan]:
        return self._input_node

    @property
    def children(self) -> Sequence[QueryPlan]:
        return self._children

    @property
    def outer_child(self) -> Optional[QueryPlan]:
        if len(self._children) == 2:
            return self._children[0]
        return None

    @property
    def inner_child(self) -> Optional[QueryPlan]:
        if len(self._children) == 2:
            return self._children[1]
        return None

    @property
    def params(self) -> PlanParams:
        return self._plan_params

    @property
    def base_table(self) -> Optional[TableReference]:
        return self._plan_params.base_table

    @property
    def filter_predicate(self) -> Optional[AbstractPredicate]:
        return self._plan_params.filter_predicate

    @property
    def estimates(self) -> PlanEstimates:
        return self._estimates

    @property
    def estimated_cardinality(self) -> Cardinality:
        return self._estimates.cardinality

    @property
    def estimated_cost(self) -> Cost:
        return self._estimates.cost

    @property
    def measures(self) -> PlanMeasures:
        return self._measures

    @property
    def actual_cardinality(self) -> Cardinality:
        return self._measures.cardinality

    @property
    def execution_time(self) -> float:
        return self._measures.execution_time

    @property
    def subplan(self) -> Optional[Subplan]:
        return self._subplan

    def is_join(self) -> bool:
        return self._operator in JoinOperators

    def is_scan(self) -> bool:
        return self._operator in ScanOperators

    def is_analyze(self) -> bool:
        return bool(self._measures)

    def is_ordered(self) -> bool:
        return bool(self._plan_params.sort_keys)

    def is_linear(self) -> bool:
        if self.is_scan():
            return True
        outer_join = self.outer_child.find_first_node(QueryPlan.is_join)
        inner_join = self.inner_child.find_first_node(QueryPlan.is_join)
        return outer_join is None or inner_join is None

    def is_bushy(self) -> bool:
        if self.is_scan():
            return False
        outer_join = self.outer_child.find_first_node(QueryPlan.is_join)
        inner_join = self.inner_child.find_first_node(QueryPlan.is_join)
        return outer_join is not None and inner_join is not None

    def is_left_deep(self) -> bool:
        if self.is_scan():
            return True
        inner_join = self.inner_child.find_first_node(QueryPlan.is_join)
        return inner_join is None

    def is_right_deep(self) -> bool:
        if self.is_scan():
            return True
        outer_join = self.outer_child.find_first_node(QueryPlan.is_join)
        return outer_join is None

    def is_zigzag(self) -> bool:
        return self.is_linear()

    def is_scan_branch(self) -> bool:
        return self.is_scan() or all(child.is_scan_branch() for child in self.children)

    def is_base_join(self) -> bool:
        if not self.is_join():
            return False
        return all(child.is_scan_branch() for child in self.children)

    def fetch_base_table(self) -> Optional[TableReference]:
        if self.is_scan():
            return self.base_table
        elif self.is_join():
            return None

        if len(self.children) == 1:
            return self.children[0].fetch_base_table()
        return None

    def tables(self) -> set[TableReference]:
        subplan_tabs = self._subplan.tables() if self._subplan else set()
        return self._plan_params.tables() | util.set_union(c.tables() for c in self._children) | subplan_tabs

    def columns(self) -> set[ColumnReference]:
        subplan_cols = self._subplan.root.columns() if self._subplan else set()
        return self._plan_params.columns() | util.set_union(c.columns() for c in self._children) | subplan_cols

    def iternodes(self) -> Iterable[QueryPlan]:
        nodes = [self]
        for child in self._children:
            nodes.extend(child.children)
        return nodes

    def lookup(self, tables: TableReference | Iterable[TableReference]) -> Optional[QueryPlan]:
        needle: set[TableReference] = set(util.enlist(tables))
        candidates = self.tables()

        if needle == candidates:
            return self
        if not needle.issubset(candidates):
            return None

        for child in self.children:
            result = child.lookup(needle)
            if result is not None:
                return result

        return None

    def find_first_node(self, predicate: Callable[[QueryPlan], bool], *args,
                        direction: TraversalDirection = "outer", **kwargs) -> Optional[QueryPlan]:
        if predicate(self, *args, **kwargs):
            return self
        if self.is_scan():
            return None

        first_candidate, second_candidate = (self.outer_child, self.inner_child if direction == "outer"
                                             else self.inner_child, self.outer_child)
        first_match = first_candidate.find_first_node(predicate, *args, direction=direction, **kwargs)
        if first_match:
            return first_match
        return second_candidate.find_first_node(predicate, *args, direction=direction, **kwargs)

    def find_all_nodes(self, predicate: Callable[[QueryPlan], bool], *args, **kwargs) -> Iterable[QueryPlan]:
        matches: list[QueryPlan] = [self] if predicate(self, *args, **kwargs) else []
        for child in self._children:
            matches.extend(child.find_all_nodes(predicate, *args, **kwargs))
        return matches

    def cout(self) -> Optional[int]:
        if not self.is_analyze():
            return None
        return self.actual_cardinality + sum(c.cout() for c in self.children)

    def qerror(self) -> float:
        if not self.is_analyze():
            return math.nan

        larger = max(self.estimated_cardinality, self.actual_cardinality) + 1
        smaller = min(self.estimated_cardinality, self.actual_cardinality) + 1
        return larger / smaller

    def inspect(self, *, fields: Optional[Iterable[str]] = None) -> str:
        fields = [] if fields is None else list(fields)
        return _explainify(self, fields=fields)

    def plan_summary(self) -> dict[str, object]:
        all_nodes = list(self.iternodes())
        summary: dict[str, object] = {
            "operator": self.node_type,
            "intermediate": " ⋈ ".join(str(t) for t in self.tables()),
            "estimated_card": round(self.estimated_cardinality, 3),
            "actual_card": round(self.actual_cardinality, 3),
            "estimated_cost": round(self.estimated_cost, 3),
            "c_out": self.cout(),
            "max_qerror": round(max(node.qerror() for node in all_nodes), 3),
            "avg_qerror": round(sum(node.qerror() for node in all_nodes) / len(all_nodes), 3),
            "phys_ops": collections.Counter(child.node_type for child in all_nodes)
        }
        return summary

    def ast(self) -> str:
        return _astify(self)

    def __json__(self) -> jsondict:
        pass

    def __eq__(self, other: object) -> bool:
        pass

    def __hash__(self) -> int:
        pass

    def __repr__(self) -> str:
        return str(self)

    def __str__(self) -> str:
        normalized_node_type = self.node_type.replace(" ", "")
        if self.base_table:
            return f"{normalized_node_type}({self.base_table.identifier()})"
        child_texts = ", ".join(str(child) for child in self.children)
        return f"{normalized_node_type}({child_texts})"


_starting_indentation = 0


def _default_explain(plan: QueryPlan, *, padding: str) -> str:
    components: list[str] = []

    estimated_card = round(plan.estimated_cardinality, 3)
    estimated_cost = round(plan.estimated_cost, 3)
    components.append(f"{padding}   Estimated Cardinality={estimated_card}, Estimated Cost={estimated_cost}")

    if plan.is_analyze():
        actual_card = round(plan.actual_cardinality, 3)
        exec_time = round(plan.execution_time, 3)
        components.append(f"{padding}   Actual Cardinality={actual_card}, Actual Time={exec_time}s")

    measures = plan.measures
    if measures.cache_hits is not None or measures.cache_misses is not None:
        cache_hits = measures.cache_hits if measures.cache_hits is not None else math.nan
        cache_misses = measures.cache_misses if measures.cache_misses is not None else math.nan
        components.append(f"{padding}   Cache Hits={cache_hits}, Cache Misses={cache_misses}")

    params = plan.params
    if params.parallel_workers is not None:
        components.append(f"{padding}   Parallel Workers={params.parallel_workers}")

    path_props: list[str] = []
    if params.index:
        path_props.append(f"Index={params.index}")
    if params.sort_keys:
        sort_keys = ", ".join(str(key) for key in params.sort_keys)
        path_props.append(f"Sort Keys={sort_keys}")
    if path_props:
        components.append(f"{padding}   {', '.join(path_props)}")

    return "\n".join(components)


def _custom_explain(plan: QueryPlan, *, fields: list[str], padding: str) -> str:
    attr_values: dict[str, str] = {}
    for attr in fields:
        if "." in attr:
            container_name, attr_name = attr.split(".")
            container = getattr(plan, container_name)
            value = getattr(container, attr_name)
        else:
            value = getattr(plan, attr)

        attr_values[attr] = str(round(value, 3)) if isinstance(value, Number) else str(value)

    attr_str = " ".join(f"{attr}={val}" for attr, val in attr_values.items())
    explain_data = f"{padding}   [{attr_str}]"
    return explain_data


def _explainify(plan: QueryPlan, *, fields: list[str], indentation: int = _starting_indentation) -> str:
    padding = " " * indentation
    prefix = f"{padding}<- " if padding else ""

    header = f"{plan.node_type}({plan.base_table})" if plan.is_scan() else plan.node_type
    explain_data = _custom_explain(plan, fields=fields, padding=padding) if fields else _default_explain(plan, padding=padding)
    child_explains = "\n".join(f"{_explainify(child, fields=fields, indentation=indentation + 2)}" for child in plan.children)

    if not child_explains:
        return f"{prefix}{header}\n{explain_data}"
    return f"{prefix}{header}\n{explain_data}\n{child_explains}"


def _astify(plan: QueryPlan, *, indentation: int = _starting_indentation) -> str:
    prefix = " " * indentation
    if plan.is_scan():
        item_str = f"{prefix}<- {plan.node_type}({plan.base_table})"
    else:
        item_str = f"{prefix}<- {plan.node_type}"
    child_str = "\n".join(_astify(child, indentation=indentation + 2) for child in plan.children)
    return f"{item_str}\n{child_str}"
