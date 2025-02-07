from __future__ import annotations

import math
from collections.abc import Callable, Iterable, Sequence
from dataclasses import dataclass
from typing import Any, Literal, Optional

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
                 cardinality: Optional[Cardinality] = None,
                 cost: Cost = math.nan) -> None:
        self._params = {
            "cardinality": cardinality,
            "cost": cost
        }

    @property
    def cardinality(self) -> Optional[Cardinality]:
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
                 cardinality: Optional[Cardinality] = None,
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
    def cardinality(self) -> Optional[Cardinality]:
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
                 estimated_cardinality: Optional[Cardinality] = None, estimated_cost: Cost = math.nan,
                 actual_cardinality: Optional[Cardinality] = None, execution_time: float = math.nan,
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
    def estimated_cardinality(self) -> Optional[Cardinality]:
        return self._estimates.cardinality

    @property
    def estimated_cost(self) -> Cost:
        return self._estimates.cost

    @property
    def measures(self) -> PlanMeasures:
        return self._measures

    @property
    def actual_cardinality(self) -> Optional[Cardinality]:
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

    def tables(self) -> set[TableReference]:
        pass

    def columns(self) -> set[ColumnReference]:
        pass

    def find_first_node(self, predicate: Callable[[QueryPlan], bool], *,
                        direction: TraversalDirection = "outer") -> Optional[QueryPlan]:
        pass

    def find_all_nodes(self, predicate: Callable[[QueryPlan], bool]) -> Iterable[QueryPlan]:
        pass

    def __json__(self) -> jsondict:
        pass
