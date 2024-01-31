
from __future__ import annotations

import abc
import collections
import dataclasses
import enum
import typing
from collections.abc import Iterable, Sequence
from typing import Optional

import networkx as nx

from postbound.qal import base, expressions as expr, predicates as preds, qal
from postbound.util import collections as collection_utils, dicts as dict_utils, networkx as nx_utils


def _collect_all_expressions(base_expressions: Iterable[expr.SqlExpression]) -> set[expr.SqlExpression]:
    all_expressions = set()
    for expression in base_expressions:
        all_expressions.add(expression)
        all_expressions |= _collect_all_expressions(expression.iterchildren())
    return all_expressions


class RelNode(abc.ABC):
    def __init__(self, parent_node: Optional[RelNode]) -> None:
        self._parent = parent_node
        self._node_type = type(self).__name__

    @property
    def node_type(self) -> str:
        return self._node_type

    @property
    def parent_node(self) -> Optional[RelNode]:
        return self._parent

    @abc.abstractmethod
    def children(self) -> Sequence[RelNode]:
        raise NotImplementedError

    def tables(self) -> frozenset[base.TableReference]:
        return frozenset(collection_utils.set_union(child.tables() for child in self.children()))

    def provided_expressions(self) -> frozenset[expr.SqlExpression]:
        """Collects all expressions that are available to parent nodes.

        These expressions will contain all expressions that are provided by child nodes as well as all expressions that are
        calculated by the current node itself.

        Returns
        -------
        frozenset[expressions.SqlExpression]
            _description_
        """
        return collection_utils.set_union(child.provided_expressions() for child in self.children())

    def required_expressions(self) -> frozenset[expr.SqlExpression]:
        """Collects all expressions that are not computed by the current node and therefore must be provided by its children.

        Returns
        -------
        frozenset[expressions.SqlExpression]
            _description_
        """
        return frozenset()

    def inspect(self, indentation: int = 0) -> str:
        padding = " " * indentation
        prefix = f"{padding}<- " if padding else ""
        inspections = [prefix + str(self)]
        for child in self.children():
            inspections.append(child.inspect(indentation + 2))
        return "\n".join(inspections)

    def _maintain_child_links(self) -> None:
        for child in self.children():
            child._parent = self

    @abc.abstractmethod
    def __hash__(self) -> int:
        raise NotImplementedError

    @abc.abstractmethod
    def __eq__(self, other: object) -> bool:
        raise NotImplementedError

    def __repr__(self) -> str:
        return str(self)

    @abc.abstractmethod
    def __str__(self) -> str:
        raise NotImplementedError


class Selection(RelNode):
    def __init__(self, input_node: RelNode, predicate: preds.AbstractPredicate, *,
                 parent_node: Optional[RelNode] = None) -> None:
        super().__init__(parent_node)
        self._input_node = input_node
        self._predicate = predicate
        self._hash_val = hash((self._input_node, self._predicate))
        self._maintain_child_links()

    @property
    def input_node(self) -> RelNode:
        return self._input_node

    @property
    def predicate(self) -> preds.AbstractPredicate:
        return self._predicate

    def children(self) -> Sequence[RelNode]:
        return [self._input_node]

    def required_expressions(self) -> frozenset[expr.SqlExpression]:
        return _collect_all_expressions(self._predicate.iterexpressions())

    def __hash__(self) -> int:
        return self._hash_val

    def __eq__(self, other: object) -> bool:
        return isinstance(other, type(self)) and self._input_node == other._input_node and self._predicate == other._predicate

    def __str__(self) -> str:
        return f"σ ({self._predicate})"


class CrossProduct(RelNode):
    def __init__(self, left_input: RelNode, right_input: RelNode, *, parent_node: Optional[RelNode] = None) -> None:
        super().__init__(parent_node)
        self._left_input = left_input
        self._right_input = right_input
        self._hash_val = hash((self._left_input, self._right_input))
        self._maintain_child_links()

    @property
    def left_input(self) -> RelNode:
        return self._left_input

    @property
    def right_input(self) -> RelNode:
        return self._right_input

    def children(self) -> Sequence[RelNode]:
        return [self._left_input, self._right_input]

    def __hash__(self) -> int:
        return self._hash_val

    def __eq__(self, other: object) -> bool:
        return (isinstance(other, type(self))
                and self._left_input == other._left_input
                and self._right_input == other._right_input)

    def __str__(self) -> str:
        return "⨯"


class Union(RelNode):
    def __init__(self, left_input: RelNode, right_input: RelNode, *, parent_node: Optional[RelNode] = None) -> None:
        super().__init__(parent_node)
        self._left_input = left_input
        self._right_input = right_input
        self._hash_val = hash((self._left_input, self._right_input))
        self._maintain_child_links()

    @property
    def left_input(self) -> RelNode:
        return self._left_input

    @property
    def right_input(self) -> RelNode:
        return self._right_input

    def children(self) -> Sequence[RelNode]:
        return [self._left_input, self._right_input]

    def __hash__(self) -> int:
        return self._hash_val

    def __eq__(self, other: object) -> bool:
        return (isinstance(other, type(self))
                and self._left_input == other._left_input
                and self._right_input == other._right_input)

    def __str__(self) -> str:
        return "∪"


class Intersection(RelNode):
    def __init__(self, left_input: RelNode, right_input: RelNode, *, parent_node: Optional[RelNode] = None) -> None:
        super().__init__(parent_node)
        self._left_input = left_input
        self._right_input = right_input
        self._hash_val = hash((self._left_input, self._right_input))
        self._maintain_child_links()

    @property
    def left_input(self) -> RelNode:
        return self._left_input

    @property
    def right_input(self) -> RelNode:
        return self._right_input

    def children(self) -> Sequence[RelNode]:
        return [self._left_input, self._right_input]

    def __hash__(self) -> int:
        return self._hash_val

    def __eq__(self, other: object) -> bool:
        return (isinstance(other, type(self))
                and self._left_input == other._left_input
                and self._right_input == other._right_input)

    def __str__(self) -> str:
        return "∩"


class Difference(RelNode):
    def __init__(self, left_input: RelNode, right_input: RelNode, *, parent_node: Optional[RelNode] = None) -> None:
        super().__init__(parent_node)
        self._left_input = left_input
        self._right_input = right_input
        self._hash_val = hash((self._left_input, self._right_input))
        self._maintain_child_links()

    @property
    def left_input(self) -> RelNode:
        return self._left_input

    @property
    def right_input(self) -> RelNode:
        return self._right_input

    def children(self) -> Sequence[RelNode]:
        return [self._left_input, self._right_input]

    def __hash__(self) -> int:
        return self._hash_val

    def __eq__(self, other: object) -> bool:
        return (isinstance(other, type(self))
                and self._left_input == other._left_input
                and self._right_input == other._right_input)

    def __str__(self) -> str:
        return "\\"


class Table(RelNode):
    def __init__(self, table: base.TableReference, provided_columns: Iterable[base.ColumnReference | expr.ColumnExpression], *,
                 parent_node: Optional[RelNode] = None) -> None:
        super().__init__(parent_node)
        self._table = table
        self._provided_cols = frozenset(col if isinstance(col, expr.ColumnExpression) else expr.ColumnExpression(col)
                                        for col in provided_columns)
        self._hash_val = hash(self._table)
        self._maintain_child_links()

    @property
    def table(self) -> base.TableReference:
        return self._table

    def children(self) -> Sequence[RelNode]:
        return []

    def tables(self) -> frozenset[base.TableReference]:
        return frozenset((self._table,))

    def provided_expressions(self) -> frozenset[expr.SqlExpression]:
        return self._provided_cols

    def __hash__(self) -> int:
        return self._hash_val

    def __eq__(self, other: object) -> bool:
        return isinstance(other, type(self)) and self._table == other._table

    def __str__(self) -> str:
        return self._table.identifier()


class ThetaJoin(RelNode):
    def __init__(self, left_input: RelNode, right_input: RelNode, predicate: preds.AbstractPredicate, *,
                 parent_node: Optional[RelNode] = None) -> None:
        super().__init__(parent_node)
        self._left_input = left_input
        self._right_input = right_input
        self._predicate = predicate
        self._hash_val = hash((self._left_input, self._right_input, self._predicate))
        self._maintain_child_links()

    @property
    def left_input(self) -> RelNode:
        return self._left_input

    @property
    def right_input(self) -> RelNode:
        return self._right_input

    @property
    def predicate(self) -> preds.AbstractPredicate:
        return self._predicate

    def children(self) -> Sequence[RelNode]:
        return [self._left_input, self._right_input]

    def required_expressions(self) -> frozenset[expr.SqlExpression]:
        return _collect_all_expressions(self._predicate.iterexpressions())

    def __hash__(self) -> int:
        return self._hash_val

    def __eq__(self, other: object) -> bool:
        return (isinstance(other, type(self))
                and self._left_input == other._left_input and self._right_input == other._right_input
                and self._predicate == other._predicate)

    def __str__(self) -> str:
        return f"⋈ ϴ=({self._predicate})"


class Projection(RelNode):
    def __init__(self, input_node: RelNode, targets: Sequence[expr.SqlExpression], *,
                 parent_node: Optional[RelNode] = None) -> None:
        super().__init__(parent_node)
        self._input_node = input_node
        self._targets = tuple(targets)
        self._hash_val = hash((self._input_node, self._targets))
        self._maintain_child_links()

    @property
    def input_node(self) -> RelNode:
        return self._input_node

    @property
    def columns(self) -> Sequence[expr.SqlExpression]:
        return self._targets

    def children(self) -> Sequence[RelNode]:
        return [self._input_node]

    def required_expressions(self) -> frozenset[expr.SqlExpression]:
        return _collect_all_expressions(self._targets)

    def __hash__(self) -> int:
        return self._hash_val

    def __eq__(self, other: object) -> bool:
        return isinstance(other, type(self)) and self._input_node == other._input_node and self._targets == other._targets

    def __str__(self) -> str:
        col_str = ", ".join(str(col) for col in self._targets)
        return f"π ({col_str})"


class GroupBy(RelNode):
    def __init__(self, input_node: RelNode, group_columns: Sequence[expr.SqlExpression], *,
                 aggregates: Optional[dict[expr.SqlExpression, expr.FunctionExpression]] = None,
                 parent_node: Optional[RelNode] = None) -> None:
        super().__init__(parent_node)
        self._input_node = input_node
        self._group_columns = tuple(group_columns)
        self._aggregates = dict_utils.frozendict(aggregates)
        self._hash_val = hash((self._input_node, self._group_columns, self._aggregates))
        self._maintain_child_links()

    @property
    def input_node(self) -> RelNode:
        return self._input_node

    @property
    def group_columns(self) -> Sequence[expr.SqlExpression]:
        return self._group_columns

    @property
    def aggregates(self) -> dict_utils.frozendict[expr.SqlExpression, expr.FunctionExpression]:
        return self._aggregates

    def children(self) -> Sequence[RelNode]:
        return [self._input_node]

    def required_expressions(self) -> frozenset[expr.SqlExpression]:
        return frozenset(self._aggregates.keys())

    def provided_expressions(self) -> frozenset[expr.SqlExpression]:
        return super().provided_expressions() | frozenset(self._aggregates.values())

    def __hash__(self) -> int:
        return self._hash_val

    def __eq__(self, other: object) -> bool:
        return (isinstance(other, type(self))
                and self._input_node == other._input_node
                and self._group_columns == other._group_columns
                and self._aggregates == other._aggregates)

    def __str__(self) -> str:
        agg_str = ", ".join(f"{col}: {agg_func}" for col, agg_func in self._aggregates.items())
        if not self._group_columns:
            return f"γ ({agg_str})"
        group_str = ", ".join(str(col) for col in self._group_columns)
        return f"{group_str} γ ({agg_str})"


class Rename(RelNode):
    def __init__(self, input_node: RelNode, mapping: dict[base.ColumnReference, base.ColumnReference], *,
                 parent_node: Optional[RelNode]) -> None:
        # TODO: check types + add provided / required expressions method
        super().__init__(parent_node)
        self._input_node = input_node
        self._mapping = dict_utils.frozendict(mapping)
        self._hash_val = hash((self._input_node, self._mapping))
        self._maintain_child_links()

    @property
    def input_node(self) -> RelNode:
        return self._input_node

    @property
    def mapping(self) -> dict_utils.frozendict[base.ColumnReference, base.ColumnReference]:
        return self._mapping

    def children(self) -> Sequence[RelNode]:
        return [self._input_node]

    def __hash__(self) -> int:
        return self._hash_val

    def __eq__(self, other: object) -> bool:
        return isinstance(other, type(self)) and self._input_node == other._input_node and self._mapping == other._mapping

    def __str__(self) -> str:
        map_str = ", ".join(f"{col}: {target}" for col, target in self._mapping)
        return f"ϱ ({map_str})"


SortDirection = typing.Literal["asc", "desc"]


class Sort(RelNode):
    def __init__(self, input_node: RelNode,
                 sorting: Sequence[tuple[expr.SqlExpression, SortDirection] | expr.SqlExpression], *,
                 parent_node: Optional[RelNode] = None) -> None:
        super().__init__(parent_node)
        self._input_node = input_node
        self._sorting = tuple([sort_item if isinstance(sort_item, tuple) else (sort_item, "asc") for sort_item in sorting])
        self._hash_val = hash((self._input_node, self._sorting))
        self._maintain_child_links()

    @property
    def input_node(self) -> RelNode:
        return self._input_node

    @property
    def sorting(self) -> Sequence[tuple[expr.SqlExpression, SortDirection]]:
        return self._sorting

    def children(self) -> Sequence[RelNode]:
        return [self._input_node]

    def required_expressions(self) -> frozenset[expr.SqlExpression]:
        return frozenset(sorting[0] for sorting in self._sorting)

    def __hash__(self) -> int:
        return self._hash_val

    def __eq__(self, other: object) -> bool:
        return isinstance(other, type(self)) and self._input_node == other._input_node and self._sorting == other._sorting

    def __str__(self) -> str:
        sorting_str = ", ".join(f"{sort_col}{'↓' if sort_dir == 'asc' else '↑'}" for sort_col, sort_dir in self._sorting)
        return f"τ ({sorting_str})"


class Map(RelNode):
    def __init__(self, input_node: RelNode, mapping: dict[frozenset[expr.SqlExpression], frozenset[expr.SqlExpression]], *,
                 parent_node: Optional[RelNode] = None) -> None:
        super().__init__(parent_node)
        self._input_node = input_node
        self._mapping = dict_utils.frozendict(mapping)
        self._hash_val = hash((self._input_node, self._mapping))
        self._maintain_child_links()

    @property
    def input_node(self) -> RelNode:
        return self._input_node

    @property
    def mapping(self) -> dict_utils.frozendict[frozenset[expr.SqlExpression], frozenset[expr.SqlExpression]]:
        return self._mapping

    def children(self) -> Sequence[RelNode]:
        return [self._input_node]

    def required_expressions(self) -> frozenset[expr.SqlExpression]:
        return collection_utils.set_union(map_source for map_source in self._mapping.keys())

    def provided_expressions(self) -> frozenset[expr.SqlExpression]:
        return super().provided_expressions() | collection_utils.set_union(map_target for map_target in self._mapping.values())

    def __hash__(self) -> int:
        return self._hash_val

    def __eq__(self, other: object) -> bool:
        return isinstance(other, type(self)) and self._input_node == other._input_node and self._mapping == other._mapping

    def __str__(self) -> str:
        mapping_str = ", ".join(f"{target_col}: {expr}" for target_col, expr in self._mapping.items())
        return f"χ ({mapping_str})"


class DuplicateElimination(RelNode):
    def __init__(self, input_node: RelNode, *, parent_node: Optional[RelNode] = None) -> None:
        super().__init__(parent_node)
        self._input_node = input_node
        self._hash_val = hash((self._input_node))
        self._maintain_child_links()

    @property
    def input_node(self) -> RelNode:
        return self._input_node

    def children(self) -> Sequence[RelNode]:
        return [self._input_node]

    def __hash__(self) -> int:
        return self._hash_val

    def __eq__(self, other: object) -> bool:
        return isinstance(other, type(self)) and self._input_node == other._input_node

    def __str__(self) -> str:
        return "δ"


class SemiJoin(RelNode):
    def __init__(self, left_input: RelNode, right_input: RelNode, predicate: Optional[preds.AbstractPredicate] = None, *,
                 parent_node: Optional[RelNode] = None) -> None:
        super().__init__(parent_node)
        self._left_input = left_input
        self._right_input = right_input
        self._predicate = predicate
        self._hash_val = hash((self._left_input, self._right_input, self._predicate))
        self._maintain_child_links()

    @property
    def left_input(self) -> RelNode:
        return self._left_input

    @property
    def right_input(self) -> RelNode:
        return self._right_input

    @property
    def predicate(self) -> Optional[preds.AbstractPredicate]:
        return self._predicate

    def children(self) -> Sequence[RelNode]:
        return [self._left_input, self._right_input]

    def required_expressions(self) -> frozenset[expr.SqlExpression]:
        return _collect_all_expressions(self._predicate.iterexpressions())

    def __hash__(self) -> int:
        return self._hash_val

    def __eq__(self, other: object) -> bool:
        return (isinstance(other, type(self))
                and self._left_input == other._left_input and self._right_input == other._right_input
                and self._predicate == other._predicate)

    def __str__(self) -> str:
        return "⋉" if self._predicate is None else f"⋉ ({self._predicate})"


class AntiJoin(RelNode):
    def __init__(self, left_input: RelNode, right_input: RelNode, predicate: Optional[preds.AbstractPredicate] = None, *,
                 parent_node: Optional[RelNode] = None) -> None:
        super().__init__(parent_node)
        self._left_input = left_input
        self._right_input = right_input
        self._predicate = predicate
        self._hash_val = hash((self._left_input, self._right_input, self._predicate))
        self._maintain_child_links()

    @property
    def left_input(self) -> RelNode:
        return self._left_input

    @property
    def right_input(self) -> RelNode:
        return self._right_input

    @property
    def predicate(self) -> Optional[preds.AbstractPredicate]:
        return self._predicate

    def children(self) -> Sequence[RelNode]:
        return [self._left_input, self._right_input]

    def required_expressions(self) -> frozenset[expr.SqlExpression]:
        return _collect_all_expressions(self._predicate.iterexpressions())

    def __hash__(self) -> int:
        return self._hash_val

    def __eq__(self, other: object) -> bool:
        return (isinstance(other, type(self))
                and self._left_input == other._left_input and self._right_input == other._right_input
                and self._predicate == other._predicate)

    def __str__(self) -> str:
        return "▷" if self._predicate is None else f"▷ ({self._predicate})"


def _is_aggregation(expression: expr.SqlExpression) -> bool:
    return isinstance(expression, expr.FunctionExpression) and expression.is_aggregate()


def _requires_aggregation(expression: expr.SqlExpression) -> bool:
    return any(_is_aggregation(child_expr) or _requires_aggregation(child_expr) for child_expr in expression.iterchildren())


def _requires_join(expression: expr.SqlExpression) -> bool:
    return len(expression.tables()) > 1


def _generate_expression_mapping_dict(expressions: list[expr.SqlExpression]
                                      ) -> dict[frozenset[expr.SqlExpression], frozenset[expr.SqlExpression]]:
    mapping: dict[frozenset[expr.SqlExpression], set[expr.SqlExpression]] = collections.defaultdict(set)
    for expression in expressions:
        child_expressions = frozenset(expression.iterchildren())
        mapping[child_expressions].add(expression)
    return {child_expressions: frozenset(derived_expressions) for child_expressions, derived_expressions in mapping.items()}


def _predicate_contains_subqueries(predicate: preds.AbstractPredicate) -> bool:
    return any(qal.collect_subqueries_in_expression(expression) for expression in predicate.iterexpressions())


def _merge_predicates(current_predicate: Optional[preds.AbstractPredicate],
                      additional_predicate: preds.AbstractPredicate) -> preds.AbstractPredicate:
    if not current_predicate:
        return additional_predicate
    return preds.CompoundPredicate.create_and(current_predicate, additional_predicate)


class EvaluationPhase(enum.IntEnum):
    BaseTable = enum.auto()
    PreJoin = enum.auto()
    Join = enum.auto()
    PostJoin = enum.auto()
    PostAggregation = enum.auto()


@dataclasses.dataclass(frozen=True)
class SemiJoinTask:
    subquery: qal.SqlQuery
    subquery_task: QueryTaskGraph
    root_predicate: preds.AbstractPredicate

    def __str__(self) -> str:
        return f"⋉ [{self.subquery.stringify(trailing_delimiter=False)}]"


@dataclasses.dataclass(frozen=True)
class AntiJoinTask:
    subquery: qal.SqlQuery
    subquery_graph: QueryTaskGraph
    root_predicate: preds.AbstractPredicate

    def __str__(self) -> str:
        return f"▷ [{self.subquery.stringify(trailing_delimiter=False)}]"


@dataclasses.dataclass(frozen=True)
class SubqueryScanTask:
    subquery: qal.SqlQuery
    subquery_graph: QueryTaskGraph
    cte_scan: bool = False

    def __str__(self) -> str:
        scan_type = "CTE Scan" if self.cte_scan else "Subquery Scan"
        return f"<<{scan_type}>> [{self.subquery.stringify(trailing_delimiter=False)}]"


@dataclasses.dataclass(frozen=True)
class StaticValueTask:
    value: expr.StaticValueExpression

    def __str__(self) -> str:
        return f"<<static>> {self.value}"


@dataclasses.dataclass(frozen=True)
class ProjectionTask:
    expressions: Sequence[expr.SqlExpression]

    def __str__(self) -> str:
        return f"Π {list(self.expressions)}"


@dataclasses.dataclass(frozen=True)
class GroupByTask:
    expressions: Sequence[expr.SqlExpression]

    def __str__(self) -> str:
        return f"Γ {list(self.expressions)}"


class QueryTaskGraph:
    def __init__(self, query: qal.SqlQuery, *, skip_static_values: bool = True, _skip_validation: bool = False) -> None:
        self._query = query
        self._graph = nx.DiGraph()
        self._node_translation: dict = {}
        self._groupby: Optional[GroupByTask] = None
        self._skip_static_values = skip_static_values
        self._skip_validation = _skip_validation
        self._init_task_graph()

    @property
    def query(self) -> qal.SqlQuery:
        return self._query

    @property
    def task_graph(self) -> nx.DiGraph:
        return self._graph.copy()

    def graph_root(self) -> object:
        return collection_utils.simplify(nx_utils.nx_sinks(self._graph))

    def _init_task_graph(self) -> None:
        for table in self._query.tables():
            self._node_translation[table] = table
            self._graph.add_node(table, type="base_table")

        if self._query.cte_clause:
            for cte in self._query.cte_clause.queries:
                cte_graph = QueryTaskGraph(cte.query)
                self._merge_subgraph(cte_graph.task_graph)
                cte_scan_node = SubqueryScanTask(cte.query, cte_graph, cte_scan=True)
                self._graph.add_node(cte_scan_node, type="cte_scan")
                self._graph.add_edge(cte_graph.graph_root(), cte_scan_node)
                self._graph.add_edge(cte_scan_node, cte.target_table)

        predicates = self._query.from_clause.predicates()
        if self._query.where_clause:
            predicates = predicates.and_(self._query.where_clause.predicate)
        if predicates:
            self._add_predicate_tasks(predicates.root)

        # TODO: insert cross product for all dangling tables
        # TODO: special cases: cross products, no WHERE clause, ...

        if self._query.groupby_clause:
            group_node = GroupByTask(self._query.groupby_clause.group_columns)
            self._graph.add_node(group_node, type="grouping")
            for group_col in self._query.groupby_clause.group_columns:
                self._add_expression_task(group_col)
                self._graph.add_edge(self._node_translation[group_col], group_node)
            self._groupby = group_node
        if self._query.having_clause:
            self._add_predicate_tasks(self._query.having_clause.condition)

        projection = ProjectionTask(self._query.select_clause.targets)
        self._graph.add_node(projection, type="projection")
        for select_target in self._query.select_clause.targets:
            # TODO: duplicate elimination
            self._add_expression_task(select_target.expression)
            self._graph.add_edge(self._node_translation[select_target.expression], projection)
        if predicates:
            self._graph.add_edge(predicates.root, projection)
        if self._query.having_clause:
            self._graph.add_edge(self._query.having_clause.condition, projection)

        if self._graph.has_node(expr.StarExpression()):
            self._graph.remove_node(expr.StarExpression())
        if self._skip_static_values:
            static_value_nodes = [node for node in self._graph.nodes if isinstance(node, expr.StaticValueExpression)]
            self._graph.remove_nodes_from(static_value_nodes)

        self._assert_valid_graph()

    def _add_predicate_tasks(self, predicate: preds.AbstractPredicate) -> None:
        self._node_translation[predicate] = predicate
        predicate_type = "filter"

        match predicate:
            case preds.CompoundPredicate():
                for child_pred in predicate.iterchildren():
                    self._add_predicate_tasks(child_pred)
                    self._graph.add_edge(child_pred, predicate)

            case preds.BinaryPredicate():
                lhs, rhs = predicate.first_argument, predicate.second_argument
                self._add_expression_task(lhs)
                self._add_expression_task(rhs)
                self._graph.add_edge(self._node_translation[lhs], predicate)
                self._graph.add_edge(self._node_translation[rhs], predicate)
                predicate_type = "filter" if predicate.is_filter() else "join"

            case preds.InPredicate():
                col, values = predicate.column, predicate.values
                self._add_expression_task(col)
                self._graph.add_edge(self._node_translation[col], predicate)
                for val in values:
                    self._add_expression_task(val)
                    self._graph.add_edge(self._node_translation[val], predicate)
                predicate_type = "filter"

            case preds.BetweenPredicate():
                col, lower, upper = predicate.column, predicate.interval_start, predicate.interval_end
                self._add_expression_task(col)
                self._graph.add_edge(self._node_translation[col], predicate)

                self._add_expression_task(lower)
                self._graph.add_edge(self._node_translation[lower], predicate)

                self._add_expression_task(upper)
                self._graph.add_edge(self._node_translation[upper], predicate)

            case preds.UnaryPredicate():
                expression, op = predicate.column, predicate.operation
                if op is None:
                    self._add_expression_task(expression)
                    self._graph.add_edge(self._node_translation[expression], predicate)

                elif op == expr.LogicalSqlOperators.Exists and isinstance(expression, expr.SubqueryExpression):
                    subquery_graph = QueryTaskGraph(expression.query)
                    subquery_result = collection_utils.simplify(nx_utils.nx_sinks(subquery_graph.task_graph))
                    self._merge_subgraph(subquery_graph.task_graph)
                    semi_join_node = SemiJoinTask(expression.query, subquery_graph, predicate)
                    self._node_translation[expression] = semi_join_node
                    self._graph.add_node(semi_join_node, type="semi_join_node")
                    self._graph.add_edge(semi_join_node, predicate)
                    self._graph.add_edge(subquery_result, semi_join_node)

                elif op == expr.LogicalSqlOperators.Missing and isinstance(expression, expr.SubqueryExpression):
                    subquery_graph = QueryTaskGraph(expression.query)
                    subquery_result = collection_utils.simplify(nx_utils.nx_sinks(subquery_graph.task_graph))
                    self._merge_subgraph(subquery_graph.task_graph)
                    anti_join_node = AntiJoinTask(expression.query, subquery_graph, predicate)
                    self._node_translation[expression] = anti_join_node
                    self._graph.add_node(anti_join_node, type="semi_join_node")
                    self._graph.add_edge(anti_join_node, predicate)
                    self._graph.add_edge(subquery_result, anti_join_node)

                else:
                    raise ValueError(f"Unknown operator/expression combination: {predicate}")

            case _:
                raise ValueError(f"Unknown predicate type: '{predicate}'")

        self._graph.add_node(predicate, type=predicate_type)

    def _add_expression_task(self, expression: expr.SqlExpression) -> None:
        match expression:
            case expr.SubqueryExpression():
                subquery_graph = QueryTaskGraph(expression.query)
                subquery_scan_node = SubqueryScanTask(expression.query, subquery_graph)
                self._node_translation[expression] = subquery_scan_node

                self._merge_subgraph(subquery_graph.task_graph)
                self._graph.add_node(subquery_scan_node, type="scalar_subquery_select")

                final_subquery_node = collection_utils.simplify(nx_utils.nx_sinks(subquery_graph.task_graph))
                self._graph.add_edge(final_subquery_node, subquery_scan_node)

            case expr.StaticValueExpression():
                static_value_select = StaticValueTask(expression)
                self._node_translation[expression] = static_value_select
                self._graph.add_node(static_value_select, type="static_value_select")

            case expr.FunctionExpression():
                self._node_translation[expression] = expression
                expr_type = "aggregate" if expression.is_aggregate() else "expression"
                self._graph.add_node(expression, type=expr_type)

                for nested_expression in expression.iterchildren():
                    self._add_expression_task(nested_expression)
                    self._graph.add_edge(self._node_translation[nested_expression], self._node_translation[expression])
                if expression.is_aggregate and self._groupby is not None:
                    self._graph.add_edge(self._groupby, expression)

            case expr.ColumnExpression():
                self._node_translation[expression] = expression
                self._graph.add_node(expression, type="column_select")
                self._graph.add_edge(expression.column.table, expression)

            case _:
                self._node_translation[expression] = expression
                self._graph.add_node(expression, type="expression")
                for nested_expression in expression.iterchildren():
                    self._add_expression_task(nested_expression)
                    self._graph.add_edge(self._node_translation[nested_expression], self._node_translation[expression])

    def _merge_subgraph(self, graph: nx.DiGraph) -> None:
        self._graph = nx.compose(self._graph, graph)

    def _assert_valid_graph(self) -> None:
        if self._skip_validation:
            return
        final_projection = nx_utils.nx_sinks(self._graph)
        if len(final_projection) != 1:
            n_sinks = len(final_projection)
            raise ValueError("Malformed query task graph. Expected exactly one sink node "
                             f"(found {n_sinks}): '{self._query}' ({final_projection})")


class _ImplicitRelalgParser:
    def __init__(self, query: qal.ImplicitSqlQuery) -> None:
        self._query = query
        self._task_graph = QueryTaskGraph(self._query)

    def parse_query(self) -> RelNode:
        pass


def parse_relalg(query: qal.ImplicitSqlQuery) -> RelNode:
    return _ImplicitRelalgParser(query).parse_query()
