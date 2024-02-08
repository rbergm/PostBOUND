
from __future__ import annotations

import abc
import collections
import dataclasses
import enum
import functools
import operator
import typing
from collections.abc import Callable, Iterable, Sequence
from typing import Optional

from postbound.qal import base, clauses, expressions as expr, predicates as preds, qal
from postbound.qal.base import TableReference
from postbound.qal.expressions import SqlExpression
from postbound.util import collections as collection_utils, dicts as dict_utils


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

    def tables(self, *, ignore_subqueries: bool = False) -> frozenset[base.TableReference]:
        return frozenset(collection_utils.set_union(child.tables(ignore_subqueries=ignore_subqueries)
                                                    for child in self.children()))

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

    @abc.abstractmethod
    def accept_visitor(self, visitor: RelNodeVisitor[VisitorResult]) -> VisitorResult:
        raise NotImplementedError

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

    def accept_visitor(self, visitor: RelNodeVisitor[VisitorResult]) -> VisitorResult:
        return visitor.visit_selection(self)

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

    def accept_visitor(self, visitor: RelNodeVisitor[VisitorResult]) -> VisitorResult:
        return visitor.visit_cross_product(self)

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

    def accept_visitor(self, visitor: RelNodeVisitor[VisitorResult]) -> VisitorResult:
        return visitor.visit_union(visitor)

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

    def accept_visitor(self, visitor: RelNodeVisitor[VisitorResult]) -> VisitorResult:
        return visitor.visit_intersection(visitor)

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

    def accept_visitor(self, visitor: RelNodeVisitor[VisitorResult]) -> VisitorResult:
        return visitor.visit_difference(visitor)

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
                 subquery_input: Optional[RelNode] = None, parent_node: Optional[RelNode] = None) -> None:
        super().__init__(parent_node)
        self._table = table
        self._provided_cols = frozenset(col if isinstance(col, expr.ColumnExpression) else expr.ColumnExpression(col)
                                        for col in provided_columns)
        self._subquery_input = subquery_input
        self._hash_val = hash((self._table, self._subquery_input))
        self._maintain_child_links()

    @property
    def table(self) -> base.TableReference:
        return self._table

    @property
    def subquery_input(self) -> Optional[RelNode]:
        return self._subquery_input

    def children(self) -> Sequence[RelNode]:
        return [self._subquery_input] if self._subquery_input else []

    def tables(self, *, ignore_subqueries: bool = False) -> frozenset[base.TableReference]:
        if ignore_subqueries:
            return frozenset((self._table,))
        return super().tables() | {self._table}

    def provided_expressions(self) -> frozenset[expr.SqlExpression]:
        return super().provided_expressions() | self._provided_cols

    def accept_visitor(self, visitor: RelNodeVisitor[VisitorResult]) -> VisitorResult:
        return visitor.visit_base_table(self)

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

    def accept_visitor(self, visitor: RelNodeVisitor[VisitorResult]) -> VisitorResult:
        return visitor.visit_theta_join(self)

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

    def accept_visitor(self, visitor: RelNodeVisitor[VisitorResult]) -> VisitorResult:
        return visitor.visit_projection(self)

    def __hash__(self) -> int:
        return self._hash_val

    def __eq__(self, other: object) -> bool:
        return isinstance(other, type(self)) and self._input_node == other._input_node and self._targets == other._targets

    def __str__(self) -> str:
        col_str = ", ".join(str(col) for col in self._targets)
        return f"π ({col_str})"


class GroupBy(RelNode):
    def __init__(self, input_node: RelNode, group_columns: Sequence[expr.SqlExpression], *,
                 aggregates: Optional[dict[frozenset[expr.SqlExpression], frozenset[expr.FunctionExpression]]] = None,
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
        return super().provided_expressions() | collection_utils.set_union(self._aggregates.values())

    def accept_visitor(self, visitor: RelNodeVisitor[VisitorResult]) -> VisitorResult:
        return visitor.visit_groupby(self)

    def __hash__(self) -> int:
        return self._hash_val

    def __eq__(self, other: object) -> bool:
        return (isinstance(other, type(self))
                and self._input_node == other._input_node
                and self._group_columns == other._group_columns
                and self._aggregates == other._aggregates)

    def __str__(self) -> str:
        pretty_aggregations: dict[str, str] = {}
        for cols, agg_funcs in self._aggregates.items():
            if len(cols) == 1:
                col_str = str(collection_utils.simplify(cols))
            else:
                col_str = "(" + ", ".join(str(c) for c in cols) + ")"
            if len(agg_funcs) == 1:
                agg_str = str(collection_utils.simplify(agg_funcs))
            else:
                agg_str = "(" + ", ".join(str(agg) for agg in agg_funcs) + ")"
            pretty_aggregations[col_str] = agg_str

        agg_str = ", ".join(f"{col}: {agg_func}" for col, agg_func in pretty_aggregations.items())
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

    def accept_visitor(self, visitor: RelNodeVisitor[VisitorResult]) -> VisitorResult:
        return visitor.visit_rename(self)

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

    def accept_visitor(self, visitor: RelNodeVisitor[VisitorResult]) -> VisitorResult:
        return visitor.visit_sort(self)

    def __hash__(self) -> int:
        return self._hash_val

    def __eq__(self, other: object) -> bool:
        return isinstance(other, type(self)) and self._input_node == other._input_node and self._sorting == other._sorting

    def __str__(self) -> str:
        sorting_str = ", ".join(f"{sort_col}{'↓' if sort_dir == 'asc' else '↑'}" for sort_col, sort_dir in self._sorting)
        return f"τ ({sorting_str})"


class Map(RelNode):
    def __init__(self, input_node: RelNode,
                 mapping: dict[frozenset[expr.SqlExpression | base.ColumnReference], frozenset[expr.SqlExpression]], *,
                 parent_node: Optional[RelNode] = None) -> None:
        super().__init__(parent_node)
        self._input_node = input_node
        self._mapping = dict_utils.frozendict(
            {expr.ColumnExpression(expression) if isinstance(expression, base.ColumnReference) else expression: target
             for expression, target in mapping.items()})
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

    def accept_visitor(self, visitor: RelNodeVisitor[VisitorResult]) -> VisitorResult:
        return visitor.visit_map(self)

    def __hash__(self) -> int:
        return self._hash_val

    def __eq__(self, other: object) -> bool:
        return isinstance(other, type(self)) and self._input_node == other._input_node and self._mapping == other._mapping

    def __str__(self) -> str:
        pretty_mapping: dict[str, str] = {}
        for target_col, expression in self._mapping.items():
            if len(target_col) == 1:
                target_col = collection_utils.simplify(target_col)
                target_str = str(target_col)
            else:
                target_str = "(" + ", ".join(str(t) for t in target_col) + ")"
            if len(expression) == 1:
                expression = collection_utils.simplify(expression)
                expr_str = str(expression)
            else:
                expr_str = "(" + ", ".join(str(e) for e in expression) + ")"
            pretty_mapping[target_str] = expr_str

        mapping_str = ", ".join(f"{target_col}: {expr}" for target_col, expr in pretty_mapping.items())
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

    def accept_visitor(self, visitor: RelNodeVisitor[VisitorResult]) -> VisitorResult:
        return visitor.visit_duplicate_elim(self)

    def __hash__(self) -> int:
        return self._hash_val

    def __eq__(self, other: object) -> bool:
        return isinstance(other, type(self)) and self._input_node == other._input_node

    def __str__(self) -> str:
        return "δ"


class SemiJoin(RelNode):
    def __init__(self, input_node: RelNode, subquery_node: RelNode, predicate: Optional[preds.AbstractPredicate] = None, *,
                 parent_node: Optional[RelNode] = None) -> None:
        # TODO: dependent iff predicate is None
        super().__init__(parent_node)
        self._input_node = input_node
        self._subquery_node = subquery_node
        self._predicate = predicate
        self._hash_val = hash((self._input_node, self._subquery_node, self._predicate))
        self._maintain_child_links()

    @property
    def input_node(self) -> RelNode:
        return self._input_node

    @property
    def subquery_node(self) -> RelNode:
        return self._subquery_node

    @property
    def predicate(self) -> Optional[preds.AbstractPredicate]:
        return self._predicate

    def children(self) -> Sequence[RelNode]:
        return [self._input_node, self._subquery_node]

    def required_expressions(self) -> frozenset[expr.SqlExpression]:
        return _collect_all_expressions(self._predicate.iterexpressions())

    def accept_visitor(self, visitor: RelNodeVisitor[VisitorResult]) -> VisitorResult:
        return visitor.visit_semijoin(self)

    def __hash__(self) -> int:
        return self._hash_val

    def __eq__(self, other: object) -> bool:
        return (isinstance(other, type(self))
                and self._input_node == other._input_node and self._subquery_node == other._subquery_node
                and self._predicate == other._predicate)

    def __str__(self) -> str:
        return "⋉" if self._predicate is None else f"⋉ ({self._predicate})"


class AntiJoin(RelNode):
    def __init__(self, input_node: RelNode, subquery_node: RelNode, predicate: Optional[preds.AbstractPredicate] = None, *,
                 parent_node: Optional[RelNode] = None) -> None:
        # TODO: dependent iff predicate is None
        super().__init__(parent_node)
        self._input_node = input_node
        self._subquery_node = subquery_node
        self._predicate = predicate
        self._hash_val = hash((self._input_node, self._subquery_node, self._predicate))
        self._maintain_child_links()

    @property
    def input_node(self) -> RelNode:
        return self._input_node

    @property
    def subquery_node(self) -> RelNode:
        return self._subquery_node

    @property
    def predicate(self) -> Optional[preds.AbstractPredicate]:
        return self._predicate

    def children(self) -> Sequence[RelNode]:
        return [self._input_node, self._subquery_node]

    def required_expressions(self) -> frozenset[expr.SqlExpression]:
        return _collect_all_expressions(self._predicate.iterexpressions())

    def accept_visitor(self, visitor: RelNodeVisitor[VisitorResult]) -> VisitorResult:
        return visitor.visit_antijoin(self)

    def __hash__(self) -> int:
        return self._hash_val

    def __eq__(self, other: object) -> bool:
        return (isinstance(other, type(self))
                and self._input_node == other._input_node and self._subquery_node == other._subquery_node
                and self._predicate == other._predicate)

    def __str__(self) -> str:
        return "▷" if self._predicate is None else f"▷ ({self._predicate})"


class SubqueryScan(RelNode):
    def __init__(self, input_node: RelNode, subquery: qal.SqlQuery, *, parent_node: Optional[RelNode] = None) -> None:
        super().__init__(parent_node)
        self._input_node = input_node
        self._subquery = subquery
        self._hash_val = hash((self._input_node, self._subquery))
        self._maintain_child_links()

    @property
    def input_node(self) -> RelNode:
        return self._input_node

    @property
    def subquery(self) -> qal.SqlQuery:
        return self._subquery

    def tables(self, *, ignore_subqueries: bool = False) -> frozenset[TableReference]:
        return frozenset() if ignore_subqueries else super().tables(ignore_subqueries=ignore_subqueries)

    def children(self) -> Sequence[RelNode]:
        return [self._input_node]

    def provided_expressions(self) -> frozenset[SqlExpression]:
        return {expr.SubqueryExpression(self._subquery)} | super().provided_expressions()

    def accept_visitor(self, visitor: RelNodeVisitor[VisitorResult]) -> VisitorResult:
        return visitor.visit_subquery(self)

    def __hash__(self) -> int:
        return self._hash_val

    def __eq__(self, other: object) -> bool:
        return isinstance(other, type(self)) and self._input_node == other._input_node and self._subquery == other._subquery

    def __str__(self) -> str:
        return "<<Scalar Subquery Scan>>" if self._subquery.is_scalar() else "<<Subquery Scan>>"


VisitorResult = typing.TypeVar("VisitorResult")


class RelNodeVisitor(abc.ABC, typing.Generic[VisitorResult]):

    @abc.abstractmethod
    def visit_selection(self, selection: Selection) -> VisitorResult:
        raise NotImplementedError

    @abc.abstractmethod
    def visit_cross_product(self, cross_product: CrossProduct) -> VisitorResult:
        raise NotImplementedError

    @abc.abstractmethod
    def visit_union(self, union: Union) -> VisitorResult:
        raise NotImplementedError

    @abc.abstractmethod
    def visit_intersection(self, intersection: Intersection) -> VisitorResult:
        raise NotImplementedError

    @abc.abstractmethod
    def visit_difference(self, difference: Difference) -> VisitorResult:
        raise NotImplementedError

    @abc.abstractmethod
    def visit_base_table(self, base_table: Table) -> VisitorResult:
        raise NotImplementedError

    @abc.abstractmethod
    def visit_theta_join(self, join: ThetaJoin) -> VisitorResult:
        raise NotImplementedError

    @abc.abstractmethod
    def visit_projection(self, projection: Projection) -> VisitorResult:
        raise NotImplementedError

    @abc.abstractmethod
    def visit_groupby(self, grouping: GroupBy) -> VisitorResult:
        raise NotImplementedError

    @abc.abstractmethod
    def visit_rename(self, rename: Rename) -> VisitorResult:
        raise NotImplementedError

    @abc.abstractmethod
    def visit_sort(self, sorting: Sort) -> VisitorResult:
        raise NotImplementedError

    @abc.abstractmethod
    def visit_map(self, mapping: Map) -> VisitorResult:
        raise NotImplementedError

    @abc.abstractmethod
    def visit_duplicate_elim(self, duplicate_elim: DuplicateElimination) -> VisitorResult:
        raise NotImplementedError

    @abc.abstractmethod
    def visit_semijoin(self, join: SemiJoin) -> VisitorResult:
        raise NotImplementedError

    @abc.abstractmethod
    def visit_antijoin(self, join: AntiJoin) -> VisitorResult:
        raise NotImplementedError

    @abc.abstractmethod
    def visit_subquery(self, subquery: SubqueryScan) -> VisitorResult:
        raise NotImplementedError


def _is_aggregation(expression: expr.SqlExpression) -> bool:
    return isinstance(expression, expr.FunctionExpression) and expression.is_aggregate()


def _requires_aggregation(expression: expr.SqlExpression) -> bool:
    return any(_is_aggregation(child_expr) or _requires_aggregation(child_expr) for child_expr in expression.iterchildren())


def _generate_expression_mapping_dict(expressions: list[expr.SqlExpression]
                                      ) -> dict[frozenset[expr.SqlExpression], frozenset[expr.SqlExpression]]:
    mapping: dict[frozenset[expr.SqlExpression], set[expr.SqlExpression]] = collections.defaultdict(set)
    for expression in expressions:
        child_expressions = frozenset(expression.iterchildren())
        mapping[child_expressions].add(expression)
    return {child_expressions: frozenset(derived_expressions) for child_expressions, derived_expressions in mapping.items()}


class EvaluationPhase(enum.IntEnum):
    BaseTable = enum.auto()
    Join = enum.auto()
    PostJoin = enum.auto()
    PostAggregation = enum.auto()


@dataclasses.dataclass(frozen=True)
class _SubquerySet:
    subqueries: frozenset[qal.SqlQuery]

    @staticmethod
    def empty() -> _SubquerySet:
        return _SubquerySet(frozenset())

    @staticmethod
    def of(subqueries: Iterable[qal.SqlQuery]) -> _SubquerySet:
        return _SubquerySet(frozenset([subqueries]))

    def __add__(self, other: _SubquerySet) -> _SubquerySet:
        if not isinstance(other, type(self)):
            return NotImplemented
        return _SubquerySet(self.subqueries | other.subqueries)

    def __bool__(self) -> bool:
        return bool(self.subqueries)


class _SubqueryDetector(expr.SqlExpressionVisitor[_SubquerySet], preds.PredicateVisitor[_SubquerySet]):
    def visit_and_predicate(self, predicate: preds.CompoundPredicate,
                            components: Sequence[preds.AbstractPredicate]) -> _SubquerySet:
        return self._traverse_predicate_expressions(predicate)

    def visit_or_predicate(self, predicate: preds.CompoundPredicate,
                           components: Sequence[preds.AbstractPredicate]) -> _SubquerySet:
        return self._traverse_predicate_expressions(predicate)

    def visit_not_predicate(self, predicate: preds.CompoundPredicate,
                            child_predicate: preds.AbstractPredicate) -> _SubquerySet:
        return self._traverse_predicate_expressions(predicate)

    def visit_binary_predicate(self, predicate: preds.BinaryPredicate) -> _SubquerySet:
        return self._traverse_predicate_expressions(predicate)

    def visit_between_predicate(self, predicate: preds.BetweenPredicate) -> _SubquerySet:
        return self._traverse_predicate_expressions(predicate)

    def visit_in_predicate(self, predicate: preds.InPredicate) -> _SubquerySet:
        return self._traverse_predicate_expressions(predicate)

    def visit_unary_predicate(self, predicate: preds.UnaryPredicate) -> _SubquerySet:
        return self._traverse_predicate_expressions(predicate)

    def visit_static_value_expr(self, expression: expr.StaticValueExpression) -> _SubquerySet:
        return _SubquerySet.empty()

    def visit_column_expr(self, expression: expr.ColumnExpression) -> _SubquerySet:
        return _SubquerySet.empty()

    def visit_cast_expr(self, expression: expr.CastExpression) -> _SubquerySet:
        return self._traverse_nested_expressions(expression)

    def visit_function_expr(self, expression: expr.FunctionExpression) -> _SubquerySet:
        return self._traverse_nested_expressions(expression)

    def visit_mathematical_expr(self, expression: expr.MathematicalExpression) -> _SubquerySet:
        return self._traverse_nested_expressions(expression)

    def visit_star_expr(self, expression: expr.StarExpression) -> _SubquerySet:
        return _SubquerySet.empty()

    def visit_subquery_expr(self, expression: expr.SubqueryExpression) -> _SubquerySet:
        return _SubquerySet.of(expression.query)

    def _traverse_predicate_expressions(self, predicate: preds.AbstractPredicate) -> _SubquerySet:
        return functools.reduce(operator.add, [expression.accept_visitor(self) for expression in predicate.iterexpressions()])

    def _traverse_nested_expressions(self, expression: expr.SqlExpression) -> _SubquerySet:
        return functools.reduce(operator.add,
                                [nested_expression.accept_visitor(self) for nested_expression in expression.iterchildren()])


class _BaseTableLookup(expr.SqlExpressionVisitor[Optional[base.TableReference]], preds.PredicateVisitor[base.TableReference]):
    def visit_and_predicate(self, predicate: preds.CompoundPredicate,
                            components: Sequence[preds.AbstractPredicate]) -> base.TableReference:
        base_tables = {child_pred.accept_visitor(self) for child_pred in components}
        return self._fetch_valid_base_tables(base_tables)

    def visit_or_predicate(self, predicate: preds.CompoundPredicate,
                           components: Sequence[preds.AbstractPredicate]) -> base.TableReference:
        base_tables = {child_pred.accept_visitor(self) for child_pred in components}
        return self._fetch_valid_base_tables(base_tables)

    def visit_not_predicate(self, predicate: preds.CompoundPredicate,
                            child_predicate: preds.AbstractPredicate) -> base.TableReference:
        return child_predicate.accept_visitor(self)

    def visit_binary_predicate(self, predicate: preds.BinaryPredicate) -> bool:
        base_tables = (predicate.first_argument.accept_visitor(self), predicate.second_argument.accept_visitor(self))
        return self._fetch_valid_base_tables(set(base_tables))

    def visit_between_predicate(self, predicate: preds.BetweenPredicate) -> bool:
        base_tables = (predicate.column.accept_visitor(self),
                       predicate.interval_start.accept_visitor(self), predicate.interval_end.accept_visitor(self))
        return self._fetch_valid_base_tables(set(base_tables))

    def visit_in_predicate(self, predicate: preds.InPredicate) -> bool:
        base_tables = {predicate.column.accept_visitor(self)}
        base_tables |= collection_utils.set_union(val.accept_visitor(self) for val in predicate.values)
        return self._fetch_valid_base_tables(base_tables)

    def visit_unary_predicate(self, predicate: preds.UnaryPredicate) -> bool:
        return predicate.column.accept_visitor(self)

    def visit_static_value_expr(self, expression: expr.StaticValueExpression) -> Optional[base.TableReference]:
        return None

    def visit_column_expr(self, expression: expr.ColumnExpression) -> Optional[base.TableReference]:
        return expression.column.table

    def visit_cast_expr(self, expression: expr.CastExpression) -> Optional[base.TableReference]:
        return expression.casted_expression.accept_visitor(self)

    def visit_function_expr(self, expression: expr.FunctionExpression) -> Optional[base.TableReference]:
        referenced_tables = {argument.accept_visitor(self) for argument in expression.arguments}
        return self._fetch_valid_base_tables(referenced_tables, accept_empty=True)

    def visit_mathematical_expr(self, expression: expr.MathematicalExpression) -> bool:
        base_tables = {child.accept_visitor(self) for child in expression.iterchildren()}
        return self._fetch_valid_base_tables(base_tables)

    def visit_star_expr(self, expression: expr.StarExpression) -> Optional[base.TableReference]:
        return None

    def visit_subquery_expr(self, expression: expr.SubqueryExpression) -> Optional[base.TableReference]:
        subquery = expression.query
        if not subquery.is_dependent():
            return None
        dependent_tables = subquery.unbound_tables()
        return self._fetch_valid_base_tables(dependent_tables, accept_empty=True)

    def _fetch_valid_base_tables(self, base_tables: set[base.TableReference | None], *,
                                 accept_empty: bool = False) -> Optional[base.TableReference]:
        if None in base_tables:
            base_tables.remove(None)
        if len(base_tables) != 1 or (accept_empty and not base_tables):
            raise ValueError(f"Expected exactly one base predicate but found {base_tables}")
        return collection_utils.simplify(base_tables) if base_tables else None

    def __call__(self, elem: preds.AbstractPredicate | expr.SqlExpression) -> base.TableReference:
        if isinstance(elem, preds.AbstractPredicate) and elem.is_join():
            raise ValueError(f"Cannot determine base table for join predicate '{elem}'")
        tables = elem.tables()
        if len(tables) == 1:
            return collection_utils.simplify(tables)
        base_table = elem.accept_visitor(self)
        if base_table is None:
            raise ValueError(f"No base table found in '{elem}'")
        return base_table


class ExpressionCollector(expr.SqlExpressionVisitor[set[expr.SqlExpression]]):
    def __init__(self, matcher: Callable[[expr.SqlExpression], bool], *, continue_after_match: bool = False) -> None:
        self.matcher = matcher
        self.continue_after_match = continue_after_match

    def visit_column_expr(self, expression: expr.ColumnExpression) -> set[expr.SqlExpression]:
        return self._check_match(expression)

    def visit_cast_expr(self, expression: expr.CastExpression) -> set[expr.SqlExpression]:
        return self._check_match(expression)

    def visit_function_expr(self, expression: expr.FunctionExpression) -> set[expr.SqlExpression]:
        return self._check_match(expression)

    def visit_mathematical_expr(self, expression: expr.MathematicalExpression) -> set[expr.SqlExpression]:
        return self._check_match(expression)

    def visit_star_expr(self, expression: expr.StarExpression) -> set[expr.SqlExpression]:
        return self._check_match(expression)

    def visit_static_value_expr(self, expression: expr.StaticValueExpression) -> set[expr.SqlExpression]:
        return self._check_match(expression)

    def visit_subquery_expr(self, expression: expr.SubqueryExpression) -> set[expr.SqlExpression]:
        return self._check_match(expression)

    def _check_match(self, expression: expr.SqlExpression) -> set[expr.SqlExpression]:
        own_match = {expression} if self.matcher(expression) else set()
        if own_match and not self.continue_after_match:
            return own_match
        return own_match | collection_utils.set_union(child.accept_visitor(self) for child in expression.iterchildren())


def _collect_all_expressions(expression: expr.SqlExpression) -> frozenset[expr.SqlExpression]:
    child_expressions = collection_utils.set_union(_collect_all_expressions(child_expr)
                                                   for child_expr in expression.iterchildren())
    all_expressions = frozenset({expression} | child_expressions)
    return frozenset({e for e in all_expressions
                      if not isinstance(e, expr.StaticValueExpression) and not isinstance(e, expr.StarExpression)})


def _determine_expression_phase(expression: expr.SqlExpression) -> EvaluationPhase:
    match expression:
        case expr.ColumnExpression():
            return EvaluationPhase.BaseTable
        case expr.FunctionExpression() if expression.is_aggregate():
            return EvaluationPhase.PostAggregation
        case expr.FunctionExpression() | expr.MathematicalExpression() | expr.CastExpression():
            own_phase = EvaluationPhase.Join if len(expression.tables()) > 1 else EvaluationPhase.BaseTable
            child_phase = max(_determine_expression_phase(child_expr) for child_expr in expression.iterchildren())
            return max(own_phase, child_phase)
        case expr.SubqueryExpression():
            return EvaluationPhase.BaseTable if len(expression.query.unbound_tables()) < 2 else EvaluationPhase.PostJoin
        case expr.StarExpression() | expr.StaticValueExpression():
            raise ValueError(f"No evaluation phase for static expression '{expression}'")
        case _:
            raise ValueError(f"Unknown expression type: '{expression}'")


def _determine_predicate_phase(predicate: preds.AbstractPredicate) -> EvaluationPhase:
    nested_subqueries = predicate.accept_visitor(_SubqueryDetector())
    subquery_tables = len(collection_utils.set_union(subquery.bound_tables() for subquery in nested_subqueries.subqueries))
    n_tables = len(predicate.tables()) - subquery_tables
    if n_tables == 1:
        # It could actually be that the number of tables is negative. E.g. HAVING count(*) < (SELECT min(r_a) FROM R)
        # Therefore, we only check for exactly 1 table
        return EvaluationPhase.BaseTable

    # FIXME

    expression_phase = max(_determine_expression_phase(expression) for expression in predicate.iterexpressions()
                           if type(expression) not in {expr.StarExpression, expr.StaticValueExpression})
    if expression_phase > EvaluationPhase.Join:
        return expression_phase

    return EvaluationPhase.Join if isinstance(predicate, preds.BinaryPredicate) else EvaluationPhase.PostJoin


def _filter_eval_phase(predicate: preds.AbstractPredicate,
                       expected_eval_phase: EvaluationPhase) -> Optional[preds.AbstractPredicate]:
    eval_phase = _determine_predicate_phase(predicate)
    print(".. Determined predicate", predicate, "to be of eval phase", eval_phase)
    if eval_phase < expected_eval_phase:
        return None

    if isinstance(predicate, preds.CompoundPredicate) and predicate.operation == expr.LogicalSqlCompoundOperators.And:
        child_predicates = [child for child in predicate.children
                            if _determine_predicate_phase(child) == expected_eval_phase]
        return preds.CompoundPredicate.create_and(child_predicates) if child_predicates else None

    return predicate if eval_phase == expected_eval_phase else None


class _ImplicitRelalgParser:
    def __init__(self, query: qal.ImplicitSqlQuery, *,
                 provided_base_tables: Optional[dict[base.TableReference, RelNode]] = None) -> None:
        self._query = query
        self._base_table_fragments: dict[base.TableReference, RelNode] = {}
        self._required_columns: dict[base.TableReference, set[base.ColumnReference]] = collections.defaultdict(set)
        self._provided_base_tables: dict[base.TableReference, RelNode] = provided_base_tables if provided_base_tables else {}

        collection_utils.foreach(self._query.columns(), lambda col: self._required_columns[col.table].add(col))

    def generate_relnode(self) -> RelNode:
        # TODO: robustness: query without FROM clause

        if self._query.cte_clause:
            for cte in self._query.cte_clause.queries:
                cte_root = self._add_subquery(cte.query)
                self._add_table(cte.target_table, input_node=cte_root)

        # we add the WHERE clause before all explicit JOIN statements to make sure filters are already present and we can
        # stitch together the correct fragments in OUTER JOINs
        # Once the explicit JOINs have been processed, we continue with all remaining implicit joins
        # TODO: since the implementation of JOIN statements is currently undergoing a major rework, we don't process such
        # statements at all

        collection_utils.foreach(self._query.from_clause.items, self._add_table_source)

        if self._query.where_clause:
            self._add_predicate(self._query.where_clause.predicate, eval_phase=EvaluationPhase.BaseTable)

        final_fragment = self._generate_initial_join_order()

        if self._query.where_clause:
            # add all post-join filters here
            final_fragment = self._add_predicate(self._query.where_clause.predicate, input_node=final_fragment,
                                                 eval_phase=EvaluationPhase.PostJoin)

        final_fragment = self._add_aggregation(final_fragment)
        if self._query.having_clause:
            final_fragment = self._add_predicate(self._query.having_clause.condition, input_node=final_fragment,
                                                 eval_phase=EvaluationPhase.PostAggregation)

        final_fragment = self._add_final_projection(final_fragment)
        return final_fragment

    def _resolve(self, table: base.TableReference) -> RelNode:
        if table in self._base_table_fragments:
            return self._base_table_fragments[table]
        return self._provided_base_tables[table]

    def _add_table(self, table: base.TableReference, *, input_node: Optional[RelNode] = None) -> RelNode:
        required_cols = self._required_columns[table]
        table_node = Table(table, required_cols, subquery_input=input_node)
        self._base_table_fragments[table] = table_node
        return table_node

    def _add_table_source(self, table_source: clauses.TableSource) -> RelNode:
        match table_source:
            case clauses.DirectTableSource():
                if table_source.table.virtual:
                    # Virtual tables in direct table sources are only created through references to CTEs. However, these CTEs
                    # have already been included in the base table fragments.
                    return self._base_table_fragments[table_source.table]
                return self._add_table(table_source.table)
            case clauses.SubqueryTableSource():
                subquery_root = self._add_subquery(table_source.query)
                self._base_table_fragments[table_source.target_table] = subquery_root
                return self._add_table(table_source.target_table, input_node=subquery_root)
            case clauses.JoinTableSource():
                raise ValueError(f"Explicit JOIN syntax is currently not supported: '{table_source}'")
            case _:
                raise ValueError(f"Unknown table source: '{table_source}'")

    def _generate_initial_join_order(self) -> RelNode:
        # TODO: figure out the interaction between implicit and explicit joins, especially regarding their timing

        joined_tables: set[base.TableReference] = set()
        for table_source in self._query.from_clause.items:
            # TODO: determine correct join partners for explicit JOINs
            joined_tables |= table_source.tables()

        if self._query.where_clause:
            self._add_predicate(self._query.where_clause.predicate, eval_phase=EvaluationPhase.Join)

        head_nodes = set(self._base_table_fragments.values())
        if len(head_nodes) == 1:
            return collection_utils.simplify(head_nodes)

        current_head, *remaining_nodes = head_nodes
        for remaining_node in remaining_nodes:
            current_head = CrossProduct(current_head, remaining_node)
        return current_head

    def _add_aggregation(self, input_node: RelNode) -> RelNode:
        aggregation_collector = ExpressionCollector(lambda e: isinstance(e, expr.FunctionExpression) and e.is_aggregate())
        aggregation_functions: set[expr.FunctionExpression] = (
            collection_utils.set_union(select_expr.accept_visitor(aggregation_collector)
                                       for select_expr in self._query.select_clause.iterexpressions()))

        if self._query.having_clause:
            aggregation_functions |= collection_utils.set_union(having_expr.accept_visitor(aggregation_collector)
                                                                for having_expr in self._query.having_clause.iterexpressions())
        if not self._query.groupby_clause and not aggregation_functions:
            return input_node

        aggregation_arguments: set[expr.SqlExpression] = set()
        for agg_func in aggregation_functions:
            aggregation_arguments |= collection_utils.set_union(_collect_all_expressions(arg) for arg in agg_func.arguments)
        missing_expressions = aggregation_arguments - input_node.provided_expressions()
        if missing_expressions:
            input_node = Map(input_node, _generate_expression_mapping_dict(missing_expressions))

        group_cols = self._query.groupby_clause.group_columns if self._query.groupby_clause else []
        aggregates: dict[frozenset[expr.SqlExpression], set[expr.FunctionExpression]] = collections.defaultdict(set)
        for agg_func in aggregation_functions:
            aggregates[agg_func.arguments].add(agg_func)
        groupby_node = GroupBy(input_node, group_columns=group_cols,
                               aggregates={agg_input: frozenset(agg_funcs) for agg_input, agg_funcs in aggregates.items()})
        return groupby_node

    def _add_final_projection(self, input_node: RelNode) -> RelNode:
        # TODO: Sorting, Duplicate elimination, limit
        if self._query.select_clause.is_star():
            return input_node
        required_expressions = collection_utils.set_union(_collect_all_expressions(target.expression)
                                                          for target in self._query.select_clause.targets)
        missing_expressions = required_expressions - input_node.provided_expressions()
        final_node = (Map(input_node, _generate_expression_mapping_dict(missing_expressions)) if missing_expressions
                      else input_node)
        return Projection(final_node, [target.expression for target in self._query.select_clause.targets])

    def _add_predicate(self, predicate: preds.AbstractPredicate, *, input_node: Optional[RelNode] = None,
                       eval_phase: EvaluationPhase = EvaluationPhase.BaseTable) -> RelNode:
        predicate = _filter_eval_phase(predicate, eval_phase)
        if predicate is None:
            return input_node

        print("Now processing predicate", predicate, "at phase", eval_phase)
        match eval_phase:
            case EvaluationPhase.BaseTable:
                for base_table, base_pred in self._split_filter_predicate(predicate).items():
                    base_table_fragment = self._convert_predicate(base_pred,
                                                                  input_node=self._base_table_fragments[base_table])
                    self._base_table_fragments[base_table] = base_table_fragment
                return base_table_fragment
            case EvaluationPhase.Join:
                for join_predicate in self._split_join_predicate(predicate):
                    join_node = self._convert_join_predicate(join_predicate)
                    for table in join_node.tables():
                        self._base_table_fragments[table] = join_node
                return join_node
            case EvaluationPhase.PostJoin | EvaluationPhase.PostAggregation:
                assert input_node is not None
                return self._convert_predicate(predicate, input_node=input_node)
            case _:
                raise ValueError(f"Unknown evaluation phase '{eval_phase}' for predicate '{predicate}'")

    def _convert_predicate(self, predicate: preds.AbstractPredicate, *, input_node: RelNode) -> RelNode:
        contains_subqueries = _SubqueryDetector()
        final_fragment = input_node

        if isinstance(predicate, preds.UnaryPredicate) and not predicate.accept_visitor(contains_subqueries):
            final_fragment = self._ensure_predicate_applicability(predicate, final_fragment)
            final_fragment = Selection(final_fragment, predicate)
            return final_fragment
        elif isinstance(predicate, preds.UnaryPredicate):
            subquery_target = ("semijoin" if predicate.operation == expr.LogicalSqlOperators.Exists
                               else "antijoin")
            return self._add_expression(predicate.column, input_node=final_fragment, subquery_target=subquery_target)

        if isinstance(predicate, preds.BetweenPredicate) and not predicate.accept_visitor(contains_subqueries):
            final_fragment = self._ensure_predicate_applicability(predicate, final_fragment)
            final_fragment = Selection(final_fragment, predicate)
            return final_fragment
        elif isinstance(predicate, preds.BetweenPredicate):
            # BETWEEN predicate with scalar subquery
            final_fragment = self._add_expression(predicate.column)
            final_fragment = self._add_expression(predicate.interval_start)
            final_fragment = self._add_expression(predicate.interval_end)
            final_fragment = Selection(final_fragment, predicate)
            return final_fragment

        if isinstance(predicate, preds.InPredicate) and not predicate.accept_visitor(contains_subqueries):
            # we need to determine the required expressions due to IN predicates like "r_a + 42 IN (1, 2, 3)"
            # or "r_a IN (r_b + 42, 42)"
            final_fragment = self._ensure_predicate_applicability(predicate, final_fragment)
            final_fragment = Selection(final_fragment, predicate)
            return final_fragment
        elif isinstance(predicate, preds.InPredicate):
            # TODO: test weird IN predicates like r_a IN (1, 2, (SELECT min(...)), 4)
            # or even r_a IN ((SELECT r_a FROM ...) + (SELECT min(...)))
            pure_in_values: list[expr.SqlExpression] = []
            subquery_in_values: list[tuple[expr.SqlExpression, _SubquerySet]] = []
            for value in predicate.values:
                detected_subqueries = value.accept_visitor(contains_subqueries)
                if detected_subqueries and not all(subquery.is_scalar() for subquery in detected_subqueries.subqueries):
                    subquery_in_values.append((value, detected_subqueries))
                else:
                    final_fragment = self._add_expression(value)
                    pure_in_values.append(value)
            final_fragment = self._add_expression(predicate.column)
            if pure_in_values:
                reduced_predicate = preds.InPredicate(predicate.column, pure_in_values)
                final_fragment = Selection(final_fragment, reduced_predicate)
            for subquery_value, detected_subqueries in subquery_in_values:
                final_fragment = self._add_expression(subquery_value, input_node=final_fragment, subquery_target="in",
                                                      in_column=predicate.column)
            return final_fragment

        if isinstance(predicate, preds.BinaryPredicate) and not predicate.accept_visitor(contains_subqueries):
            final_fragment = self._ensure_predicate_applicability(predicate, final_fragment)
            final_fragment = Selection(final_fragment, predicate)
            return final_fragment
        elif isinstance(predicate, preds.BinaryPredicate):
            if predicate.first_argument.accept_visitor(contains_subqueries):
                final_fragment = self._add_expression(predicate.first_argument, input_node=final_fragment,
                                                      subquery_target="scalar")
            if predicate.second_argument.accept_visitor(contains_subqueries):
                final_fragment = self._add_expression(predicate.second_argument, input_node=final_fragment,
                                                      subquery_target="scalar")
            final_fragment = self._ensure_predicate_applicability(predicate, final_fragment)
            final_fragment = Selection(final_fragment, predicate)
            return final_fragment

        if not isinstance(predicate, preds.CompoundPredicate):
            raise ValueError(f"Unknown predicate type: '{predicate}'")
        match predicate.operation:
            case expr.LogicalSqlCompoundOperators.And | expr.LogicalSqlCompoundOperators.Or:
                regular_predicates: list[preds.AbstractPredicate] = []
                subquery_predicates: list[preds.AbstractPredicate] = []
                for child_pred in predicate.iterchildren():
                    if child_pred.accept_visitor(contains_subqueries):
                        subquery_predicates.append(child_pred)
                    else:
                        regular_predicates.append(child_pred)
                if regular_predicates:
                    simplified_composite = preds.CompoundPredicate.create(predicate.operation, regular_predicates)
                    final_fragment = self._ensure_predicate_applicability(simplified_composite, final_fragment)
                    final_fragment = Selection(final_fragment, simplified_composite)
                for subquery_pred in subquery_predicates:
                    if predicate.operation == expr.LogicalSqlCompoundOperators.And:
                        final_fragment = self._convert_predicate(subquery_pred, input_node=final_fragment)
                        continue
                    subquery_branch = self._convert_predicate(subquery_pred, input_node=input_node)
                    final_fragment = Union(final_fragment, subquery_branch)
                return final_fragment
            case expr.LogicalSqlCompoundOperators.Not:
                if not predicate.children.accept_visitor(contains_subqueries):
                    final_fragment = self._ensure_predicate_applicability(predicate, final_fragment)
                    final_fragment = Selection(final_fragment, predicate)
                    return final_fragment
                subquery_branch = self._convert_predicate(predicate.children, input_node=input_node)
                final_fragment = Difference(final_fragment, subquery_branch)
                return final_fragment
            case _:
                raise ValueError(f"Unknown operation for composite predicate '{predicate}'")

    def _convert_join_predicate(self, predicate: preds.AbstractPredicate) -> RelNode:
        contains_subqueries = _SubqueryDetector()
        nested_subqueries = predicate.accept_visitor(contains_subqueries)
        subquery_tables = collection_utils.set_union(subquery.bound_tables() for subquery in nested_subqueries.subqueries)
        table_fragments = {self._resolve(join_partner) for join_partner in predicate.tables() - subquery_tables}
        if len(table_fragments) == 1:
            input_node = collection_utils.simplify(table_fragments)
            required_expressions = collection_utils.set_union(_collect_all_expressions(e) for e in predicate.iterexpressions())
            missing_expressions = required_expressions - input_node.provided_expressions()
            if missing_expressions:
                final_fragment = Map(input_node, _generate_expression_mapping_dict(missing_expressions))
            else:
                final_fragment = input_node
            return Selection(final_fragment, predicate)
        if len(table_fragments) != 2:
            raise ValueError("Expected exactly two base table fragments for join predicate "
                             f"'{predicate}', but found {table_fragments}")

        required_expressions = collection_utils.set_union(_collect_all_expressions(e) for e in predicate.iterexpressions())
        if isinstance(predicate, preds.BinaryPredicate):
            first_input, second_input = table_fragments
            first_arg, second_arg = predicate.first_argument, predicate.second_argument
            if (first_arg.tables() <= first_input.tables(ignore_subqueries=True)
                    and second_arg.tables() <= second_input.tables(ignore_subqueries=True)):
                left_input, right_input = first_input, second_input
            elif (first_arg.tables() <= second_input.tables(ignore_subqueries=True)
                    and second_arg.tables() <= first_input.tables(ignore_subqueries=True)):
                left_input, right_input = second_input, first_input
            else:
                raise ValueError(f"Unsupported join predicate '{predicate}'")

            left_input = self._add_expression(first_arg, input_node=left_input)
            right_input = self._add_expression(second_arg, input_node=right_input)

            provided_expressions = left_input.provided_expressions() | right_input.provided_expressions()
            missing_expressions = required_expressions - provided_expressions
            left_mappings: list[expr.SqlExpression] = []
            right_mappings: list[expr.SqlExpression] = []
            for missing_expr in missing_expressions:
                if missing_expr.tables() <= left_input.tables():
                    left_mappings.append(missing_expr)
                elif missing_expr.tables() <= right_input.tables():
                    right_mappings.append(missing_expr)
                else:
                    raise ValueError("Cannot calculate expression on left or right input: "
                                     f"'{missing_expr}' for predicate '{predicate}'")
            if left_mappings:
                left_input = Map(left_input, _generate_expression_mapping_dict(left_mappings))
            if right_mappings:
                right_input = Map(right_input, _generate_expression_mapping_dict(right_mappings))
            return ThetaJoin(left_input, right_input, predicate)

        if not isinstance(predicate, preds.CompoundPredicate):
            raise ValueError(f"Unsupported join predicate '{predicate}'. Perhaps this should be a post-join filter?")

        match predicate.operation:
            case expr.LogicalSqlCompoundOperators.And | expr.LogicalSqlCompoundOperators.Or:
                regular_predicates: list[preds.AbstractPredicate] = []
                subquery_predicates: list[preds.AbstractPredicate] = []
                for child_pred in predicate.children:
                    if predicate.accept_visitor(contains_subqueries):
                        subquery_predicates.append(child_pred)
                    else:
                        regular_predicates.append(child_pred)
                if regular_predicates:
                    simplified_composite = preds.CompoundPredicate(predicate.operation, regular_predicates)
                    final_fragment = self._convert_join_predicate(simplified_composite)
                else:
                    first_input, second_input = table_fragments
                    final_fragment = CrossProduct(first_input, second_input)
                for subquery_pred in subquery_predicates:
                    final_fragment = self._convert_predicate(subquery_pred, input_node=final_fragment)
                return final_fragment
            case expr.LogicalSqlCompoundOperators.Not:
                pass
            case _:
                raise ValueError(f"Unknown operation for composite predicate '{predicate}'")

    def _add_expression(self, expression: expr.SqlExpression, *, input_node: RelNode,
                        subquery_target: typing.Literal["semijoin", "antijoin", "scalar", "in"] = "scalar",
                        in_column: Optional[expr.SqlExpression] = None) -> RelNode:
        match expression:
            case expr.ColumnExpression() | expr.StaticValueExpression():
                return input_node
            case expr.SubqueryExpression():
                subquery_root = self._add_subquery(expression.query)
                match subquery_target:
                    case "semijoin":
                        return SemiJoin(input_node, subquery_root)
                    case "antijoin":
                        return AntiJoin(input_node, subquery_root)
                    case "scalar":
                        return CrossProduct(input_node, subquery_root)
                    case "in" if expression.query.is_scalar():
                        return CrossProduct(input_node, subquery_root)
                    case "in" if not expression.query.is_scalar():
                        assert isinstance(subquery_root, Projection) and len(subquery_root.columns) == 1
                        in_predicate = preds.BinaryPredicate.equal(in_column, subquery_root.columns[0])
                        return SemiJoin(input_node, subquery_root, in_predicate)
            case expr.CastExpression() | expr.FunctionExpression() | expr.MathematicalExpression():
                required_expressions = _collect_all_expressions(expression)
                missing_expressions = required_expressions - input_node.provided_expressions()
                return Map(input_node, _generate_expression_mapping_dict(missing_expressions))
            case _:
                raise ValueError(f"Did not expect expression '{expression}'")

    def _add_subquery(self, subquery: qal.SqlQuery) -> RelNode:
        subquery_parser = _ImplicitRelalgParser(subquery, provided_base_tables=self._base_table_fragments)
        subquery_root = subquery_parser.generate_relnode()
        self._required_columns = dict_utils.merge(subquery_parser._required_columns, self._required_columns)
        return SubqueryScan(subquery_root, subquery)

    def _infer_base_table(self, elem: preds.AbstractPredicate | expr.SqlExpression,
                          relnode: RelNode | None) -> base.TableReference:
        if relnode is not None:
            return relnode
        return _BaseTableLookup()(elem)

    def _split_filter_predicate(self, predicate: preds.AbstractPredicate
                                ) -> dict[base.TableReference, preds.AbstractPredicate]:
        if not isinstance(predicate, preds.CompoundPredicate):
            return {self._infer_base_table(predicate, None): predicate}
        if predicate.operation != expr.LogicalSqlCompoundOperators.And:
            return {self._infer_base_table(predicate, None): predicate}

        raw_predicate_components: dict[base.TableReference, set[preds.AbstractPredicate]] = collections.defaultdict(set)
        for child_pred in predicate.children:
            child_split = self._split_filter_predicate(child_pred)
            for tab, pred in child_split.items():
                raw_predicate_components[tab].add(pred)
        return {base_table: preds.CompoundPredicate.create_and(predicates)
                for base_table, predicates in raw_predicate_components.items()}

    def _split_join_predicate(self, predicate: preds.AbstractPredicate) -> set[preds.AbstractPredicate]:
        if isinstance(predicate, preds.CompoundPredicate) and predicate.operation == expr.LogicalSqlCompoundOperators.And:
            return set(predicate.children)
        return {predicate}

    def _ensure_predicate_applicability(self, predicate: preds.AbstractPredicate, input_node: RelNode) -> RelNode:
        provided_expressions = input_node.provided_expressions()
        required_expressions = collection_utils.set_union(_collect_all_expressions(child)
                                                          for child in predicate.iterexpressions())
        missing_expressions = required_expressions - provided_expressions
        if missing_expressions:
            return Map(input_node, _generate_expression_mapping_dict(missing_expressions))
        return input_node


def parse_relalg(query: qal.ImplicitSqlQuery) -> RelNode:
    return _ImplicitRelalgParser(query).generate_relnode()
