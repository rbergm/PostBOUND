
from __future__ import annotations

import abc
from collections.abc import Sequence
from typing import Optional

from postbound.qal import base, expressions, predicates, qal
from postbound.util import collections as collection_utils


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
    def __init__(self, input_node: RelNode, predicate: predicates.AbstractPredicate, *,
                 parent_node: Optional[RelNode] = None) -> None:
        super().__init__(parent_node)
        self._input_node = input_node
        self._predicate = predicate
        self._maintain_child_links()

    @property
    def input_node(self) -> RelNode:
        return self._input_node

    @property
    def predicate(self) -> predicates.AbstractPredicate:
        return self._predicate

    def children(self) -> Sequence[RelNode]:
        return [self._input_node]

    def __hash__(self) -> int:
        return hash((self._input_node, self._predicate))

    def __eq__(self, other: object) -> bool:
        return isinstance(other, type(self)) and self._input_node == other._input_node and self._predicate == other._predicate

    def __str__(self) -> str:
        return f"σ ({self._predicate})"


class CrossProduct(RelNode):
    def __init__(self, left_input: RelNode, right_input: RelNode, *, parent_node: Optional[RelNode] = None) -> None:
        super().__init__(parent_node)
        self._left_input = left_input
        self._right_input = right_input
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
        return hash((self._left_input, self._right_input))

    def __eq__(self, other: object) -> bool:
        return (isinstance(other, type(self))
                and self._left_input == other._left_input
                and self._right_input == other._right_input)

    def __str__(self) -> str:
        return "⨯"


class Table(RelNode):
    def __init__(self, table: base.TableReference, *, parent_node: Optional[RelNode] = None) -> None:
        super().__init__(parent_node)
        self._table = table
        self._maintain_child_links()

    @property
    def table(self) -> base.TableReference:
        return self._table

    def children(self) -> Sequence[RelNode]:
        return []

    def tables(self) -> frozenset[base.TableReference]:
        return frozenset((self._table,))

    def __hash__(self) -> int:
        return hash(self._table)

    def __eq__(self, other: object) -> bool:
        return isinstance(other, type(self)) and self._table == other._table

    def __str__(self) -> str:
        return self._table.identifier()


class Join(RelNode):
    def __init__(self, left_input: RelNode, right_input: RelNode, predicate: predicates.AbstractPredicate, *,
                 parent_node: Optional[RelNode] = None) -> None:
        super().__init__(parent_node)
        self._left_input = left_input
        self._right_input = right_input
        self._predicate = predicate
        self._maintain_child_links()

    @property
    def left_input(self) -> RelNode:
        return self._left_input

    @property
    def right_input(self) -> RelNode:
        return self._right_input

    @property
    def predicate(self) -> predicates.AbstractPredicate:
        return self._predicate

    def children(self) -> Sequence[RelNode]:
        return [self._left_input, self._right_input]

    def __hash__(self) -> int:
        return hash((self._left_input, self._right_input, self._predicate))

    def __eq__(self, other: object) -> bool:
        return (isinstance(other, type(self))
                and self._left_input == other._left_input and self._right_input == other._right_input
                and self._predicate == other._predicate)

    def __str__(self) -> str:
        return f"⋈ ({self._predicate})"


class Projection(RelNode):
    def __init__(self, input_node: RelNode, columns: Sequence[base.ColumnReference], *,
                 parent_node: Optional[RelNode] = None) -> None:
        super().__init__(parent_node)
        self._input_node = input_node
        self._columns = tuple(columns)
        self._maintain_child_links()

    @property
    def input_node(self) -> RelNode:
        return self._input_node

    @property
    def columns(self) -> Sequence[base.ColumnReference]:
        return self._columns

    def children(self) -> Sequence[RelNode]:
        return [self._input_node]

    def __hash__(self) -> int:
        return hash((self._input_node, self._columns))

    def __eq__(self, other: object) -> bool:
        return isinstance(other, type(self)) and self._input_node == other._input_node and self._columns == other._columns

    def __str__(self) -> str:
        col_str = ", ".join(str(col) for col in self._columns)
        return f"π ({col_str})"


class GroupBy(RelNode):
    def __init__(self, input_node: RelNode, group_columns: Sequence[base.ColumnReference], *,
                 parent_node: Optional[RelNode] = None) -> None:
        super().__init__(parent_node)
        self._input_node = input_node
        self._group_columns = tuple(group_columns)
        self._maintain_child_links()

    @property
    def input_node(self) -> RelNode:
        return self._input_node

    @property
    def group_columns(self) -> Sequence[base.ColumnReference]:
        return self._group_columns

    def children(self) -> Sequence[RelNode]:
        return [self._input_node]

    def __hash__(self) -> int:
        return hash((self._input_node, self._group_columns))

    def __eq__(self, other: object) -> bool:
        return (isinstance(other, type(self))
                and self._input_node == other._input_node and self._group_columns == other._group_columns)

    def __str__(self) -> str:
        group_str = ", ".join(str(col) for col in self._group_columns)
        return f"Γ ({group_str})"


def parse_relalg(query: qal.ImplicitSqlQuery) -> RelNode:
    base_tables = query.from_clause.tables()
    if not base_tables:
        raise ValueError("Relational algebra for queries without tables is currently unsupported!")
    predicates = query.predicates()

    base_table_fragments: dict[base.TableReference, RelNode] = {}
    for base_table in base_tables:
        table_fragment = Table(base_table)
        filter_predicate = predicates.filters_for(base_table)
        if filter_predicate:
            table_fragment = Selection(table_fragment, filter_predicate)
        base_table_fragments[base_table] = table_fragment

    for join in predicates.joins():
        if len(join.tables()) != 2:
            # TODO
            continue
        first_tab, second_tab = join.tables()
        join_node = Join(base_table_fragments[first_tab], base_table_fragments[second_tab], join)
        for joined_table in join_node.tables():
            base_table_fragments[joined_table] = join_node

    join_fragments = set(base_table_fragments.values())
    if len(join_fragments) == 1:
        final_fragment: RelNode = collection_utils.simplify(join_fragments)
    else:
        first_fragment, *remaining_fragments = join_fragments
        final_fragment = first_fragment
        for remaining_fragment in remaining_fragments:
            final_fragment = CrossProduct(final_fragment, remaining_fragment)

    if query.groupby_clause is not None:
        group_columns: list[base.ColumnReference] = []
        for column in query.groupby_clause.columns:
            if not isinstance(column, expressions.ColumnExpression):
                # TODO
                continue
            group_columns.append(column.column)
        final_fragment = GroupBy(final_fragment, group_columns)

    if query.having_clause is not None:
        final_fragment = Selection(final_fragment, query.having_clause.condition)

    if query.orderby_clause is not None:
        # TODO
        pass

    if query.limit_clause is not None:
        # TODO
        pass

    if not query.select_clause.is_star():
        projection_targets: list[base.ColumnReference] = []
        for projection in query.select_clause.targets:
            if not isinstance(projection.expression, expressions.ColumnExpression):
                # TODO
                continue
            projection_targets.append(projection.expression.column)
        final_fragment = Projection(final_fragment, projection_targets)

    return final_fragment
