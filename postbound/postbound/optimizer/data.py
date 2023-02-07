from __future__ import annotations

import abc
from dataclasses import dataclass
from typing import Iterable

import networkx as nx

from postbound.qal import base, predicates, qal, transform
from postbound.db import db
from postbound.util import collections as collection_utils, errors, networkx as nx_utils


@dataclass
class JoinPath:
    start_table: base.TableReference
    target_table: base.TableReference
    join_condition: predicates.AbstractPredicate | None = None

    def tables(self) -> Iterable[base.TableReference]:
        return [self.start_table, self.target_table]


class IndexInfo:
    @staticmethod
    def primary_index(column: base.ColumnReference) -> IndexInfo:
        return IndexInfo(column, "primary")

    @staticmethod
    def secondary_index(column: base.ColumnReference) -> IndexInfo:
        return IndexInfo(column, "secondary")

    @staticmethod
    def no_index(column: base.ColumnReference) -> IndexInfo:
        return IndexInfo(column, "none")

    @staticmethod
    def generate_for(column: base.ColumnReference, db_schema: db.DatabaseSchema) -> IndexInfo:
        if db_schema.is_primary_key(column):
            return IndexInfo.primary_index(column)
        elif db_schema.has_secondary_index(column):
            return IndexInfo.secondary_index(column)
        else:
            return IndexInfo.no_index(column)

    def __init__(self, column: base.ColumnReference, index_type: str) -> None:
        self.column = column
        self.index_type = index_type
        self.is_invalid = False

    def is_primary(self) -> bool:
        return self.index_type == "primary"

    def is_secondary(self) -> bool:
        return self.index_type == "secondary"

    def is_indexed(self) -> bool:
        return self.is_primary() or self.is_secondary()

    def can_pk_fk_join(self, other: IndexInfo) -> bool:
        if not self.is_indexed() or not other.is_indexed():
            return False

        if self.is_secondary() and other.is_secondary():
            return False

        # all other cases have at least one primary key index available
        return True

    def invalidate(self) -> None:
        self.is_invalid = True

    def __repr__(self) -> str:
        return str(self)

    def __str__(self) -> str:
        if self.is_invalid:
            return f"INVALID({self.column})"
        elif self.index_type == "none":
            return f"NO INDEX({self.column})"
        elif self.index_type == "primary":
            return f"PRIMARY INDEX({self.column})"
        else:
            return f"SECONDARY INDEX({self.column})"


class JoinGraph:
    """

    The wording of the different join graph methods distinguishes three states of joins (and correspondingly tables):

    - a join might be `free`, if at least one of the corresponding tables have not been marked as joined, yet
    - a join might be `available`, if it is `free` and one of the tables is already included in some intermediate join
    - a join is `consumed`, if its no longer `free`

    By calling the `mark_joined` method,  the state of individual joins and their corresponding tables might change.
    This also means, that former primary key/foreign key joins might become n:m joins (which is the case exactly when
    the primary key table is inserted into an intermediate join result).

    """

    def __init__(self, query: qal.ImplicitSqlQuery, db_schema: db.DatabaseSchema) -> None:
        self.query = query
        self._db_schema = db_schema
        self._index_structures: dict[base.ColumnReference, IndexInfo] = {}

        graph = nx.Graph()
        graph.add_nodes_from(query.tables(), free=True)
        edges = []
        for join_predicate in query.predicates().joins():
            first_col, second_col = join_predicate.columns()
            edges.append((first_col, second_col, {"predicate": join_predicate}))
            self._index_structures[first_col] = IndexInfo.generate_for(first_col, db_schema)
            self._index_structures[second_col] = IndexInfo.generate_for(second_col, db_schema)

        graph.add_edges_from(edges)

        self._graph = graph

    def contains_cross_products(self) -> bool:
        return len(nx.connected_components(self._graph)) > 1

    def contains_free_tables(self) -> bool:
        return any(is_free for __, is_free in self._graph.nodes.data("free"))

    def contains_free_n_m_joins(self) -> bool:
        for first_tab, second_tab, predicate in self._graph.edges.data("predicate"):
            first_col, second_col = predicate.columns()
            if not self._index_structures[first_col].can_pk_fk_join(self._index_structures[second_col]):
                return False
        return True

    def join_components(self) -> Iterable[JoinGraph]:
        components = []
        for component in nx.connected_components(self._graph):
            tables = component.nodes
            component_query = transform.extract_query_fragment(self.query, tables)
            components.append(JoinGraph(component_query, self._db_schema))
        return components

    def available_join_paths(self) -> Iterable[JoinPath]:
        pass

    def nx_graph(self) -> nx.Graph:
        return self._graph

    def is_pk_fk_join(self, fk_table: base.TableReference, pk_table: base.TableReference) -> bool:
        predicate: predicates.AbstractPredicate = self._graph.edges[fk_table, pk_table]["predicate"]
        fk_col = collection_utils.simplify(predicate.columns_of(fk_table))
        pk_col = collection_utils.simplify(predicate.columns_of(pk_table))
        return self._index_structures[fk_col].is_secondary() and self._index_structures[pk_col].is_primary()

    def available_pk_fk_joins_for(self, fk_table: base.TableReference) -> Iterable[JoinPath]:
        pass

    def mark_joined(self, table: base.TableReference) -> None:
        pass


class JoinTreeNode(abc.ABC):
    def __init(self, upper_bound: int) -> None:
        self.upper_bound = upper_bound

    @abc.abstractmethod
    def __hash__(self) -> int:
        raise NotImplementedError


class JoinNode(JoinTreeNode):
    def __init__(self, left_child: JoinTreeNode, right_child: JoinTreeNode, *, join_bound: int,
                 join_condition: predicates.AbstractPredicate | None = None, pk_fk_join: bool = False) -> None:
        self.left_child = left_child
        self.right_child = right_child
        self.join_condition = join_condition
        self.pk_fk_join = pk_fk_join
        self.join_bound = join_bound

    def __hash__(self) -> int:
        return hash(tuple([self.left_child, self.right_child, self.join_condition, self.join_bound]))

    def __eq__(self, other) -> bool:
        return (isinstance(other, type(self))
                and self.left_child == other.left_child
                and self.right_child == other.right_child
                and self.join_condition == other.join_condition
                and self.join_bound == other.join_bound)


class BaseTableNode(JoinTreeNode):
    def __init__(self, table: base.TableReference, cardinality_estimate: int,
                 filter_condition: predicates.AbstractPredicate | None = None) -> None:
        self.table = table
        self.filter = filter_condition
        self.cardinality_estimate = cardinality_estimate

    def __hash__(self) -> int:
        return hash((self.table, self.filter, self.cardinality_estimate))

    def __eq__(self, other) -> bool:
        return (isinstance(other, type(self))
                and self.table == other.table
                and self.filter == other.filter
                and self.cardinality_estimate == other.cardinality_estimate)


class JoinTree:
    @staticmethod
    def cross_product_of(*trees: JoinTree) -> JoinTree:
        if not trees:
            raise ValueError("No trees given")
        elif len(trees) == 1:
            return trees[0]
        first_tree, *additional_trees = trees

        current_root = first_tree.root
        for additional_tree in additional_trees:
            current_root = JoinNode(additional_tree.root, current_root)

        cross_product_tree = JoinTree()
        cross_product_tree.root = current_root
        return cross_product_tree

    @staticmethod
    def for_base_table(table: base.TableReference, base_cardinality: int,
                       filter_predicates: predicates.AbstractPredicate) -> JoinTree:
        root = BaseTableNode(table, base_cardinality, filter_predicates)
        return JoinTree(root)

    def join_with_base_table(self, table: base.TableReference, *, base_cardinality: int,
                             join_predicate: predicates.AbstractPredicate | None = None, join_bound: int | None = None,
                             base_filter_predicate: predicates.AbstractPredicate | None = None) -> JoinTree:
        base_node = BaseTableNode(table, base_cardinality, base_filter_predicate)
        if self.is_empty():
            return JoinTree(base_node)
        else:
            new_root = JoinNode(base_node, self.root, join_bound=join_bound, join_condition=join_predicate)
            return JoinTree(new_root)

    def join_with_subquery(self, subquery: JoinTree, join_predicate: predicates.AbstractPredicate,
                           join_bound: int) -> JoinTree:
        if self.is_empty():
            return JoinTree(subquery.root)
        new_root = JoinNode(subquery.root, self.root, join_bound=join_bound, join_condition=join_predicate)
        return JoinTree(new_root)

    def __init__(self, root: JoinTreeNode | None = None) -> None:
        self.root = root

    def is_empty(self) -> bool:
        return self.root is None

    def _get_upper_bound(self) -> int:
        if self.is_empty():
            raise errors.StateError("Join tree is empty")
        return self.root.upper_bound

    upper_bound = property(_get_upper_bound)

    def __hash__(self) -> int:
        return hash(self.root)

    def __eq__(self, other) -> bool:
        return isinstance(other, type(self)) and self.root == other.root
