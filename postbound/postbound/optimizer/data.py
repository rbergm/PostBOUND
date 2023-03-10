from __future__ import annotations

import abc
import collections
from collections.abc import Container
from dataclasses import dataclass
from typing import Callable, Iterable

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

    def __repr__(self) -> str:
        return str(self)

    def __str__(self) -> str:
        return f"{self.start_table} ⋈ {self.target_table} ({self.join_condition})"


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
        return not self.is_invalid and self.index_type == "primary"

    def is_secondary(self) -> bool:
        return not self.is_invalid and self.index_type == "secondary"

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
        invalid_state = " INVALID" if self.is_invalid else ""
        if self.index_type == "none":
            return f"NO INDEX({self.column})"
        elif self.index_type == "primary":
            return f"PRIMARY INDEX({self.column}){invalid_state}"
        else:
            return f"SECONDARY INDEX({self.column}){invalid_state}"


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

    def __init__(self, query: qal.ImplicitSqlQuery, db_schema: db.DatabaseSchema | None = None) -> None:
        db_schema = db_schema if db_schema else db.DatabasePool.get_instance().current_database().schema()
        self.query = query
        self._db_schema = db_schema
        self._index_structures: dict[base.ColumnReference, IndexInfo] = {}

        graph = nx.Graph()
        graph.add_nodes_from(query.tables(), free=True)
        edges = []
        predicate_map = collections.defaultdict(list)
        for join_predicate in query.predicates().joins():
            first_col, second_col = join_predicate.columns()
            predicate_map[frozenset([first_col.table, second_col.table])].append(join_predicate)

        for tables, joins in predicate_map.items():
            first_tab, second_tab = tables
            join_predicate = predicates.CompoundPredicate.create_and(joins)
            edges.append((first_tab, second_tab, {"predicate": join_predicate}))
            for column in join_predicate.columns():
                self._index_structures[column] = IndexInfo.generate_for(column, db_schema)

        graph.add_edges_from(edges)
        self._graph = graph

    def initial(self) -> bool:
        return all(is_free for __, is_free in self._graph.nodes.data("free"))

    def contains_cross_products(self) -> bool:
        return len(list(nx.connected_components(self._graph))) > 1

    def contains_free_tables(self) -> bool:
        return any(is_free for __, is_free in self._graph.nodes.data("free"))

    def contains_free_n_m_joins(self) -> bool:
        is_first_join = self.initial()
        for first_tab, second_tab, predicate in self._graph.edges.data("predicate"):
            if not self.is_available_join(first_tab, second_tab) and not is_first_join:
                continue
            first_col, second_col = predicate.columns()
            if not self._index_structures[first_col].can_pk_fk_join(self._index_structures[second_col]):
                return True
        return False

    def count_consumed_tables(self) -> int:
        return len([is_free for __, is_free in self._graph.nodes.data("free") if not is_free])

    def join_components(self) -> Iterable[JoinGraph]:
        components = []
        for component in nx.connected_components(self._graph):
            component_query = transform.extract_query_fragment(self.query, component)
            components.append(JoinGraph(component_query, self._db_schema))
        return components

    def available_join_paths(self) -> Iterable[JoinPath]:
        join_paths = []
        if self.initial():
            for join_edge in self._graph.edges.data("predicate"):
                source_table, target_table, join_condition = join_edge
                join_paths.append(JoinPath(source_table, target_table, join_condition))
            return join_paths

        for join_edge in self._graph.edges.data("predicate"):
            source_table, target_table, join_condition = join_edge
            if self.is_free_table(source_table) and self.is_free_table(target_table):
                # both tables are still free -> no path
                continue
            elif not self.is_free_table(source_table) and not self.is_free_table(target_table):
                # both tables are already joined -> no path
                continue

            if self.is_free_table(source_table):
                # fix directionality
                source_table, target_table = target_table, source_table
            join_paths.append(JoinPath(source_table, target_table, join_condition))

        return join_paths

    def available_n_m_join_paths(self) -> Iterable[JoinPath]:
        n_m_paths = []
        for join_path in self.available_join_paths():
            start_table, target_table = join_path.start_table, join_path.target_table
            if not self.is_pk_fk_join(start_table, target_table) and not self.is_pk_fk_join(target_table, start_table):
                n_m_paths.append(join_path)
        return n_m_paths

    def nx_graph(self) -> nx.Graph:
        return self._graph

    def is_free_table(self, table: base.TableReference):
        return self._graph.nodes[table]["free"]

    def is_available_join(self, first_table: base.TableReference, second_table: base.TableReference) -> bool:
        first_free, second_free = self._graph.nodes[first_table]["free"], self._graph.nodes[second_table]["free"]
        return (first_free and not second_free) or (not first_free and second_free)

    def is_pk_fk_join(self, fk_table: base.TableReference, pk_table: base.TableReference) -> bool:
        if not (fk_table, pk_table) in self._graph.edges:
            return False

        predicate: predicates.AbstractPredicate = self._graph.edges[fk_table, pk_table]["predicate"]
        for base_predicate in predicate.base_predicates():
            fk_col = collection_utils.simplify(base_predicate.columns_of(fk_table))
            pk_col = collection_utils.simplify(base_predicate.columns_of(pk_table))
            if self._index_structures[fk_col].is_indexed() and self._index_structures[pk_col].is_primary():
                return True
        return False

    def available_pk_fk_joins_for(self, fk_table: base.TableReference) -> Iterable[JoinPath]:
        if self.initial():
            return [join for join in self.available_join_paths()
                    if self.is_pk_fk_join(join.start_table, join.target_table)
                    or self.is_pk_fk_join(join.target_table, join.start_table)]

        return [join for join in self.available_join_paths() if self.is_pk_fk_join(fk_table, join.target_table)]

    def available_deep_pk_join_paths_for(self, fk_table: base.TableReference,
                                         ordering: Callable[[base.TableReference, dict], int] | None = None
                                         ) -> Iterable[JoinPath]:
        available_joins = nx_utils.nx_bfs_tree(self._graph, fk_table, self._check_pk_fk_join, node_order=ordering)
        join_paths = []
        for join in available_joins:
            current_pk_table: base.TableReference = join[0]
            join_predicate: predicates.AbstractPredicate = join[1]["predicate"]
            current_fk_table = collection_utils.simplify({column.table for column
                                                          in join_predicate.join_partners_of(current_pk_table)})
            join_paths.append(JoinPath(current_fk_table, current_pk_table, join_predicate))
        return join_paths

    def mark_joined(self, table: base.TableReference, join_edge: predicates.AbstractPredicate | None = None) -> None:
        self._graph.nodes[table]["free"] = False
        if not join_edge:
            return

        partner_table = collection_utils.simplify({col.table for col in join_edge.join_partners_of(table)})
        pk_fk_join = self.is_pk_fk_join(table, partner_table)
        fk_pk_join = self.is_pk_fk_join(partner_table, table)

        if pk_fk_join and fk_pk_join:  # PK/PK join
            return

        for col1, col2 in join_edge.join_partners():
            joined_col, partner_col = (col1, col2) if col1.table == table else (col2, col1)
            if pk_fk_join:
                self._index_structures[partner_col].invalidate()
            elif fk_pk_join:
                self._index_structures[joined_col].invalidate()
            else:
                self._index_structures[partner_col].invalidate()
                self._index_structures[joined_col].invalidate()

        if pk_fk_join:
            return

        for table, is_free in self._graph.nodes.data("free"):
            if is_free or table == partner_table:
                continue
            self._invalidate_indexes_on(table)

    def _check_pk_fk_join(self, pk_table: base.TableReference, edge_data: dict) -> bool:
        join_predicate: predicates.AbstractPredicate = edge_data["predicate"]
        for base_predicate in join_predicate.base_predicates():
            fk_table = collection_utils.simplify({column.table for column in base_predicate.join_partners_of(pk_table)})
            if self.is_pk_fk_join(fk_table, pk_table):
                return True
        return False

    def _invalidate_indexes_on(self, table: base.TableReference) -> None:
        for column, index in self._index_structures.items():
            if column.table == table:
                index.invalidate()


class JoinTreeNode(abc.ABC, Container):
    def __init(self, upper_bound: int) -> None:
        self.upper_bound = upper_bound

    @abc.abstractmethod
    def is_join_node(self) -> bool:
        raise NotImplementedError

    @abc.abstractmethod
    def tables(self) -> set[base.TableReference]:
        raise NotImplementedError

    @abc.abstractmethod
    def columns(self) -> set[base.ColumnReference]:
        raise NotImplementedError

    @abc.abstractmethod
    def join_sequence(self) -> Iterable[JoinNode]:
        raise NotImplementedError

    def as_join_tree(self) -> JoinTree:
        return JoinTree(self)

    @abc.abstractmethod
    def __contains__(self, item) -> bool:
        raise NotImplementedError

    @abc.abstractmethod
    def __hash__(self) -> int:
        raise NotImplementedError

    def __repr__(self) -> str:
        return str(self)

    @abc.abstractmethod
    def __str__(self) -> str:
        raise NotImplementedError


class JoinNode(JoinTreeNode):
    def __init__(self, left_child: JoinTreeNode, right_child: JoinTreeNode, *, join_bound: int,
                 join_condition: predicates.AbstractPredicate | None = None, n_m_join: bool = True,
                 n_m_joined_table: base.TableReference | None = None) -> None:
        self.left_child = left_child
        self.right_child = right_child
        self.join_condition = join_condition
        self.n_m_join = n_m_join
        self.n_m_joined_table = n_m_joined_table if self.n_m_join else None
        self.join_bound = join_bound

    def is_join_node(self) -> bool:
        return True

    def tables(self) -> set[base.TableReference]:
        tables = set()
        tables |= self.left_child.tables()
        tables |= self.right_child.tables()
        return tables

    def columns(self) -> set[base.ColumnReference]:
        columns = set(self.join_condition.columns())
        columns |= self.left_child.columns()
        columns |= self.right_child.columns()
        return columns

    def join_sequence(self) -> Iterable[JoinNode]:
        leaf_node = not self.left_child.is_join_node() and not self.right_child.is_join_node()
        if leaf_node:
            return [self]
        sequence = []
        if self.right_child.is_join_node():
            sequence.extend(self.right_child.join_sequence())
        if self.left_child.is_join_node():
            sequence.extend(self.left_child.join_sequence())
        sequence.append(self)
        return sequence

    def __contains__(self, item) -> bool:
        if not isinstance(item, JoinTreeNode):
            return False

        if self == item:
            return True
        return item in self.left_child or item in self.right_child

    def __hash__(self) -> int:
        return hash(tuple([self.left_child, self.right_child, self.join_condition,
                           self.n_m_join, self.n_m_joined_table,
                           self.join_bound]))

    def __eq__(self, other) -> bool:
        return (isinstance(other, type(self))
                and self.left_child == other.left_child
                and self.right_child == other.right_child
                and self.join_condition == other.join_condition
                and self.n_m_join == other.n_m_join
                and self.n_m_joined_table == other.n_m_joined_table
                and self.join_bound == other.join_bound)

    def __str__(self) -> str:
        # perform a right-deep string generation, left branches are subqueries
        left_str = f"({self.left_child})" if self.left_child.is_join_node() else str(self.left_child)
        right_str = str(self.right_child)
        return f"{right_str} ⋈ {left_str}"


class BaseTableNode(JoinTreeNode):
    def __init__(self, table: base.TableReference, cardinality_estimate: int,
                 filter_condition: predicates.AbstractPredicate | None = None) -> None:
        self.table = table
        self.filter = filter_condition
        self.cardinality_estimate = cardinality_estimate

    def is_join_node(self) -> bool:
        return False

    def tables(self) -> set[base.TableReference]:
        return {self.table}

    def columns(self) -> set[base.ColumnReference]:
        return set()

    def join_sequence(self) -> Iterable[JoinNode]:
        return []

    def __contains__(self, item) -> bool:
        return self == item

    def __hash__(self) -> int:
        return hash((self.table, self.filter, self.cardinality_estimate))

    def __eq__(self, other) -> bool:
        return (isinstance(other, type(self))
                and self.table == other.table
                and self.filter == other.filter
                and self.cardinality_estimate == other.cardinality_estimate)

    def __str__(self) -> str:
        return str(self.table)


class JoinTree(Container[JoinTreeNode]):
    """A right-deep join tree abstraction."""

    @staticmethod
    def cross_product_of(*trees: JoinTree) -> JoinTree:
        if not trees:
            raise ValueError("No trees given")
        elif len(trees) == 1:
            return trees[0]
        first_tree, *additional_trees = trees

        current_root = first_tree.root
        for additional_tree in additional_trees:
            cross_product_bound = current_root.upper_bound * additional_tree.root.upper_bound
            current_root = JoinNode(additional_tree.root, current_root, join_bound=cross_product_bound)

        cross_product_tree = JoinTree()
        cross_product_tree.root = current_root
        return cross_product_tree

    @staticmethod
    def for_base_table(table: base.TableReference, base_cardinality: int,
                       filter_predicates: predicates.AbstractPredicate) -> JoinTree:
        root = BaseTableNode(table, base_cardinality, filter_predicates)
        return JoinTree(root)

    def __init__(self, root: JoinTreeNode | None = None) -> None:
        self.root = root

    def join_with_base_table(self, table: base.TableReference, *, base_cardinality: int,
                             join_predicate: predicates.AbstractPredicate | None = None, join_bound: int | None = None,
                             base_filter_predicate: predicates.AbstractPredicate | None = None,
                             n_m_join: bool = True, insert_left: bool = True) -> JoinTree:
        base_node = BaseTableNode(table, base_cardinality, base_filter_predicate)
        if self.is_empty():
            return JoinTree(base_node)
        else:
            left, right = (base_node, self.root) if insert_left else (self.root, base_node)
            new_root = JoinNode(left, right, join_bound=join_bound, join_condition=join_predicate,
                                n_m_join=n_m_join, n_m_joined_table=table)
            return JoinTree(new_root)

    def join_with_subquery(self, subquery: JoinTree, join_predicate: predicates.AbstractPredicate,
                           join_bound: int, *, n_m_join: bool = True, n_m_table: base.TableReference | None = None,
                           insert_left: bool = True) -> JoinTree:
        if self.is_empty():
            return JoinTree(subquery.root)
        left, right = (subquery.root, self.root) if insert_left else (self.root, subquery.root)
        new_root = JoinNode(left, right, join_bound=join_bound, join_condition=join_predicate,
                            n_m_join=n_m_join, n_m_joined_table=n_m_table)
        return JoinTree(new_root)

    def is_empty(self) -> bool:
        return self.root is None

    def tables(self) -> set[base.TableReference]:
        if self.is_empty():
            return set()
        return self.root.tables()

    def columns(self) -> set[base.ColumnReference]:
        if self.is_empty():
            return set()
        return self.root.columns()

    def join_sequence(self) -> Iterable[JoinNode]:
        if self.is_empty():
            return []
        return self.root.join_sequence()

    def _get_upper_bound(self) -> int:
        if self.is_empty():
            raise errors.StateError("Join tree is empty")
        return self.root.upper_bound

    upper_bound = property(_get_upper_bound)

    def __contains__(self, item: object) -> bool:
        if not isinstance(item, JoinTree | JoinTreeNode):
            return False

        other_tree = item if isinstance(item, JoinTree) else JoinTree(item)
        if self.is_empty() and not item.is_empty():
            return False
        elif item.is_empty():
            return True

        return item.root in self.root

    def __hash__(self) -> int:
        return hash(self.root)

    def __eq__(self, other) -> bool:
        return isinstance(other, type(self)) and self.root == other.root

    def __repr__(self) -> str:
        return str(self)

    def __str__(self) -> str:
        if self.is_empty():
            return "[EMPTY JOIN TREE]"
        return str(self.root)
