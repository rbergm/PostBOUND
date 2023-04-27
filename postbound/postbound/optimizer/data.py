"""Provides data structures that are used throughout the optimizer implementation."""
from __future__ import annotations

import abc
import collections
from collections.abc import Container, Collection
from dataclasses import dataclass
from typing import Callable, Iterable, Optional

import networkx as nx
import numpy as np

from postbound.qal import base, predicates, qal, transform
from postbound.db import db
from postbound.optimizer.physops import operators as physops
from postbound.optimizer.planmeta import hints as plan_param
from postbound.util import collections as collection_utils, errors, networkx as nx_utils


@dataclass
class JoinPath:
    """A `JoinPath` models the join between two tables where one table is part of an intermediate result.

    The `start_table` is the table that is already included in the intermediate result, whereas the `target_table` is
    the table being joined. The `join_condition` annotates the actual join to execute.
    """
    start_table: base.TableReference
    target_table: base.TableReference
    join_condition: predicates.AbstractPredicate | None = None

    def tables(self) -> Collection[base.TableReference]:
        """Provides the tables that are joined."""
        return [self.start_table, self.target_table]

    def spans_table(self, table: base.TableReference) -> bool:
        """Checks, whether the given table is either the start, or the target table in this path."""
        return table == self.start_table or table == self.target_table

    def flip_direction(self) -> JoinPath:
        """Provides a new join path with swapped roles for start and target tables."""
        return JoinPath(self.target_table, self.start_table, join_condition=self.join_condition)

    def __repr__(self) -> str:
        return str(self)

    def __str__(self) -> str:
        return f"{self.start_table} ⋈ {self.target_table} ({self.join_condition})"


class IndexInfo:
    """The `IndexInfo` captures index structures that are defined over database columns as well as their lifecycle.

    Note that this does not support multidimensional indexes for now.
    """

    @staticmethod
    def primary_index(column: base.ColumnReference) -> IndexInfo:
        """Constructs a primary index for the given column."""
        return IndexInfo(column, "primary")

    @staticmethod
    def secondary_index(column: base.ColumnReference) -> IndexInfo:
        """Constructs a secondary index for the given column."""
        return IndexInfo(column, "secondary")

    @staticmethod
    def no_index(column: base.ColumnReference) -> IndexInfo:
        """Constructs index info for a column that does not have any index."""
        return IndexInfo(column, "none")

    @staticmethod
    def generate_for(column: base.ColumnReference, db_schema: db.DatabaseSchema) -> IndexInfo:
        """Creates the appropriate index info for the given column according to the available schema information."""
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
        """Checks, whether this is a valid primary index."""
        return not self.is_invalid and self.index_type == "primary"

    def is_secondary(self) -> bool:
        """Checks, whether this is a valid secondary index."""
        return not self.is_invalid and self.index_type == "secondary"

    def is_indexed(self) -> bool:
        """Checks, whether there is any valid index structure defined for the column."""
        return self.is_primary() or self.is_secondary()

    def can_pk_fk_join(self, other: IndexInfo) -> bool:
        """Checks, whether the given columns could be joined using a primary key/foreign key join.

        This method does not restrict the direction of such a join, i.e. each column could act as the primary key or
        foreign key. Likewise, no datatype checks are performed and it is assumed that a database system would be
        able to actually join the two columns involved.
        """
        if not self.is_indexed() or not other.is_indexed():
            return False

        if self.is_secondary() and other.is_secondary():
            return False

        # all other cases have at least one primary key index available
        return True

    def invalidate(self) -> None:
        """Marks the index as invalid if necessary.

        Once a table is included in an intermediate join result, the index structures of its columns most likely
        become invalid, and it is no longer possible to use the index to query for specific tuples (because the
        occurrences of the individual tuples are multiplied when executing the join). This method can be used to model
        the lifecycle of index structures.
        """
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


_PredicateMap = collections.defaultdict[frozenset[base.TableReference], list[predicates.AbstractPredicate]]
"""Type alias for an (internally used) predicate map"""


class JoinGraph:
    """The `JoinGraph` models the connection between different tables of in a query.

    All tables that are referenced in the query are represented as the nodes of the graph. If two tables are joined
    via a join predicate in the SQL query, they will be linked with an edge in the join graph. This graph is further
    annotated by the join predicate. Additionally, the join graph stores index information for each of the relevant
    columns in the query.

    A join graph is a mutable structure, i.e. it also models the current state of the optimizer once specific tables
    have been included in an intermediate join result.

    The wording of the different join graph methods distinguishes three states of joins (and correspondingly tables):

    - a join might be `free`, if at least one of the corresponding tables have not been marked as joined, yet
    - a join might be `available`, if it is `free` and one of the tables is already included in some intermediate join
    - a join is `consumed`, if its no longer `free`

    A further distinction is made between n:m joins and primary key/foreign key joins and information about the
    available joins of each type can be queried easily. To determine the precise join types, a join graph needs to
    access the database schema.

    By calling the `mark_joined` method,  the state of individual joins and their corresponding tables might change.
    This also means that former primary key/foreign key joins might become n:m joins (which is the case exactly when
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
        predicate_map: _PredicateMap = collections.defaultdict(list)
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
        """Checks, whether the join graph has not been modified yet and is still in its initial state."""
        return all(is_free for __, is_free in self._graph.nodes.data("free"))

    def contains_cross_products(self) -> bool:
        """Checks, whether there are any cross products in the input query.

        A cross product is a join between tables without a restricting join predicate. Note that this is only the case
        if those tables are also not linked via a sequence of join predicates with other tables.
        """
        return not nx.is_connected(self._graph)

    def contains_free_tables(self) -> bool:
        """Checks, whether there is at least one more free tables remaining in the graph."""
        return any(is_free for __, is_free in self._graph.nodes.data("free"))

    def contains_free_n_m_joins(self) -> bool:
        """Checks, whether there is at least one more free n:m join remaining in the graph."""
        is_first_join = self.initial()
        for first_tab, second_tab, predicate in self._graph.edges.data("predicate"):
            if not self.is_available_join(first_tab, second_tab) and not is_first_join:
                continue
            for first_col, second_col in predicate.join_partners():
                if not self._index_structures[first_col].can_pk_fk_join(self._index_structures[second_col]):
                    return True
        return False

    def count_consumed_tables(self) -> int:
        """The number of tables that have been joined already (or at least included in the intermediate result).

        This number might be 1 if only the initial tables has been selected.
        """
        return len([is_free for __, is_free in self._graph.nodes.data("free") if not is_free])

    def join_components(self) -> Iterable[JoinGraph]:
        """Provides all components of the join graph.

        A component is a subgraph of the original join graph, such that the subgraph is connected but there was no
        edge between nodes from different sub-graphs. This corresponds to the parts of the query that have to be joined
        via a cross product.
        """
        components = []
        for component in nx.connected_components(self._graph):
            component_query = transform.extract_query_fragment(self.query, component)
            components.append(JoinGraph(component_query, self._db_schema))
        return components

    def available_join_paths(self) -> Iterable[JoinPath]:
        """Provides all joins that can be executed in the current join graph.

        The precise output of this method depends on the current state of the join graph. If none of the tables in
        the join graph has been joined already (i.e. `join_graph.initial()` returns `True`), this method returns all
        possible joins. The assignment of start table and target table is arbitrary in that case.
        Otherwise, this method returns all join paths where the source table is part of the intermediate result (i.e.
        it has been joined already) and the target table is still free.
        """
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
        """Returns exactly those join paths from `available_join_paths` that correspond to n:m joins."""
        n_m_paths = []
        for join_path in self.available_join_paths():
            start_table, target_table = join_path.start_table, join_path.target_table
            if not self.is_pk_fk_join(start_table, target_table) and not self.is_pk_fk_join(target_table, start_table):
                n_m_paths.append(join_path)
        return n_m_paths

    def available_join_paths_for(self, free_table: base.TableReference) -> Iterable[JoinPath]:
        """Returns all possible joins of the given free table with consumed tables.

        The returned join paths will have the consumed tables as start tables and the free table as target table.

        This method adapts the behavior of `available_join_paths` for initial join graphs.
        """
        return [path if free_table == path.start_table else path.flip_direction()
                for path in self.available_join_paths() if path.spans_table(free_table)]

    def nx_graph(self) -> nx.Graph:
        """Provides the underlying graph object for this join graph."""
        return self._graph

    def is_free_table(self, table: base.TableReference):
        """Checks, whether the given table is still free in this join graph."""
        return self._graph.nodes[table]["free"]

    def joins_tables(self, first_table: base.TableReference, second_table: base.TableReference) -> bool:
        """Checks, whether the join graph contains an edge between the given tables.

        This check does not require the join in question to be available (this is what `is_available_join` is for).
        """
        return (first_table, second_table) in self._graph.edges

    def is_available_join(self, first_table: base.TableReference, second_table: base.TableReference) -> bool:
        """Checks, whether the join between the supplied tables is still available."""
        first_free, second_free = self._graph.nodes[first_table]["free"], self._graph.nodes[second_table]["free"]
        valid_join = self.joins_tables(first_table, second_table)
        return valid_join and (first_free and not second_free) or (not first_free and second_free)

    def is_pk_fk_join(self, fk_table: base.TableReference, pk_table: base.TableReference) -> bool:
        """Checks, whether the join between the supplied tables is a primary key/foreign key join.

        This check does not require the indicated join to be available.
        """
        if not self.joins_tables(fk_table, pk_table):
            return False

        predicate: predicates.AbstractPredicate = self._graph.edges[fk_table, pk_table]["predicate"]
        for base_predicate in predicate.base_predicates():
            fk_col = collection_utils.simplify(base_predicate.columns_of(fk_table))
            pk_col = collection_utils.simplify(base_predicate.columns_of(pk_table))
            if self._index_structures[fk_col].is_indexed() and self._index_structures[pk_col].is_primary():
                return True
        return False

    def is_n_m_join(self, first_table: base.TableReference, second_table: base.TableReference) -> bool:
        """Checks, whether the join between the supplied tables is an n:m join.

        This check does not require the indicated join to be available.
        """
        return (self.joins_tables(first_table, second_table)
                and not self.is_pk_fk_join(first_table, second_table)
                and not self.is_pk_fk_join(second_table, first_table))

    def available_pk_fk_joins_for(self, fk_table: base.TableReference) -> Iterable[JoinPath]:
        """Provides all available primary key/foreign key joins with `fk_table`.

        The given table acts as the foreign key table in all returned join paths. However, this method does not
        restrict, whether the foreign key tables is assigned to the start or target table in the join path (this
        depends on the behavior of `available_join_paths` for the current join graph).
        """
        if self.initial():
            return [join for join in self.available_join_paths()
                    if self.is_pk_fk_join(join.start_table, join.target_table)
                    or self.is_pk_fk_join(join.target_table, join.start_table)]

        return [join for join in self.available_join_paths() if self.is_pk_fk_join(fk_table, join.target_table)]

    def available_deep_pk_join_paths_for(self, fk_table: base.TableReference,
                                         ordering: Callable[[base.TableReference, dict], int] | None = None
                                         ) -> Iterable[JoinPath]:
        """Provides all available pk/fk joins with the given table, as well as follow-up pk/fk joins.

        In contrast to the `available_pk_fk_joins_for` method, this method does not only return direct joins
        between the foreign key table, but augments its output in the following way: suppose the `fk_table` is pk/fk
        joined with a primary key table `t`. Then, this method also includes all joins of `t` with additional tables
        `t'`, such that `t ⋈ t'` is once again a primary key/foreign key join, but this time with `t` acting as the
        foreign key and `t'` as the primary key. This procedure is repeated for all `t'` tables recursively until no
        more primary key/foreign key joins are available.

        Essentially, this is equivalent to performing a breadth-first search on all (directed) primary key/foreign key
        joins, starting at `fk_table`. The sequence in which joins on the same level are placed into the resulting
        iterable can be customized via the `ordering` parameter. This callable receives the current primary key table
        and the edge data as input and produces a numerical position weight as output (smaller values meaning earlier
        placement). The provided edge data contains the join predicate under the `predicate` key. Using the join
        predicate, the join partner (i.e. the foreign key table) can be retrieved.
        """
        available_joins = nx_utils.nx_bfs_tree(self._graph, fk_table, self._check_pk_fk_join, node_order=ordering)
        join_paths = []
        for join in available_joins:
            current_pk_table: base.TableReference = join[0]
            join_predicate: predicates.AbstractPredicate = join[1]["predicate"]
            current_fk_table = collection_utils.simplify({column.table for column
                                                          in join_predicate.join_partners_of(current_pk_table)})
            join_paths.append(JoinPath(current_fk_table, current_pk_table, join_predicate))
        return join_paths

    def join_partners_from(self, table: base.TableReference,
                           candidate_tables: Iterable[base.TableReference]) -> set[base.TableReference]:
        """Provides exactly those tables from the candidates that are joined with the given table.

        This check does not require the joins in question to be available. The existence of a join edge is sufficient.
        """
        candidate_tables = set(candidate_tables)
        return set(neighbor for neighbor in self._graph.adj[table].keys() if neighbor in candidate_tables)

    def join_predicates_between(self, first_tables: base.TableReference | Iterable[base.TableReference],
                                second_tables: Optional[base.TableReference | Iterable[base.TableReference]] = None
                                ) -> Collection[predicates.AbstractPredicate]:
        """Provides all defined join predicates between the first tables with the second tables.

        If `second_tables` is `None`, returns all join predicates within the first tables instead.
        """
        first_tables = collection_utils.enlist(first_tables)
        second_tables = collection_utils.enlist(second_tables) if second_tables else first_tables
        matching_predicates = set()

        for first_table in first_tables:
            for second_table in second_tables:
                join_predicate = self._fetch_join_predicate(first_table, second_table)
                if join_predicate:
                    matching_predicates.add(join_predicate)

        return matching_predicates

    def mark_joined(self, table: base.TableReference, join_edge: predicates.AbstractPredicate | None = None) -> None:
        """Updates the join graph to include the given table in the intermediate result.

        This procedure also changes the available index structures according to the kind of join that was executed.
        This is determined based on the current state of the join graph, as well as the supplied join predicate.

        If no join predicate is supplied, no index structures are updated.
        """
        # TODO: allow auto-update

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
        """Checks, whether the given table acts as a primary key in the join as indicated by the join graph edge."""
        join_predicate: predicates.AbstractPredicate = edge_data["predicate"]
        for base_predicate in join_predicate.base_predicates():
            fk_table = collection_utils.simplify({column.table
                                                  for column in base_predicate.join_partners_of(pk_table)})
            if self.is_pk_fk_join(fk_table, pk_table):
                return True
        return False

    def _invalidate_indexes_on(self, table: base.TableReference) -> None:
        """Invalidates all indexes on all columns that belong to the given table."""
        for column, index in self._index_structures.items():
            if column.table == table:
                index.invalidate()

    def _fetch_join_predicate(self, first_table: base.TableReference,
                              second_table: base.TableReference) -> Optional[predicates.AbstractPredicate]:
        """Provides the join predicate between the given tables if there is one."""
        if (first_table, second_table) not in self._graph.edges:
            return None
        return self._graph.edges[first_table, second_table]["predicate"]


class JoinTreeNode(abc.ABC, Container):
    """The fundamental type to construct a join tree. This node contains the actual entries/data.

    A join tree distinguishes between two types of nodes: join nodes which act as intermediate nodes that join
    together their child nodes and base table nodes that represent a scan over a base table. These nodes act as leaves
    in the join tree.

    The `JoinTreeNode` describes the behavior that is common to both node types.
    """

    def __init__(self, upper_bound: int) -> None:
        self.upper_bound = upper_bound

    @abc.abstractmethod
    def is_join_node(self) -> bool:
        """Checks, whether this node is a join node."""
        raise NotImplementedError

    def is_leaf_table_node(self) -> bool:
        """Checks, whether this node is a base table node."""
        return not self.is_join_node()

    @abc.abstractmethod
    def tables(self) -> set[base.TableReference]:
        """Provides all tables that are present in the subtree under and including this node."""
        raise NotImplementedError

    @abc.abstractmethod
    def columns(self) -> set[base.ColumnReference]:
        """Provides all columns that are present in the subtree under and including this node."""
        raise NotImplementedError

    @abc.abstractmethod
    def join_sequence(self) -> Collection[JoinNode]:
        """Provides all joins under and including this node in a right-deep manner.

        If this node is a base table node, the returned container will be empty.
        """
        raise NotImplementedError

    @abc.abstractmethod
    def as_list(self) -> list:
        """Provides the selected join order as a nested list.

        The table of each base table node will be contained directly in the join order. Each join node will be
        represented as a list of the list-representations of its child nodes.

        For example, the join order R ⋈ (S ⋈ T) will be represented as `[R, [S, T]]`.
        """
        raise NotImplementedError

    def as_join_tree(self) -> JoinTree:
        """Creates a new join tree with this node as root and all children as sub-nodes."""
        return JoinTree(self)

    @abc.abstractmethod
    def count_cross_product_joins(self) -> int:
        """Counts the number of joins below and including this node that do not have an attached join predicate."""
        raise NotImplementedError

    @abc.abstractmethod
    def homomorphic_hash(self) -> int:
        """
        Calculates a hash value that is independent of join directions, i.e. R ⋈ S and S ⋈ R have the same hash value.
        """
        raise NotImplementedError

    @abc.abstractmethod
    def base_table(self, traverse_right: bool = True) -> base.TableReference:
        """Provides the left-most or right-most table in the join tree."""
        raise NotImplementedError

    @abc.abstractmethod
    def inspect(self, *, indentation: int = 0) -> str:
        """Produces a human-readable structure that describes the structure of this join tree."""
        raise NotImplementedError

    @abc.abstractmethod
    def __contains__(self, item) -> bool:
        raise NotImplementedError

    @abc.abstractmethod
    def __len__(self) -> int:
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
    """The `JoinNode` represents an intermediate node in a join tree.

    Its children may either be join nodes themselves, or base table nodes.

    The join node is constructed based on its child nodes. It can further be annotated by a cardinality estimate
    (be it an upper bound or not) of how many result tuples the join is estimated to produce, the join condition that
    is used to execute the join, as well as whether the join represents an n:m join.

    The last annotation should be interpreted as follows: all nodes of a join tree can be assigned to one of two
    different categories: they might be part of the intermediate result already, or they are new tables that are
    becoming part of the intermediate result right now (i.e. via this very join). In the second case, this is what
    the `n_m_joined_table` indicates.

    Note that for now, this assumes that a join only happens between one freshly joined table and an arbitrary number
    of intermediate tables (at least one). The special case of a join node putting together two partitions of the
    join graph via a combination of different join predicates is not supported, yet.
    """

    def __init__(self, left_child: JoinTreeNode, right_child: JoinTreeNode, *, join_bound: int,
                 join_condition: predicates.AbstractPredicate | None = None, n_m_join: bool = True,
                 n_m_joined_table: base.TableReference | None = None) -> None:
        # TODO: the n_m_joined table does not work for joins over more than one target table
        super().__init__(join_bound)
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

    def as_list(self) -> list:
        return [self.right_child.as_list(), self.left_child.as_list()]

    def count_cross_product_joins(self) -> int:
        own_cross_product = 1 if not self.join_condition else 0
        return (own_cross_product
                + self.left_child.count_cross_product_joins()
                + self.right_child.count_cross_product_joins())

    def homomorphic_hash(self) -> int:
        left_hash = self.left_child.homomorphic_hash()
        right_hash = self.right_child.homomorphic_hash()
        child_hash = hash((left_hash, right_hash)) if left_hash < right_hash else hash((right_hash, left_hash))
        return hash((self.join_condition, child_hash))

    def base_table(self, traverse_right: bool = True) -> base.TableReference:
        return (self.right_child.base_table(traverse_right) if traverse_right
                else self.left_child.base_table(traverse_right))

    def inspect(self, *, indentation: int = 0) -> str:
        padding = " " * indentation
        prefix = f"{padding}<- " if padding else ""
        own_inspection = f"{prefix}JOIN ON {self.join_condition}" if self.join_condition else f"{prefix}CROSS JOIN"
        left_inspection = self.left_child.inspect(indentation=indentation + 2)
        right_inspection = self.right_child.inspect(indentation=indentation + 2)
        return "\n".join([own_inspection, left_inspection, right_inspection])

    def __contains__(self, item) -> bool:
        if not isinstance(item, JoinTreeNode):
            return False

        if self == item:
            return True
        return item in self.left_child or item in self.right_child

    def __len__(self) -> int:
        return len(self.left_child) + len(self.right_child)

    def __hash__(self) -> int:
        return hash(tuple([self.left_child, self.right_child, self.join_condition,
                           self.n_m_join, self.n_m_joined_table]))

    def __eq__(self, other) -> bool:
        return (isinstance(other, type(self))
                and self.left_child == other.left_child
                and self.right_child == other.right_child
                and self.join_condition == other.join_condition
                and self.n_m_join == other.n_m_join
                and self.n_m_joined_table == other.n_m_joined_table)

    def __str__(self) -> str:
        # perform a right-deep string generation, left branches are subqueries
        left_str = f"({self.left_child})" if self.left_child.is_join_node() else str(self.left_child)
        right_str = str(self.right_child)
        return f"{right_str} ⋈ {left_str}"


class BaseTableNode(JoinTreeNode):
    """A `BaseTableNode` represents a leaf node in a join tree.

    It corresponds to a scan operation on the given base table.

    The node can be further annotated by a cardinality estimate of how many rows the scan on the base table will
    produce (independent of the specific scan algorithm used), as well as the filter predicate that should be applied
    to the base table.
    """

    def __init__(self, table: base.TableReference, cardinality_estimate: int,
                 filter_condition: predicates.AbstractPredicate | None = None) -> None:
        super().__init__(cardinality_estimate)
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

    def as_list(self) -> list:
        return self.table

    def count_cross_product_joins(self) -> int:
        return 0

    def homomorphic_hash(self) -> int:
        return hash(self)

    def base_table(self, traverse_right: bool = True) -> base.TableReference:
        return self.table

    def inspect(self, *, indentation: int = 0) -> str:
        padding = " " * indentation
        prefix = f"{padding} <- " if padding else ""
        return f"{prefix} SCAN :: {self.table}"

    def __contains__(self, item) -> bool:
        return self == item

    def __len__(self) -> int:
        return 1

    def __hash__(self) -> int:
        return hash((self.table, self.filter))

    def __eq__(self, other) -> bool:
        return (isinstance(other, type(self))
                and self.table == other.table
                and self.filter == other.filter)

    def __str__(self) -> str:
        return str(self.table)


class JoinTree(Container[JoinTreeNode]):
    """The `JoinTree` describes in which sequence joins should be executed for a given query.

    A join tree allows for iterative expansion by including additional joins. At the same time, each individual join
    tree instance should be treated as immutable, i.e. the expansion of a join tree results in a new join tree
    instance.

    Join trees can be left-deep, right-deep or bushy. By default, new joins are inserted such that a right-deep
    join tree is created.

    Other than the join sequence, each join tree can also store information about the join and filter predicates that
    should be applied to base tables and joins, cardinality estimates and upper bounds of the tables and joins and
    even an assignment of physical operators to the different nodes in the join tree.
    """

    @staticmethod
    def cross_product_of(*trees: JoinTree) -> JoinTree:
        """Generates a new join tree with by applying a cross product over the given join trees."""
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
        """Generates a new join tree that for now only includes the given base table."""
        root = BaseTableNode(table, base_cardinality, filter_predicates)
        return JoinTree(root)

    @staticmethod
    def joining(left_tree: JoinTree, right_tree: JoinTree, *,
                join_condition: Optional[predicates.AbstractPredicate] = None,
                join_bound: int = np.nan, n_m_join: bool = True,
                n_m_joined_table: Optional[base.TableReference] = None) -> JoinTree:
        """Constructs a new join tree that joins the two input trees."""
        if left_tree.is_empty():
            return right_tree
        if right_tree.is_empty():
            return left_tree

        join_node = JoinNode(left_tree.root, right_tree.root, join_bound=join_bound, join_condition=join_condition,
                             n_m_join=n_m_join, n_m_joined_table=n_m_joined_table)

        if left_tree.operator_assignment and right_tree.operator_assignment:
            operator_assignment = left_tree.operator_assignment.merge_with(right_tree.operator_assignment)
        else:
            operator_assignment = (left_tree.operator_assignment if left_tree.operator_assignment
                                   else right_tree.operator_assignment)

        if left_tree.plan_parameterization and right_tree.plan_parameterization:
            plan_parameterization = left_tree.plan_parameterization.merge_with(right_tree.plan_parameterization)
        else:
            plan_parameterization = (left_tree.plan_parameterization if left_tree.plan_parameterization
                                     else right_tree.plan_parameterization)

        return JoinTree(join_node, operator_assignment=operator_assignment, plan_parameters=plan_parameterization)

    def __init__(self, root: JoinTreeNode | None = None, *,
                 operator_assignment: Optional[physops.PhysicalOperatorAssignment] = None,
                 plan_parameters: Optional[plan_param.PlanParameterization] = None) -> None:
        self.root = root
        self.operator_assignment = operator_assignment
        self.plan_parameterization = plan_parameters

    def join_with_base_table(self, table: base.TableReference, *, base_cardinality: int,
                             join_predicate: predicates.AbstractPredicate | None = None, join_bound: int | None = None,
                             base_filter_predicate: predicates.AbstractPredicate | None = None,
                             n_m_join: bool = True, insert_left: bool = True) -> JoinTree:
        """Constructs a new join tree that includes an additional join with the given table.

        The base table can be annotated with various meta information (see `BaseTableNode` for details). Likewise,
        the actual join can be annotated according to the `JoinNode`.

        The `insert_left` parameter determines the structure of the resulting join tree. By default, the new node will
        be inserted to the left of the current join tree, thereby creating a right-deep join tree. If `insert_left` is
        false, the join will be inserted to the right of the current tree instead, thereby creating a left-deep join
        tree.
        """
        base_node = BaseTableNode(table, base_cardinality, base_filter_predicate)
        if self.is_empty():
            return JoinTree(base_node,
                            operator_assignment=self.operator_assignment,
                            plan_parameters=self.plan_parameterization)
        else:
            left, right = (base_node, self.root) if insert_left else (self.root, base_node)
            new_root = JoinNode(left, right, join_bound=join_bound, join_condition=join_predicate,
                                n_m_join=n_m_join, n_m_joined_table=table)
            return JoinTree(new_root,
                            operator_assignment=self.operator_assignment,
                            plan_parameters=self.plan_parameterization)

    def join_with_subquery(self, subquery: JoinTree, join_predicate: predicates.AbstractPredicate,
                           join_bound: int, *, n_m_join: bool = True, n_m_table: base.TableReference | None = None,
                           insert_left: bool = True) -> JoinTree:
        """Constructs a new join tree that includes an additional join with the given subtree.

        The join can be further annotated with metadata as described in `JoinNode`.

        The `insert_left` parameter determines the structure of the resulting join tree. By default, the new node will
        be inserted to the left of the current join tree, thereby creating a right-deep join tree. If `insert_left` is
        false, the join will be inserted to the right of the current tree instead, thereby creating a left-deep join
        tree.
        """
        if self.is_empty():
            return JoinTree(subquery.root,
                            operator_assignment=self.operator_assignment,
                            plan_parameters=self.plan_parameterization)
        left, right = (subquery.root, self.root) if insert_left else (self.root, subquery.root)
        new_root = JoinNode(left, right, join_bound=join_bound, join_condition=join_predicate,
                            n_m_join=n_m_join, n_m_joined_table=n_m_table)
        return JoinTree(new_root,
                        operator_assignment=self.operator_assignment,
                        plan_parameters=self.plan_parameterization)

    def is_empty(self) -> bool:
        """Checks, whether there is at least one table in the join tree."""
        return self.root is None

    def tables(self, frozen: bool = False) -> set[base.TableReference] | frozenset[base.TableReference]:
        """Provides all tables that are currently contained in the join tree."""
        if self.is_empty():
            return set()
        return frozenset(self.root.tables()) if frozen else self.root.tables()

    def columns(self) -> set[base.ColumnReference]:
        """Provides all columns that are currently referenced by the join and filter predicates of this join tree."""
        if self.is_empty():
            return set()
        return self.root.columns()

    def join_sequence(self) -> Collection[JoinNode]:
        """Provides a right-deep iteration of all join nodes in the join tree. See `JoinTreeNode`."""
        if self.is_empty():
            return []
        return self.root.join_sequence()

    def as_list(self) -> list:
        """Provides the selected join order as a nested list.

        The table of each base table node will be contained directly in the join order. Each join node will be
        represented as a list of the list-representations of its child nodes.

        For example, the join order R ⋈ (S ⋈ T) will be represented as `[R, [S, T]]`.
        """
        if self.is_empty():
            return []
        return self.root.as_list()

    def base_table(self, direction: str = "right") -> base.TableReference:
        """Provides the left-most or right-most table in the join tree."""
        if direction not in {"right", "left"}:
            raise ValueError(f"Direction must be either 'left' or 'right', not '{direction}'")
        self._assert_not_empty()
        return self.root.base_table(direction == "right")

    def count_cross_product_joins(self) -> int:
        """Counts the number of joins in this tree that do not have an attached join predicate."""
        return 0 if self.is_empty() else self.root.count_cross_product_joins()

    def homomorphic_hash(self) -> int:
        """
        Calculates a hash value that is independent of join directions, i.e. R ⋈ S and S ⋈ R have the same hash value.
        """
        return self.root.homomorphic_hash() if self.root else hash(self.root)

    def inspect(self) -> str:
        """Produces a human-readable structure that describes the structure of this join tree."""
        if self.is_empty():
            return "[EMPTY JOIN TREE]"
        return self.root.inspect()

    def _get_upper_bound(self) -> int:
        """Provides the upper bound associated with this tree, raises an error if empty."""
        if self.is_empty():
            raise errors.StateError("Join tree is empty")
        return self.root.upper_bound

    def _assert_not_empty(self) -> None:
        """Raises an error if this tree is empty."""
        if self.is_empty():
            raise errors.StateError("Empty join tree")

    upper_bound = property(_get_upper_bound)
    """Provides the current upper bound or cardinality estimate of this join tree (i.e. its root node)."""

    def __contains__(self, item: object) -> bool:
        if not isinstance(item, (JoinTree, JoinTreeNode)):
            return False

        other_tree = item if isinstance(item, JoinTree) else JoinTree(item)
        if self.is_empty() and not item.is_empty():
            return False
        elif other_tree.is_empty():
            return True

        return other_tree.root in self.root

    def __len__(self) -> int:
        return 0 if self.is_empty() else len(self.root)

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
