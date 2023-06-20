"""Provides data structures that are used throughout the optimizer implementation."""
from __future__ import annotations

import collections
from collections.abc import Callable, Collection, Iterable, Iterator, Mapping
from dataclasses import dataclass
from typing import Optional

import networkx as nx

from postbound.qal import base, predicates, qal, transform
from postbound.db import db
from postbound.util import collections as collection_utils, networkx as nx_utils


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


@dataclass(frozen=True)
class TableInfo:
    """Captures information about the state of tables in the join graph."""
    free: bool
    index_info: Collection[IndexInfo]


_PredicateMap = collections.defaultdict[frozenset[base.TableReference], list[predicates.AbstractPredicate]]
"""Type alias for an (internally used) predicate map"""


class JoinGraph(Mapping[base.TableReference, TableInfo]):
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

    def joined_tables(self) -> frozenset[base.TableReference]:
        """Provides all non-free tables in the join graph."""
        return frozenset(table for table in self if not self.is_free_table(table))

    def all_joins(self) -> Iterable[tuple[base.TableReference, base.TableReference]]:
        """Provides all edges in the join graph, no matter whether they are available or not."""
        return list(self._graph.edges)

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

        If no join predicate is supplied, it is inferred from the query predicates.
        """

        # TODO: check, if we actually need to handle transient index updates here as well

        self._graph.nodes[table]["free"] = False
        if len(self.joined_tables()) == 1:
            return

        join_edge = join_edge if join_edge else self.query.predicates().joins_between(table, self.joined_tables())
        if not join_edge:
            # We still need this check even though we already know that there are at least two tables joined, since
            # these two tables might have nothing to do with each other (e.g. different components in the join graph)
            return

        partner_tables = {col.table for col in join_edge.join_partners_of(table)}
        for partner_table in partner_tables:
            pk_fk_join = self.is_pk_fk_join(table, partner_table)
            fk_pk_join = self.is_pk_fk_join(partner_table, table)

            if pk_fk_join and fk_pk_join:  # PK/PK join
                continue

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
                continue

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

    def _index_info_for(self, table: base.TableReference) -> Collection[IndexInfo]:
        """Provides all index info for the given table (i.e. for each column that belongs to the table)."""
        return [info for info in self._index_structures.values() if info.column.belongs_to(table)]

    def __len__(self) -> int:
        return len(self._graph)

    def __iter__(self) -> Iterator[base.TableReference]:
        return iter(self._graph.nodes)

    def __contains__(self, x: object) -> bool:
        return x in self._graph.nodes

    def __getitem__(self, key: base.TableReference) -> TableInfo:
        if key not in self:
            raise KeyError(f"Table {key} is not part of the join graph")
        free = self.is_free_table(key)
        index_info = self._index_info_for(key)
        return TableInfo(free, index_info)
