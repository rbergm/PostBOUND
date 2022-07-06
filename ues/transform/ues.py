
import abc
import collections
import operator
from typing import Dict, List, Set, Union, Tuple
import warnings

import matplotlib.pyplot as plt
import networkx as nx
import numpy as np

from transform import db, mosp, util


class JoinCardinalityEstimator(abc.ABC):
    """A cardinality estimator is capable of calculating an upper bound of the number result tuples for a given join.

    How this is achieved precisely is up to the concrete estimator.
    """

    @abc.abstractmethod
    def calculate_upper_bound(self) -> int:
        return NotImplemented


class DefaultUESCardinalityEstimator(JoinCardinalityEstimator):
    def __init__(self, query: mosp.MospQuery):
        # TODO: determine maximum frequency values for each attribute
        self.query = query

    def calculate_upper_bound(self) -> int:
        # TODO: implementation
        return 0


class TopkUESCardinalityEstimator(JoinCardinalityEstimator):
    pass


class BaseCardinalityEstimator(abc.ABC):

    @abc.abstractmethod
    def estimate_rows(self, predicate: Union[mosp.MospPredicate, mosp.CompoundMospFilterPredicate], *,
                      dbs: db.DBSchema = db.DBSchema.get_instance()) -> int:
        return NotImplemented

    def all_tuples(self, table: db.TableRef, *, dbs: db.DBSchema = db.DBSchema.get_instance()) -> int:
        return dbs.count_tuples(table)


class PostgresCardinalityEstimator(BaseCardinalityEstimator):
    def estimate_rows(self, predicate: Union[mosp.MospPredicate, mosp.CompoundMospFilterPredicate], *,
                      dbs: db.DBSchema = db.DBSchema.get_instance()) -> int:
        return predicate.estimate_result_rows(sampling=False, dbs=dbs)


class SamplingCardinalityEstimator(BaseCardinalityEstimator):
    def estimate_rows(self, predicate: Union[mosp.MospPredicate, mosp.CompoundMospFilterPredicate], *,
                      dbs: db.DBSchema = db.DBSchema.get_instance()) -> int:
        return predicate.estimate_result_rows(sampling=True, sampling_pct=25, dbs=dbs)


def _is_pk_fk_join(join: mosp.MospPredicate, *, dbs: db.DBSchema = db.DBSchema.get_instance()) -> bool:
    first_attr, second_attr = join.parse_left_attribute(), join.parse_right_attribute()
    pk, fk = None, None

    if dbs.is_primary_key(first_attr):
        pk = first_attr
    elif dbs.is_primary_key(second_attr):
        if pk:
            warnings.warn("PK/PK join found: {}".format(join))
        pk = second_attr

    if dbs.has_secondary_idx_on(first_attr):
        fk = first_attr
    elif dbs.has_secondary_idx_on(second_attr):
        fk = second_attr

    if pk is None or fk is None:
        return {"pk_fk_join": False}
    return {"pk_fk_join": True, "pk": pk, "fk": fk}


class _EqualityJoinView:
    """A join view represents a join from the 'perspective' of a certain table.

    Consider a join R.a = S.b. In contrast to normal SQL where this join would be equal to S.b = R.a, a join view
    considers the attribute's order. More specifically, a join view takes the perspective of the left-hand side of the
    join and considers the right-hand side its join partner.
    """

    @staticmethod
    def aggregate_view_list(join_views: List["_EqualityJoinView"]
                            ) -> Dict[db.TableRef, List[List["_EqualityJoinView"]]]:

        # step 1: turn list [(R.a, S.b, pred)] into dict {(R.a, S.b) -> [pred]}
        deflated_joins = collections.defaultdict(list)
        for join_view in join_views:
            deflated_joins[(join_view.table, join_view.partner)].append(join_view)

        # step 2: turn dict {(R.a, S.b) -> [pred]} into multi-level dict {R.a -> {S.b -> [pred]}}
        aggregated_views = collections.defaultdict(lambda: collections.defaultdict(list))
        for (table, partner), view in deflated_joins.items():
            aggregated_views[table][partner].extend(view)

        # step 3: turn multi-level dict {R.a -> {S.b -> [pred]}} into dict w/ nested lists {R.a -> [[(S.b, pred)]]}
        merged_views = collections.defaultdict(list)
        for table, partner_data in aggregated_views.items():
            merged_views[table].extend(partner_data.values())

        # make sure to return an actual dict
        return dict(merged_views)

    def __init__(self, table_attribute: db.AttributeRef, partner_attribute: db.AttributeRef, *,
                 predicate: mosp.MospPredicate = None):
        self.table_attribute = table_attribute
        self.partner_attribute = partner_attribute

        self.table = table_attribute.table
        self.partner = partner_attribute.table

        # TODO: why do we need the predicate? We already know its an equality join between the two attributes!?
        if predicate and not predicate.is_join_predicate():
            raise ValueError("Not a join predicate: '{}'".format(predicate))
        self.predicate = predicate

    def __hash__(self) -> int:
        return hash((self.table_attribute, self.partner_attribute))

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, _EqualityJoinView):
            return False
        return self.table_attribute == other.table_attribute and self.partner_attribute == other.partner_attribute

    def __repr__(self) -> str:
        return str(self)

    def __str__(self) -> str:
        return f"{self.table_attribute} = {self.partner_attribute}"


class _JoinGraph:
    """The join graph provides a nice interface for querying information about the joins we have to execute.

    The graph is treated mutable in that tables will subsequently be marked as included in the join. Many methods
    operate on tables that are either already joined, or not yet joined and therefore provide different results
    depending on the current state of the query processing.
    """

    @staticmethod
    def build_for(query: mosp.MospQuery, *, dbs: db.DBSchema = db.DBSchema.get_instance()) -> "_JoinGraph":
        # For easy implementation of graph-theoretical functions, we represent the join graph as an actual graph
        # Nodes correspond to tables and edges correspond to joins between the tables.
        #
        # Each node will either be involved in at least one n:m join, or exclusively in Primary Key/Foreign Key
        # joins. As soon as a node (i.e. table) is inserted into the join tree, it does not need to be considered
        # further in any way.
        #
        # To keep track of whether a table has already been inserted, each node is annotated with
        # a `free` attribute (set to True for as long as the node has not been inserted). Therefore, at the beginning
        # of the UES algorithm, each node will be free. After each iteration, at least one more node will be dropped
        # from this list.
        #
        # Each edge will be annotated by a number of attributes:
        # 1) whether it describes a Primary Key/Foreign Key join
        # 2) In case it is a PK/FK join, what the Primary Key table and Foreign Key table are
        # 3) Which join predicate is used
        # Since a join may in principle involve multiple predicates (e.g. r.a = s.b AND r.c = s.d), each edge can
        # contain multiple predicates associated to it.

        graph = nx.Graph()
        graph.add_nodes_from(query.collect_tables(), free=True)

        predicate_map = collections.defaultdict(list)
        for join_predicate in [j for j in query.predicates() if j.is_join_predicate()]:
            predicate_map[tuple(sorted(join_predicate.tables()))].append(join_predicate)

        for join_predicate in predicate_map.values():
            left_tab = join_predicate[0].parse_left_attribute().table
            right_tab = join_predicate[0].parse_right_attribute().table

            # since there may be multiple join predicates for a single join (although we don't support this yet
            # on a MospPredicate level), we need to find the most specific one here
            join_types = [_is_pk_fk_join(jp, dbs=dbs) for jp in join_predicate]
            join_type = next((jt for jt in join_types if jt["pk_fk_join"]), join_types[0])

            pk, fk = (join_type["pk"], join_type["fk"]) if join_type["pk_fk_join"] else (None, None)
            graph.add_edge(left_tab, right_tab, pk_fk_join=join_type["pk_fk_join"], predicate=join_predicate,
                           primary_key=pk, foreign_key=fk)

        # mark each node as PK/FK node or n:m node
        for node in graph.nodes:
            neighbors = graph.adj[node]
            all_pk_fk_joins = all(join_data["pk_fk_join"] for join_data in neighbors.values())
            graph.nodes[node]["pk_fk_node"] = all_pk_fk_joins
            graph.nodes[node]["n_m_node"] = not all_pk_fk_joins

        return _JoinGraph(graph)

    def __init__(self, graph: nx.Graph):
        self.graph: nx.Graph = graph

    def join_components(self) -> Set["_JoinGraph"]:
        """
        A join component is a subset of all of the joins of a query, s.t. each table in the component is joined with
        at least one other table in the component.

        This implies that no join predicates exist that span multiple components. When executing the query, a cross
        product will have to be performed between the components.
        """
        return set(_JoinGraph(self.graph.subgraph(component).copy())
                   for component in nx.connected_components(self.graph))

    def free_n_m_joined_tables(self) -> Set[db.TableRef]:
        """Queries the join graph for all tables that are still free and part of at least one n:m join."""
        free_tables = [tab for tab, node_data in list(self.graph.nodes.data())
                       if node_data["free"] and node_data["n_m_node"]]
        return free_tables

    def free_pk_fk_joins_with(self, table: db.TableRef) -> Dict[db.TableRef, List[List[_EqualityJoinView]]]:
        """
        Determines all (yet free) tables S which are joined with the given table R, such that R joined S is a Primary
        Key/Foreign Key join.

        The provided join view is created for R and has the tables S as its partner.
        If R and S are joined via multiple predicates (such as `R.a = S.b AND R.c = S.d`), the join view will contain
        a list of predicates.
        """
        join_partners: List[Tuple[db.TableRef, List[mosp.MospPredicate]]] = [
            (partner, join["predicate"]) for partner, join in self.graph.adj[table].items()
            if join["pk_fk_join"] and self.is_free(partner)]
        partner_views = collections.defaultdict(list)
        for partner, joins in join_partners:
            partner_views[partner].append([_EqualityJoinView(predicate.attribute_of(table),
                                                             predicate.attribute_of(partner), predicate=predicate)
                                           for predicate in joins])
        return dict(partner_views)

    def free_pk_joins_with(self, table: db.TableRef) -> List[_EqualityJoinView]:
        """
        Determines all (yet free) tables S which are joined with the given table R, such that S joined R is a Primary
        Key/Foreign Key join with S acting as the Primary Key and (the given) table R acting as the Foreign Key.

        The resulting list contains one entry for each Primary Key/Foreign Key join.
        E.g., if there is a PK/FK join between R.a = S.b, free_pk_joins(R) will return a join view for R with S as its
        partner (provided that S is still a free table). If there are multiple PK/FK joins available on R, an entry
        is created for each one. The same holds if R and S are joined via multiple attributes
        (such as R.a = S.b AND R.c = S.d): Each such predicate will be returned in its own view.
        """
        join_partners: List[Tuple[db.TableRef, List[mosp.MospPredicate]]] = [
            (partner, join["predicate"]) for partner, join in self.graph.adj[table].items()
            if join["pk_fk_join"] and partner == join["primary_key"].table
            and self.is_free(partner) and self.is_pk_fk_table(partner)]
        return self._explode_compound_predicate_edges(table, join_partners)

    def free_n_m_join_partners(self, tables: Union[db.TableRef, List[db.TableRef]]) -> List[_EqualityJoinView]:
        """
        Determines all (yet free) tables S which are joined with any of the given tables R, such that R and S are
        joined by an n:m join.

        The resulting list contains one entry for each join, from the perspective of one of the given tables.
        If the join is carried out via multiple attributes (such as R.a = S.b AND R.c = S.d), each predicate will have
        its own join.
        """
        tables = util.enlist(tables)
        join_views = []
        for table in tables:
            join_partners: List[Tuple[db.TableRef, List[mosp.MospPredicate]]] = (
                [(partner, join["predicate"]) for partner, join in self.graph.adj[table].items()
                 if not join["pk_fk_join"] and self.is_free(partner)])
            join_views.extend(self._explode_compound_predicate_edges(table, join_partners))
        return join_views

    def free_fk_tables(self) -> List[_EqualityJoinView]:
        """Queries for all free tables that are exclusively joined in PK/FK joins and always act as the FK partner.

        The provided join views are from the perspective of the FK table and join to one of the available PK partners.
        """
        all_free_tables = [tab for tab, node_data in list(self.graph.nodes.data())
                           if node_data["free"] and node_data["pk_fk_node"]]
        free_fk_tables = []
        for candidate_table in all_free_tables:
            neighbors = self.graph.adj[candidate_table].items()
            if all(edge_data["fk"].table == candidate_table for __, edge_data in neighbors):
                __, join_data = next(iter(neighbors))
                free_fk_tables.append(_EqualityJoinView(join_data["fk"], join_data["pk"],
                                                        predicate=join_data["predicate"]))
        return free_fk_tables

    def is_free(self, table: db.TableRef) -> bool:
        """Checks, whether the given table is not inserted into the join tree, yet."""
        return self.graph.nodes[table]["free"]

    def is_pk_fk_table(self, table: db.TableRef) -> bool:
        """Checks, whether the given table is exclusively joined via PK/FK joins."""
        return self.graph.nodes[table]["pk_fk_node"]

    def is_free_fk_table(self, table: db.TableRef) -> bool:
        """
        Checks, whether the given table only takes part in PK/FK joins and acts as a FK partner in at least one of
        them.
        """
        return self.is_free(table) and any(True for join_data in self.graph.adj[table].values()
                                           if join_data["foreign_key"].table == table)

    def available_join_paths(self, table: db.TableRef) -> Dict[db.TableRef, List[mosp.MospPredicate]]:
        """Searches for all tables that are already joined and have a valid join predicate with the given table."""
        if not self.is_free(table):
            raise ValueError("Join paths for already joined table are undefined")
        join_partners = {partner: util.simplify(join["predicate"]) for partner, join in self.graph.adj[table].items()
                         if not self.is_free(partner)}
        return join_partners

    def used_join_paths(self, table: db.TableRef) -> Dict[db.TableRef, List[mosp.MospPredicate]]:
        if self.is_free(table):
            raise ValueError("Cannot search for used join paths for a free table")
        join_partners = {partner: util.simplify(join["predicate"]) for partner, join in self.graph.adj[table].items()
                         if not self.is_free(partner)}
        return join_partners

    def contains_free_n_m_tables(self) -> bool:
        """Checks, whether at least one free table remains in the graph."""
        # Since we need to access multiple attributes on the node, we cannot pull them from the data-dictionary
        # directly. Instead we have to operate on the entire dict.
        return any(node_data["free"] for (__, node_data) in list(self.graph.nodes.data()) if node_data["n_m_node"])

    def contains_free_pk_fk_tables(self) -> bool:
        return any(node_data["free"] for (__, node_data) in list(self.graph.nodes.data()) if node_data["pk_fk_node"])

    def contains_free_tables(self) -> bool:
        return any(free for (__, free) in self.graph.nodes.data("free"))

    def mark_joined(self, table: db.TableRef, *, n_m_join: bool = False, trace: bool = False):
        """Annotates the given table as joined in the graph.

        Setting `n_m_join` to `True` will trigger an invalidation pass setting all free joins to Foreign Key tables (
        i.e. the Primary Key table is already joined) to n:m joins. This is necessary b/c the application of an n:m
        join most likely duplicated the primary keys and thus invalidated the uniqueness property of the primary key.
        """
        self.graph.nodes[table]["free"] = False

        if n_m_join:
            self._invalidate_pk_joins(trace=trace)

    def count_free_joins(self, table: db.TableRef) -> int:
        return len([partner for partner, __ in self.graph.adj[table].items() if self.is_free(partner)])

    def count_selected_joins(self) -> int:
        return len([node for node, free in self.graph.nodes.data("free") if not free])

    def print(self, title: str = "", *, annotate_fk: bool = True, node_size: int = 1500, layout: str = "shell"):
        """Writes the current join graph structure to a matplotlib device."""
        node_labels = {node: node.alias for node in self.graph.nodes}
        edge_sizes = [3.0 if pk_fk_join else 1.0 for (__, __, pk_fk_join) in self.graph.edges.data("pk_fk_join")]
        node_edge_color = ["black" if free else "red" for (__, free) in self.graph.nodes.data("free")]
        edge_data = dict(((source, target), data) for source, target, data in self.graph.edges.data())
        edge_labels = {edge: f'[{data["foreign_key"].table.alias}]' if data["pk_fk_join"] else ""
                       for edge, data in edge_data.items()} if annotate_fk else {edge: "" for edge in edge_data}

        layouts = {"shell": nx.shell_layout, "planar": nx.planar_layout, "circular": nx.circular_layout,
                   "spring": nx.spring_layout, "spiral": nx.spiral_layout}
        pos = layouts.get(layout, nx.shell_layout)(self.graph)
        nx.draw_networkx_nodes(self.graph, pos, node_size=node_size, node_color="white", edgecolors=node_edge_color,
                               linewidths=1.8)
        nx.draw_networkx_labels(self.graph, pos, node_labels)
        nx.draw_networkx_edges(self.graph, pos, edge_color="black", width=edge_sizes)
        nx.draw_networkx_edge_labels(self.graph, pos, edge_labels)

        ax = plt.gca()
        ax.margins(0.08)
        ax.set_title(title)
        plt.axis("off")
        plt.tight_layout()

    def _explode_compound_predicate_edges(self, table, join_edges):
        join_views = []
        for partner, compound_join in join_edges:
            for join_predicate in compound_join:
                table_attribute = join_predicate.attribute_of(table)
                partner_attribute = join_predicate.attribute_of(partner)
                join_views.append(_EqualityJoinView(table_attribute, partner_attribute, predicate=join_predicate))
        return join_views

    def _free_nodes(self) -> List[db.TableRef]:
        """Provides all nodes that are currently free."""
        return [node for node, free in self.graph.nodes.data("free") if free]

    def _current_shell(self) -> Set[db.TableRef]:
        """Queries the graph for all nodes that are not free, but connected to at least one free node."""
        shell = set()
        for tab1, tab2 in self.graph.edges:
            if self.is_free(tab1) and not self.is_free(tab2):
                shell.add(tab2)
            elif self.is_free(tab2) and not self.is_free(tab1):
                shell.add(tab1)
        return shell

    def _invalidate_pk_joins(self, *, trace: bool = False):
        # TODO: extensive documentation
        trace_logger = util.make_logger(trace)
        shell_fk_neighbors = []
        for shell_node in self._current_shell():
            shell_fk_neighbors += [(shell_node, neighbor) for neighbor, join_data in self.graph.adj[shell_node].items()
                                   if join_data["pk_fk_join"]
                                   and join_data["primary_key"].table == shell_node
                                   and self.is_free(neighbor)]

        for shell_node, fk_neighbor in shell_fk_neighbors:
            # the node data might actually be updated multiple times for some nodes, but that is okay
            node_data = self.graph.nodes[fk_neighbor]
            node_data["pk_fk_node"] = False
            node_data["n_m_node"] = True
            trace_logger(".. Marking FK neighbor", fk_neighbor, "as n:m joined")

            edge_data = self.graph.edges[(shell_node, fk_neighbor)]
            edge_data["pk_fk_join"] = False
            edge_data["primary_key"] = None
            edge_data["foreign_key"] = None


class JoinTree:
    """The join tree contains a semi-ordered sequence of joins.

    The ordering results from the different levels of the tree, but joins within the same level and parent are
    unordered.
    """

    @staticmethod
    def empty_join_tree() -> "JoinTree":
        return JoinTree()

    @staticmethod
    def load_from_query(query: mosp.MospQuery) -> "JoinTree":
        # TODO: include join predicates here
        join_tree = JoinTree.empty_join_tree()
        join_tree = join_tree.with_base_table(query.base_table())
        for join in query.joins():
            if join.is_subquery():
                subquery_join = JoinTree.load_from_query(join.subquery)
                join_tree = join_tree.joined_with_subquery(subquery_join)
            else:
                join_tree = join_tree.joined_with_base_table(join.base_table())
        return join_tree

    def __init__(self, *, predicate: mosp.MospPredicate = None):
        self.left: Union["JoinTree", db.TableRef] = None
        self.right: Union["JoinTree", db.TableRef] = None
        self.predicate: mosp.MospPredicate = predicate

    def is_empty(self) -> bool:
        return self.right is None

    def all_tables(self) -> List[db.TableRef]:
        left_tables = []
        if self.left_is_base_table():
            left_tables = [self.left]
        elif self.left_is_subquery():
            left_tables = self.left.all_tables()

        right_tables = []
        if self.right_is_base_table():
            right_tables = [self.right]
        elif self.right is not None:
            right_tables = self.right.all_tables()

        return util.flatten([left_tables, right_tables])

    def all_attributes(self) -> Set[db.AttributeRef]:
        right_attributes = self.right.all_attributes() if self.right_is_tree() else set()
        left_attributes = self.left.all_attributes() if self.left_is_subquery() else set()
        own_attributes = set(self.predicate.parse_attributes()) if self.predicate else set()
        return right_attributes | left_attributes | own_attributes

    def left_is_base_table(self) -> bool:
        return isinstance(self.left, db.TableRef)

    def right_is_base_table(self) -> bool:
        return isinstance(self.right, db.TableRef)

    def right_is_tree(self) -> bool:
        return isinstance(self.right, JoinTree)

    def left_is_subquery(self) -> bool:
        return isinstance(self.left, JoinTree)

    def with_base_table(self, table: db.TableRef) -> "JoinTree":
        self.right = table
        return self

    def joined_with_base_table(self, table: db.TableRef, *, predicate: mosp.MospPredicate = None) -> "JoinTree":
        if not self.left:
            self.left = table
            self.predicate = predicate
            return self

        new_root = JoinTree(predicate=predicate)
        new_root.left = table
        new_root.right = self
        return new_root

    def joined_with_subquery(self, subquery: "JoinTree", *, predicate: mosp.MospPredicate = None) -> "JoinTree":
        if not self.left:
            self.left = subquery
            self.predicate = predicate
            return self

        new_root = JoinTree(predicate=predicate)
        new_root.left = subquery
        new_root.right = self
        return new_root

    def traverse_right_deep(self) -> dict:
        if self.right_is_base_table():
            yield_right = [{"subquery": False, "table": self.right}]
        else:
            yield_right = self.right.traverse_right_deep()

        if self.left_is_base_table():
            yield_left = [{"subquery": False, "table": self.left, "predicate": self.predicate}]
        else:
            yield_left = [{"subquery": True, "children": self.left.traverse_right_deep(), "predicate": self.predicate}]

        return yield_right + yield_left

    def pretty_print(self, *, _indentation=0, _inner=False):
        indent_str = (" " * _indentation) + "<- "

        if self.left_is_base_table():
            left_str = indent_str + str(self.left)
        else:
            left_str = indent_str + self.left.pretty_print(_indentation=_indentation+2, _inner=True)

        if self.right_is_base_table():
            right_str = indent_str + str(self.right)
        else:
            right_str = self.right.pretty_print(_indentation=_indentation+2, _inner=True)

        if _inner:
            return "\n".join([left_str, right_str])
        else:
            print(left_str, right_str, sep="\n")

    def __hash__(self) -> int:
        return hash((self.left, self.right))

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, JoinTree):
            return False
        return self.left == other.left and self.right == other.right

    def __repr__(self) -> str:
        return str(self)

    def __str__(self) -> str:
        if not self.left:
            left_label = "[NONE]"
        elif self.left_is_base_table():
            left_label = self.left.full_name
        else:
            left_label = f"({str(self.left)})"

        if not self.right:
            right_label = "[NONE]"
        elif self.right_is_base_table():
            right_label = self.right.full_name
        else:
            right_label = str(self.right)

        join_str = f"{right_label} ⋈ {left_label}"  # let's read joins from left to right!
        return join_str


class _BaseAttributeFrequenciesLoader:
    def __init__(self, base_estimates: Dict[db.TableRef, int], dbs: db.DBSchema = db.DBSchema.get_instance()):
        self.dbs = dbs
        self.base_estimates = base_estimates
        self.attribute_frequencies = {}

    def __getitem__(self, key: db.AttributeRef) -> int:
        if key not in self.attribute_frequencies:
            top1 = self.dbs.load_most_common_values(key, k=1)
            if not top1:
                top1 = self.base_estimates[key.table]
            else:
                __, top1 = top1[0]
                top1 = round(top1 * self.dbs.load_tuple_count(key.table))
            top1 = min(top1, self.base_estimates[key.table])
            self.attribute_frequencies[key] = top1
        return self.attribute_frequencies[key]

    def __repr__(self) -> str:
        return str(self)

    def __str__(self) -> str:
        return str(self.attribute_frequencies)


class _JoinAttributeFrequenciesLoader:
    def __init__(self, base_frequencies: _BaseAttributeFrequenciesLoader):
        self.base_frequencies = base_frequencies
        self.attribute_frequencies = {}
        self.current_multiplier = 1

    def __getitem__(self, key: db.AttributeRef) -> int:
        if key not in self.attribute_frequencies:
            base_frequency = self.base_frequencies[key]
            self.attribute_frequencies[key] = base_frequency * self.current_multiplier
        return self.attribute_frequencies[key]

    def __setitem__(self, key: db.AttributeRef, value: int):
        self.attribute_frequencies[key] = value

    def __repr__(self) -> str:
        return str(self)

    def __str__(self) -> str:
        return str(self.attribute_frequencies)


class _TableBoundStatistics:
    def __init__(self, query: mosp.MospQuery,
                 predicate_map: Dict[db.TableRef, Union[mosp.MospPredicate, mosp.CompoundMospFilterPredicate]], *,
                 base_cardinality_estimator: BaseCardinalityEstimator,
                 dbs: db.DBSchema = db.DBSchema.get_instance()):
        self.query = query
        self.base_estimates = _estimate_filtered_cardinalities(predicate_map, base_cardinality_estimator,
                                                               dbs=dbs)
        self.base_frequencies = _BaseAttributeFrequenciesLoader(self.base_estimates)
        self.joined_frequencies = _JoinAttributeFrequenciesLoader(self.base_frequencies)
        self.upper_bounds = {}
        self.dbs = dbs

    def update_frequencies(self, joined_table: db.TableRef,
                           join_predicate: Union[mosp.MospPredicate, List[mosp.MospPredicate]], *,
                           join_tree: JoinTree):
        # FIXME: the current implementation only works for single join predicates
        # (i.e. JOIN R.a = S.b AND R.c = S.d is not supported!)
        # However, our formulas can be easily expanded to work with multiple predicates by choosing the smaller
        # bound. This should be fixed soon-ish.
        if util.contains_multiple(join_predicate):
            raise ValueError("Compound join predicates are not yet supported.")
        join_predicate: mosp.MospPredicate = util.simplify(join_predicate)

        new_join_attribute = join_predicate.attribute_of(joined_table)
        existing_partner_attribute = join_predicate.join_partner(joined_table)
        new_attribute_frequency = self.base_frequencies[new_join_attribute]

        for existing_attribute in [attr for attr in join_tree.all_attributes() if attr != new_join_attribute]:
            self.joined_frequencies[existing_attribute] *= new_attribute_frequency

        self.joined_frequencies[new_join_attribute] = self.joined_frequencies[existing_partner_attribute]
        self.joined_frequencies.current_multiplier *= new_attribute_frequency

    def base_bounds(self) -> Dict[db.TableRef, int]:
        return {tab: bound for tab, bound in self.upper_bounds.items() if isinstance(tab, db.TableRef)}

    def __repr__(self) -> str:
        return str(self)

    def __str__(self) -> str:
        return str(self.upper_bounds)


class SubqueryGenerationStrategy(abc.ABC):
    """
    A subquery generator is capable of both deciding whether a certain join should be implemented as a subquery, as
    well as rolling out the transformation itself.
    """

    @abc.abstractmethod
    def execute_as_subquery(self, candidate: db.TableRef, join_graph: _JoinGraph, join_tree: JoinTree, *,
                            stats: _TableBoundStatistics) -> bool:
        return NotImplemented


class DefensiveSubqueryGeneration(SubqueryGenerationStrategy):
    def execute_as_subquery(self, candidate: db.TableRef, join_graph: _JoinGraph, join_tree: JoinTree, *,
                            stats: _TableBoundStatistics) -> bool:
        return (stats.upper_bounds[candidate] < stats.base_estimates[candidate]
                and join_graph.count_selected_joins() > 2)


class GreedySubqueryGeneration(SubqueryGenerationStrategy):
    def execute_as_subquery(self, candidate: db.TableRef, join_graph: _JoinGraph, join_tree: JoinTree, *,
                            stats: _TableBoundStatistics) -> bool:
        return join_graph.count_selected_joins() > 2


class NoSubqueryGeneration(SubqueryGenerationStrategy):
    def execute_as_subquery(self, candidate: db.TableRef, join_graph: _JoinGraph, join_tree: JoinTree, *,
                            stats: _TableBoundStatistics) -> bool:
        return False


def _build_predicate_map(query: mosp.MospQuery
                         ) -> Dict[db.TableRef, Union[mosp.MospPredicate, mosp.CompoundMospFilterPredicate]]:
    """The predicate map is a dictionary which maps each table to the filter predicates that apply to this table."""
    all_filter_predicates = [pred for pred in query.predicates() if not pred.is_join_predicate()]
    raw_predicate_map = collections.defaultdict(list)

    for filter_pred in all_filter_predicates:
        if filter_pred.is_compound():
            for pred in filter_pred.collect_left_attributes():
                raw_predicate_map[pred.table].append(filter_pred)
        else:
            raw_predicate_map[filter_pred.parse_left_attribute().table].append(filter_pred)

    aggregated_predicate_map = {table: mosp.CompoundMospFilterPredicate.build_and_predicate(predicate)
                                for table, predicate in raw_predicate_map.items()}

    for tab in query.collect_tables():
        if tab not in aggregated_predicate_map:
            aggregated_predicate_map[tab] = []

    return aggregated_predicate_map


def _build_join_map(query: mosp.MospQuery) -> Dict[db.TableRef, Dict[db.TableRef, Set[mosp.MospPredicate]]]:
    """The join map is a dictionary which maps each table to the join predicates that apply to this table."""
    all_join_predicates = [pred for pred in query.predicates() if pred.is_join_predicate()]
    predicate_map = collections.defaultdict(lambda: collections.defaultdict(set))

    for join_pred in all_join_predicates:
        left_attribute, right_attribute = join_pred.parse_attributes()
        predicate_map[left_attribute.table][right_attribute.table].add(join_pred)
        predicate_map[right_attribute.table][left_attribute.table].add(join_pred)

    # castings
    predicate_map.default_factory = lambda: collections.defaultdict(list)
    for table, partners in predicate_map.items():
        predicate_map[table].default_factory = list
        for partner in partners:
            predicate_map[table][partner] = list(predicate_map[table][partner])
    return predicate_map


def _estimate_filtered_cardinalities(predicate_map: dict, estimator: BaseCardinalityEstimator, *,
                                     dbs: db.DBSchema = db.DBSchema.get_instance()) -> Dict[db.TableRef, int]:
    """Fetches the PG estimates for all tables in the predicate_map according to their associated filters."""
    cardinality_dict = {}
    for table, predicate in predicate_map.items():
        cardinality_dict[table] = estimator.estimate_rows(predicate) if predicate else estimator.all_tuples(table)
    return cardinality_dict


def _absorb_pk_fk_hull_of(table: db.TableRef, *, join_graph: _JoinGraph, join_tree: JoinTree,
                          subquery_generator: SubqueryGenerationStrategy,
                          base_table_estimates: Dict[db.TableRef, int],
                          pk_only: bool = False, verbose: bool = False, trace: bool = False) -> JoinTree:
    # TODO: the choice of estimates and the iteration itself are actually not optimal. We do not consider filters at
    # all!
    # A better strategy would be: always merge PKs in first (since they can only reduce the intermediate size)
    # Thereby we would effectively treat FK tables as n:m tables! Why did we make that distinction in the first place?

    logger = util.make_logger(verbose)

    if pk_only:
        join_paths = collections.defaultdict(list)
        for join in join_graph.free_pk_joins_with(table):
            if util.contains_multiple(join):
                partner = join[0].predicate.join_partner(table).table
            else:
                partner = join.predicate.join_partner(table).table
            join_paths[partner].append(join)
        join_paths = dict(join_paths)
    else:
        join_paths: dict = join_graph.free_pk_fk_joins_with(table)
    candidate_estimates: dict = {join: base_table_estimates[join]
                                 for join in join_paths}

    pk_fk_join_sequence = []
    while candidate_estimates:
        # always insert the table with minimum cardinality next
        next_pk_fk_join = util.argmin(candidate_estimates)
        pk_fk_join_sequence.append(next_pk_fk_join)
        join_graph.mark_joined(next_pk_fk_join)

        logger(".. Also including PK/FK join from hull:", next_pk_fk_join)

        # after inserting the join into our join tree, new join paths may become available
        if pk_only:
            fresh_joins = _EqualityJoinView.aggregate_view_list(join_graph.free_pk_joins_with(next_pk_fk_join))
        else:
            fresh_joins = join_graph.free_pk_fk_joins_with(next_pk_fk_join)
        join_paths = util.dict_merge(join_paths, fresh_joins,
                                     update=lambda __, existing_paths, new_paths: existing_paths + new_paths)
        candidate_estimates = util.dict_merge(candidate_estimates,
                                              {join: base_table_estimates[join] for join in fresh_joins})

        candidate_estimates.pop(next_pk_fk_join)
        # TODO: if inserting a FK join, update statistics here

    # TODO: check if executed as subquery and insert into join tree
    for join in pk_fk_join_sequence:
        # TODO: for now we just always use the first predicate available. Is this sufficient or does the choice of
        # predicate matter?
        join_tree = join_tree.joined_with_base_table(join, predicate=util.simplify(join_paths[join][0]).predicate)
    return join_tree


JoinOrderOptimizationResult = collections.namedtuple("JoinOrderOptimizationResult",
                                                     ["final_order", "intermediate_bounds", "final_bound", "regular"])


def _calculate_join_order(query: mosp.MospQuery, *,
                          predicate_map: Dict[db.TableRef, Union[mosp.MospPredicate,
                                                                 mosp.CompoundMospFilterPredicate]],
                          join_estimator: JoinCardinalityEstimator = None,
                          base_estimator: BaseCardinalityEstimator = PostgresCardinalityEstimator(),
                          subquery_generator: SubqueryGenerationStrategy = DefensiveSubqueryGeneration(),
                          visualize: bool = False, visualize_args: dict = None,
                          verbose: bool = False, trace: bool = False,
                          dbs: db.DBSchema = db.DBSchema.get_instance()
                          ) -> Union[JoinOrderOptimizationResult, List[JoinOrderOptimizationResult]]:
    join_estimator = join_estimator if join_estimator else DefaultUESCardinalityEstimator(query)
    join_graph = _JoinGraph.build_for(query)

    # In principle it could be that our query involves a cross-product between some of its relations. If that is the
    # case, we cannot simply build a single join tree b/c a tree cannot capture the semantics of cross-product of
    # multiple independent join trees very well. Therefore, we are going to return either a single join tree (which
    # should be the case for most queries), or a list of join trees (one for each closed/connected part of the join
    # graph).
    partitioned_join_trees = [_calculate_join_order_for_join_partition(query, partition, predicate_map=predicate_map,
                                                                       join_cardinality_estimator=join_estimator,
                                                                       base_cardinality_estimator=base_estimator,
                                                                       subquery_generator=subquery_generator,
                                                                       verbose=verbose, trace=trace,
                                                                       visualize=visualize,
                                                                       visualize_args=visualize_args,
                                                                       dbs=dbs)
                              for partition in join_graph.join_components()]
    return util.simplify(partitioned_join_trees)


def _calculate_join_order_for_join_partition(query: mosp.MospQuery, join_graph: _JoinGraph, *,
                                             predicate_map: Dict[db.TableRef, Union[mosp.MospPredicate,
                                                                                    mosp.CompoundMospFilterPredicate]],
                                             join_cardinality_estimator: JoinCardinalityEstimator,
                                             base_cardinality_estimator: BaseCardinalityEstimator,
                                             subquery_generator: SubqueryGenerationStrategy,
                                             visualize: bool = False, visualize_args: dict = None,
                                             verbose: bool = False, trace: bool = False,
                                             dbs: db.DBSchema = db.DBSchema.get_instance()
                                             ) -> JoinOrderOptimizationResult:
    trace_logger = util.make_logger(trace)
    logger = util.make_logger(verbose or trace)

    join_tree = JoinTree.empty_join_tree()
    stats = _TableBoundStatistics(query, predicate_map, base_cardinality_estimator=base_cardinality_estimator, dbs=dbs)

    if not join_graph.free_n_m_joined_tables():
        # TODO: documentation
        first_fk_table = util.argmin({table: estimate for table, estimate in stats.base_estimates.items()
                                      if join_graph.is_free_fk_table(table)})
        join_tree = join_tree.with_base_table(first_fk_table)
        join_graph.mark_joined(first_fk_table)
        join_tree = _absorb_pk_fk_hull_of(first_fk_table, join_graph=join_graph, join_tree=join_tree,
                                          subquery_generator=NoSubqueryGeneration(),
                                          base_table_estimates=stats.base_estimates)
        assert not join_graph.contains_free_tables()
        return JoinOrderOptimizationResult(join_tree, None, None, False)

    # This has nothing to do with the actual algorithm is merely some technical code for visualizations
    if visualize:
        visualize_args = {} if not visualize_args else visualize_args
        n_iterations = len(join_graph.free_n_m_joined_tables())
        current_iteration = 0
        figsize = visualize_args.pop("figsize", (8, 5))  # magic numbers which normally produce well-sized plots
        width, height = figsize
        plt.figure(figsize=(width, height * n_iterations))

    # The UES algorithm will iteratively expand the join order one join at a time. In each iteration, the best n:m
    # join (after/before being potentially filtered via PK/FK joins) is selected. In the join graph, we keep track of
    # all the tables that we still have to join.
    while join_graph.contains_free_n_m_tables():

        # First up, we have to update the upper bounds for all the relation that have not yet been joined
        # This step is necessary in each iteration, b/c the attribute frequencies are updated each time a new join
        # was performed.

        # We keep track of the lowest bounds to short-circuit intialization of the join tree with the very first
        # n:m table
        lowest_min_bound = np.inf
        lowest_bound_table = None
        for candidate_table in join_graph.free_n_m_joined_tables():
            # The minimum bound is the lower value of either the base table cardinality after applying all filter
            # predicates, or the cardinality after applying a PK/FK join where our candidate table acted as the
            # Foreign Key (and was therefore subject to filtering by the Primary Key)

            # Let's start by getting the filtered estimate and calculate the PK/FK join bound next
            filter_estimate = stats.base_estimates[candidate_table]

            # Now, let's look at all remaining PK/FK joins with the candidate table. We can get these easily via
            # join_graph.free_pk_joins.
            pk_joins = [(join_view.partner, join_view.table_attribute)
                        for join_view in join_graph.free_pk_joins_with(candidate_table)]
            pk_fk_bounds = [stats.base_frequencies[attribute] * stats.base_estimates[partner]
                            for (partner, attribute) in pk_joins]  # Formula: MF(candidate, fk_attr) * |pk_table|
            candidate_min_bound = min([filter_estimate] + pk_fk_bounds)  # this is concatenation, not addition!

            trace_logger(".. Bounds for candidate", candidate_table, ":: Filter:", filter_estimate,
                         "| PKs:", pk_fk_bounds)

            stats.upper_bounds[candidate_table] = candidate_min_bound

            # If the bound we just calculated is less than our current best bound, we found an improved candidate
            if candidate_min_bound < lowest_min_bound:
                lowest_min_bound = candidate_min_bound
                lowest_bound_table = candidate_table

        # If we are choosing the base table, just insert it right away and continue with the next table
        if join_tree.is_empty():
            join_tree = join_tree.with_base_table(lowest_bound_table)
            join_graph.mark_joined(lowest_bound_table, trace=trace)

            # FIXME: should this also include all PK/FK joins on the base table?!
            pk_joins = sorted(join_graph.free_pk_joins_with(lowest_bound_table),
                              key=lambda fk_join_view: stats.base_estimates[fk_join_view.partner])
            logger("Selected first table:", lowest_bound_table, "with PK/FK joins",
                   [pk_table.partner for pk_table in pk_joins])

            for pk_table, join_predicate in [(pk_join.partner, pk_join.predicate) for pk_join in pk_joins]:
                trace_logger(".. Adding PK join with", pk_table, "on", join_predicate)
                join_tree = join_tree.joined_with_base_table(pk_table, predicate=join_predicate)
                join_graph.mark_joined(pk_table)
                join_tree = _absorb_pk_fk_hull_of(pk_table, join_graph=join_graph, join_tree=join_tree,
                                                  subquery_generator=NoSubqueryGeneration(),
                                                  base_table_estimates=stats.base_estimates,
                                                  pk_only=True, verbose=verbose, trace=trace)

            stats.upper_bounds[join_tree] = lowest_min_bound

            # This has nothing to do with the actual algorithm and is merely some technical code for visualization
            if visualize:
                current_iteration += 1
                plt.subplot(n_iterations, 1, current_iteration)
                join_graph.print(title=f"Join graph after base table selection, selected table: {lowest_bound_table}",
                                 **visualize_args)
            trace_logger(".. Base estimates:", stats.base_estimates)
            trace_logger("")
            continue

        trace_logger(".. Current frequencies:", stats.joined_frequencies)

        # Now that the bounds are up-to-date for each relation, we can select the next table to join based on the
        # number of outgoing tuples after including the relation in our current join tree.
        lowest_min_bound = np.inf
        selected_candidate = None
        candidate_bounds = {}
        for candidate_view in join_graph.free_n_m_join_partners(join_tree.all_tables()):
            # We examine the join between the candidate table and a table that is already part of the join tree.
            # To do so, we need to figure out on which attributes the join takes place. There is one attribute
            # belonging to the existing join tree (the tree_attribute) and one attribute for the candidate table (the
            # candidate_attribute). Likewise, we need to find out how many distinct values are in each relation.
            # The final formula is
            # min(upper(T) / MF(T, a), upper(C) / MF(C, b)) * MF(T, a) * MF(C, b)
            # for join tree T with join attribute a and candidate table C with join attribute b.
            # TODO: this should be subject to the join cardinality estimation strategy, not hard-coded.
            candidate_table, candidate_attribute = candidate_view.partner, candidate_view.partner_attribute
            tree_attribute = candidate_view.table_attribute
            n_values_tree = (stats.upper_bounds[join_tree]
                             /
                             stats.joined_frequencies[tree_attribute])
            n_values_candidate = (stats.upper_bounds[candidate_table]
                                  /
                                  stats.base_frequencies[candidate_attribute])
            candidate_bound = (min(n_values_tree, n_values_candidate)
                               * stats.joined_frequencies[tree_attribute]
                               * stats.base_frequencies[candidate_attribute])

            trace_logger(".. Checking candidate", candidate_view, "with bound", candidate_bound)

            if candidate_bound < lowest_min_bound:
                lowest_min_bound = candidate_bound
                selected_candidate = candidate_table

            if candidate_table in candidate_bounds:
                curr_bound = candidate_bounds[candidate_table]
                if curr_bound > candidate_bound:
                    candidate_bounds[candidate_table] = candidate_bound
            else:
                candidate_bounds[candidate_table] = candidate_bound

        trace_logger(".. Base frequencies:", stats.base_frequencies)
        trace_logger(".. Current bounds:", candidate_bounds)

        # We have now selected the next n:m join to execute. But before we can actually include this join in our join
        # tree, we have to figure out one last thing: what to with the Primary Key/Foreign Key joins on the selected
        # table. There are two general strategies here: most naturally, they may simply be executed after the candidate
        # table has been joined. However, a more efficient approach could be to utilize the Primary Key tables as
        # filters on the selected candidate to reduce its cardinality before the join and therefore minimize the size
        # of intermediates. This idea corresponds to an execution of the candidate join as a subquery.
        # Which idea is the right one in the current situation is not decided by the algorithm itself. Instead, the
        # decision is left to a policy (the subquery_generator), which decides the appropriate action.

        # TODO: for now we just always use the first predicate available. Is this sufficient or does the choice of
        # predicate matter?
        join_graph.mark_joined(selected_candidate, n_m_join=True, trace=trace)
        join_predicate = util.dict_value(join_graph.used_join_paths(selected_candidate), pull_any=True)
        pk_joins = sorted(join_graph.free_pk_joins_with(selected_candidate),
                          key=lambda fk_join_view: stats.base_estimates[fk_join_view.partner])
        logger("Selected next table:", selected_candidate, "with PK/FK joins",
               [pk_table.partner for pk_table in pk_joins], "on predicate", join_predicate)

        if pk_joins and subquery_generator.execute_as_subquery(selected_candidate, join_graph, join_tree,
                                                               stats=stats):
            subquery_join = JoinTree().with_base_table(selected_candidate)
            for pk_join in pk_joins:
                trace_logger(".. Adding PK join with", pk_join.partner, "on", pk_join.predicate)
                subquery_join = subquery_join.joined_with_base_table(pk_join.partner, predicate=pk_join.predicate)
                join_graph.mark_joined(pk_join.partner)
                subquery_join = _absorb_pk_fk_hull_of(pk_join.partner, join_graph=join_graph, join_tree=subquery_join,
                                                      subquery_generator=NoSubqueryGeneration(),
                                                      base_table_estimates=stats.base_estimates,
                                                      pk_only=True)
            join_tree = join_tree.joined_with_subquery(subquery_join, predicate=join_predicate)
            logger(".. Creating subquery for PK joins", subquery_join)
        else:
            join_tree = join_tree.joined_with_base_table(selected_candidate, predicate=join_predicate)
            for pk_join in pk_joins:
                trace_logger(".. Adding PK join with", pk_join.partner, "on", pk_join.predicate)
                join_tree = join_tree.joined_with_base_table(pk_join.partner, predicate=pk_join.predicate)
                join_graph.mark_joined(pk_join.partner)
                join_tree = _absorb_pk_fk_hull_of(pk_join.partner, join_graph=join_graph, join_tree=join_tree,
                                                  subquery_generator=NoSubqueryGeneration(),
                                                  base_table_estimates=stats.base_estimates,
                                                  pk_only=True)

        # Update our statistics based on the join(s) we just executed.
        stats.upper_bounds[join_tree] = lowest_min_bound
        stats.update_frequencies(selected_candidate, join_predicate, join_tree=join_tree)

        # This has nothing to do with the actual algorithm and is merely some technical code for visualization
        if visualize:
            current_iteration += 1
            plt.subplot(n_iterations, 1, current_iteration)
            join_graph.print(title=f"Join graph after iteration {current_iteration}, "
                             f"selected table: {selected_candidate}", **visualize_args)

        trace_logger("")

    assert not join_graph.contains_free_tables()
    trace_logger("Final join ordering:", join_tree)

    return JoinOrderOptimizationResult(join_tree, stats.upper_bounds, stats.upper_bounds[join_tree], True)


def _determine_referenced_attributes(join_sequence: List[dict]) -> Dict[db.TableRef, Set[db.AttributeRef]]:
    referenced_attributes = collections.defaultdict(set)
    for join in join_sequence:
        if join["subquery"]:
            subquery_referenced_attributes = _determine_referenced_attributes(join["children"])
            for table, attributes in subquery_referenced_attributes.items():
                referenced_attributes[table] |= attributes
            for left, right in [predicate.parse_attributes() for predicate in util.enlist(join["predicate"])]:
                referenced_attributes[left.table].add(left)
                referenced_attributes[right.table].add(right)
        elif "predicate" in join or join["subquery"]:
            for left, right in [predicate.parse_attributes() for predicate in util.enlist(join["predicate"])]:
                referenced_attributes[left.table].add(left)
                referenced_attributes[right.table].add(right)
        else:
            continue
    referenced_attributes = dict(referenced_attributes)
    return referenced_attributes


def _collect_tables(join_sequence: List[dict]) -> db.TableRef:
    tables = set()
    for join in join_sequence:
        if join["subquery"]:
            tables |= _collect_tables(join["children"])
        else:
            tables.add(join["table"])
    return tables


def _rename_predicate_if_necessary(predicate: Union[mosp.MospPredicate, mosp.CompoundMospFilterPredicate],
                                   table_renamings: Dict[db.TableRef, db.TableRef]
                                   ) -> Union[mosp.MospPredicate, mosp.CompoundMospFilterPredicate]:
    for table in util.enlist(predicate.parse_tables(), strict=False):
        if table in table_renamings:
            predicate = predicate.rename_table(from_table=table, to_table=table_renamings[table],
                                               prefix_attribute=True)
    return predicate


def _generate_mosp_data_for_sequence(join_sequence: List[dict], *,
                                     predicate_map: Dict[db.TableRef,
                                                         Union[mosp.MospPredicate, mosp.CompoundMospFilterPredicate]],
                                     join_predicates: Dict[db.TableRef,
                                                           Dict[db.TableRef, List[mosp.MospPredicate]]],
                                     referenced_attributes: Dict[db.TableRef, Set[db.AttributeRef]] = None,
                                     table_renamings: Dict[db.TableRef, db.TableRef] = None,
                                     joined_tables: Set[db.TableRef] = None,
                                     in_subquery: bool = False):

    # TODO: lots and lots of documentation

    if not referenced_attributes:
        referenced_attributes = _determine_referenced_attributes(join_sequence)
    if not table_renamings:
        table_renamings = {}
    if joined_tables is None:  # we might also encouter empty lists but want to leave them untouched
        joined_tables = set()

    base_table, *joins = join_sequence
    base_table = base_table["table"]
    joined_tables.add(base_table)
    from_list = [mosp.tableref_to_mosp(base_table)]

    for join_idx, join in enumerate(joins):
        applicable_join_predicates: List[mosp.MospPredicate] = []
        if join["subquery"]:
            subquery_mosp = _generate_mosp_data_for_sequence(join["children"],
                                                             predicate_map=predicate_map,
                                                             referenced_attributes=referenced_attributes,
                                                             table_renamings=table_renamings,
                                                             join_predicates=join_predicates,
                                                             joined_tables=joined_tables,
                                                             in_subquery=True)
            subquery_tables = _collect_tables(join["children"])

            # modify the subquery such that all necessary attributes are exported
            select_clause_with_attributes = []
            for subquery_table in subquery_tables:
                select_clause_with_attributes.extend(referenced_attributes[subquery_table])
            renamed_select_attributes = []
            for attribute in select_clause_with_attributes:
                renamed_attribute = f"{attribute.table.alias}_{attribute.attribute}"
                renamed_select_attributes.append({"value": str(attribute), "name": renamed_attribute})
            subquery_mosp["select"] = renamed_select_attributes

            # generate the virtual table name of the subquery
            subquery_target_name = db.TableRef("", "_".join(sorted(table.alias for table in subquery_tables)),
                                               virtual=True)
            for subquery_table in subquery_tables:
                table_renamings[subquery_table] = subquery_target_name

            # generate the subquery predicate, renaming the attributes as appropriate
            for subquery_table in subquery_tables:
                applicable_join_predicates.extend(util.flatten(predicates for partner, predicates
                                                               in join_predicates[subquery_table].items()
                                                               if partner in joined_tables))

            subquery_predicate = [_rename_predicate_if_necessary(predicate, table_renamings).to_mosp() for predicate
                                  in applicable_join_predicates]
            if join_idx == 0:
                subquery_predicate += util.enlist(predicate_map[base_table])

            if util.contains_multiple(subquery_predicate):
                subquery_predicate = {"and": subquery_predicate}
            else:
                subquery_predicate = util.simplify(subquery_predicate)

            mosp_join = {"join": {"value": subquery_mosp, "name": subquery_target_name.alias},
                         "on": subquery_predicate}
            for subquery_table in subquery_tables:
                joined_tables.add(subquery_table)
        else:
            join_partner, join_predicate = join["table"], join["predicate"]
            filter_predicates = util.enlist(predicate_map[join_partner])
            if join_idx == 0:
                filter_predicates += util.enlist(predicate_map[base_table])

            if in_subquery:
                applicable_join_predicates = util.enlist(join_predicate)
            else:
                applicable_join_predicates = util.flatten(predicates for partner, predicates
                                                          in join_predicates[join_partner].items()
                                                          if partner in joined_tables)
            full_predicate = applicable_join_predicates + filter_predicates
            full_predicate = [_rename_predicate_if_necessary(pred, table_renamings) for pred in full_predicate]
            full_predicate = mosp.flatten_and_predicate([pred for pred in full_predicate])

            if util.contains_multiple(full_predicate):
                mosp_predicate = {"and": [pred.to_mosp() for pred in full_predicate]}
            elif not full_predicate:
                raise ValueError()
            else:
                mosp_predicate = util.simplify(full_predicate).to_mosp()

            mosp_join = {"join": mosp.tableref_to_mosp(join_partner),
                         "on": mosp_predicate}
            joined_tables.add(join_partner)

        for predicate in applicable_join_predicates:
            join_table, related_table = predicate.parse_tables()
            join_predicates[join_table][related_table].remove(predicate)
            join_predicates[related_table][join_table].remove(predicate)
        from_list.append(mosp_join)

    select_clause = {"value": {"count": "*"}}
    mosp_data = {"select": select_clause, "from": from_list}
    return mosp_data


def optimize_query(query: mosp.MospQuery, *,
                   table_cardinality_estimation: str = "explain",
                   join_cardinality_estimation: str = "basic",
                   subquery_generation: str = "defensive",
                   dbs: db.DBSchema = db.DBSchema.get_instance(),
                   visualize: bool = False, visualize_args: dict = None,
                   verbose: bool = False, trace: bool = False) -> mosp.MospQuery:
    # if there are no joins in the query, there is nothing to do
    if not util.contains_multiple(query.from_clause()):
        return query

    if table_cardinality_estimation == "sample":
        base_estimator = SamplingCardinalityEstimator()
    elif table_cardinality_estimation == "explain":
        base_estimator = PostgresCardinalityEstimator()
    else:
        raise ValueError("Unknown base table estimation strategy: '{}'".format(table_cardinality_estimation))

    if join_cardinality_estimation == "basic":
        join_estimator = DefaultUESCardinalityEstimator(query)
    elif join_cardinality_estimation == "advanced":
        warnings.warn("Advanced join estimation is not supported yet. Falling back to basic estimation.")
        join_estimator = DefaultUESCardinalityEstimator(query)
    else:
        raise ValueError("Unknown cardinality estimation strategy: '{}'".format(join_cardinality_estimation))

    if subquery_generation == "defensive":
        subquery_generator = DefensiveSubqueryGeneration()
    elif subquery_generation == "greedy":
        subquery_generator = GreedySubqueryGeneration()
    elif subquery_generation == "disabled":
        subquery_generator = NoSubqueryGeneration()
    else:
        raise ValueError("Unknown subquery generation: '{}'".format(subquery_generation))

    predicate_map = _build_predicate_map(query)
    join_predicates = _build_join_map(query)
    optimization_result = _calculate_join_order(query, dbs=dbs, predicate_map=predicate_map,
                                                base_estimator=base_estimator, join_estimator=join_estimator,
                                                subquery_generator=subquery_generator,
                                                visualize=visualize, visualize_args=visualize_args,
                                                verbose=verbose, trace=trace)

    if util.contains_multiple(optimization_result):
        # query contains a cross-product
        ordered_join_trees = sorted(optimization_result, key=operator.attrgetter("final_bound"))
        mosp_datasets = [_generate_mosp_data_for_sequence(optimizer_run.final_order, predicate_map=predicate_map,
                                                          join_predicates=join_predicates)
                         for optimizer_run in ordered_join_trees]
        first_set, *remaining_sets = mosp_datasets
        for partial_query in remaining_sets:
            partial_query["select"] = {"value": "*"}
            first_set["from"].append({"join": {"value": partial_query}})
        return mosp.MospQuery(first_set)
    elif optimization_result:
        join_sequence = optimization_result.final_order.traverse_right_deep()
        mosp_data = _generate_mosp_data_for_sequence(join_sequence, predicate_map=predicate_map,
                                                     join_predicates=join_predicates)
        return mosp.MospQuery(mosp_data)
    else:
        return query
