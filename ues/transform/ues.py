
import abc
import collections
import pprint
from typing import Dict, List, Set, Union, Tuple

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
        pass


class DefaultUESCardinalityEstimator(JoinCardinalityEstimator):
    def __init__(self, query: mosp.MospQuery):
        # TODO: determine maximum frequency values for each attribute
        self.query = query

    def calculate_upper_bound(self) -> int:
        # TODO: implementation
        return 0


class BaseTableCardinalityEstimator(abc.ABC):

    @abc.abstractmethod
    def estimate_rows(self, predicate: Union[mosp.MospPredicate, mosp.CompoundMospFilterPredicate], *,
                      dbs: db.DBSchema = db.DBSchema.get_instance()) -> int:
        pass

    def all_tuples(self, table: db.TableRef, *, dbs: db.DBSchema = db.DBSchema.get_instance()) -> int:
        return dbs.count_tuples(table)


class PostgresBaseTableCardinalityEstimator(BaseTableCardinalityEstimator):
    def estimate_rows(self, predicate: Union[mosp.MospPredicate, mosp.CompoundMospFilterPredicate], *,
                      dbs: db.DBSchema = db.DBSchema.get_instance()) -> int:
        return predicate.estimate_result_rows(sampling=False)


class SamplingBaseTableCardinalityEstimator(BaseTableCardinalityEstimator):
    def estimate_rows(self, predicate: Union[mosp.MospPredicate, mosp.CompoundMospFilterPredicate], *,
                      dbs: db.DBSchema = db.DBSchema.get_instance()) -> int:
        return predicate.estimate_result_rows(sampling=True)


class SubqueryGenerationStrategy(abc.ABC):
    """
    A subquery generator is capable of both deciding whether a certain join should be implemented as a subquery, as
    well as rolling out the transformation itself.
    """

    @abc.abstractmethod
    def execute_as_subquery(self, candidate: db.TableRef, join_graph: nx.DiGraph, *,
                            current_bounds, current_frequencies, base_estimates) -> bool:
        pass


class DefensiveSubqueryGeneration(SubqueryGenerationStrategy):
    def execute_as_subquery(self, candidate: db.TableRef, join_graph: nx.DiGraph, *,
                            current_bounds, current_frequencies, base_estimates) -> bool:
        return current_bounds[candidate] < base_estimates[candidate]


class GreedySubqueryGeneration(SubqueryGenerationStrategy):
    def execute_as_subquery(self, candidate: db.TableRef, join_graph: nx.DiGraph, *,
                            current_bounds, current_frequencies, base_estimates) -> bool:
        return True


def _is_pk_fk_join(join: mosp.MospPredicate, *, dbs: db.DBSchema = db.DBSchema.get_instance()) -> bool:
    first_attr, second_attr = join.parse_left_attribute(), join.parse_right_attribute()
    pk, fk = None, None

    if dbs.is_primary_key(first_attr):
        pk = first_attr
    elif dbs.is_primary_key(second_attr):
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
    def __init__(self, table_attribute: db.AttributeRef, partner_attribute: db.AttributeRef, *,
                 predicate: mosp.MospPredicate = None):
        self.table_attribute = table_attribute
        self.partner_attribute = partner_attribute

        self.table = table_attribute.table
        self.partner = partner_attribute.table

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
    """The join graph provides a nice interface for querying information about the joins we have to execute."""

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

    def __init__(self, graph: nx.DiGraph):
        self.graph = graph

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

    def free_pk_joins(self, table: db.TableRef) -> List[_EqualityJoinView]:
        """
        Determines all (yet free) tables S which are joined with the given table R, such that S joined R is a Primary
        Key/Foreign Key join with S acting as the Primary Key and (the given) table R acting as the Foreign Key.

        The resulting list contains one entry for each Primary Key/Foreign Key join.
        E.g., if there is a PK/FK join between R.a = S.b, free_pk_joins(R) will return a join view for R with S as its
        partner (provided that S is still a free table). If there are multiple PK/FK joins available on R, an entry
        is created for each one. The same holds if R and S are joined via multiple attributes
        (such as R.a = S.b AND R.c = S.d): Each such predicate will be returned in its own view.
        """
        join_partners: List[Tuple[db.TableRef, List[mosp.MospPredicate]]] = (
            [(partner, join["predicate"]) for partner, join in self.graph.adj[table].items()
             if join["pk_fk_join"] and partner == join["primary_key"].table and self.is_free(partner)])
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
                 if not join["pk_fk_join"] and self.graph.nodes[partner]["free"]])
            join_views.extend(self._explode_compound_predicate_edges(table, join_partners))
        return join_views

    def is_free(self, table: db.TableRef) -> bool:
        """Checks, whether the given table is not inserted into the join tree, yet."""
        return self.graph.nodes[table]["free"]

    def contains_free_n_m_tables(self) -> bool:
        """Checks, whether at least one free table remains in the graph."""
        return any(node_data["free"] for (__, node_data) in list(self.graph.nodes.data()) if node_data["n_m_node"])

    def contains_free_tables(self) -> bool:
        return any(free for (__, free) in self.graph.nodes.data("free"))

    def mark_joined(self, table: db.TableRef):
        """Annotates the given table as joined in the graph."""
        self.graph.nodes[table]["free"] = False

    def count_free_joins(self, table: db.TableRef):
        return len([partner for partner, __ in self.graph.adj[table].items() if self.is_free(partner)])

    def print(self):
        """Writes the current join graph structure to a matplotlib device."""
        node_labels = {node: node.alias for node in self.graph.nodes}
        edge_sizes = [3.0 if pk_fk_join else 1.0 for (__, __, pk_fk_join) in self.graph.edges.data("pk_fk_join")]
        node_edge_color = ["blue" if free else "grey" for (__, free) in self.graph.nodes.data("free")]
        nx.draw_shell(self.graph, with_labels=True,
                      width=edge_sizes, linewidths=1.8, node_size=1750,
                      node_color="white", edgecolors=node_edge_color, edge_color="black",
                      labels=node_labels)

    def _explode_compound_predicate_edges(self, table, join_edges):
        join_views = []
        for partner, compound_join in join_edges:
            for join_predicate in compound_join:
                table_attribute = join_predicate.attribute_of(table)
                partner_attribute = join_predicate.attribute_of(partner)
                join_views.append(_EqualityJoinView(table_attribute, partner_attribute, predicate=join_predicate))
        return join_views


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
        join_tree = JoinTree.empty_join_tree()
        join_tree = join_tree.with_base_table(query.base_table())
        for join in query.joins():
            if join.is_subquery():
                subquery_join = JoinTree.load_from_query(join.subquery)
                join_tree = join_tree.joined_with_subquery(subquery_join)
            else:
                join_tree = join_tree.joined_with_base_table(join.base_table())
        return join_tree

    def __init__(self):
        self.left = None
        self.right = None

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

    def left_is_base_table(self) -> bool:
        return isinstance(self.left, db.TableRef)

    def right_is_base_table(self) -> bool:
        return isinstance(self.right, db.TableRef)

    def left_is_subquery(self) -> bool:
        return isinstance(self.left, JoinTree)

    def with_base_table(self, table: db.TableRef) -> "JoinTree":
        self.right = table
        return self

    def joined_with_base_table(self, table: db.TableRef) -> "JoinTree":
        if not self.left:
            self.left = table
            return self

        new_root = JoinTree()
        new_root.left = table
        new_root.right = self
        return new_root

    def joined_with_subquery(self, subquery: "JoinTree") -> "JoinTree":
        if not self.left:
            self.left = subquery
            return self

        new_root = JoinTree()
        new_root.left = subquery
        new_root.right = self
        return new_root

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

        return f"{right_label} ⋈ {left_label}"  # let's read joins from left to right!


class _AttributeFrequenciesLoader:
    def __init__(self, dbs: db.DBSchema = db.DBSchema.get_instance()):
        self.dbs = dbs
        self.attribute_frequencies = {}

    def __getitem__(self, key: Tuple[Union[db.TableRef, JoinTree], db.AttributeRef]) -> int:
        __, attribute = key
        if key not in self.attribute_frequencies:
            top1 = self.dbs.load_most_common_values(attribute, k=1)
            if not top1:
                top1 = 1
            else:
                __, top1 = top1[0]
                top1 = round(top1 * self.dbs.load_tuple_count(attribute.table))
            self.attribute_frequencies[key] = top1
        return self.attribute_frequencies[key]

    def __setitem__(self, key: Tuple[Union[db.TableRef, JoinTree], db.AttributeRef], value: int):
        self.attribute_frequencies[key] = value


class _TopKBoundsList:
    """The TopKBoundsList combines storing information about attribute frequencies with upper bounds."""
    def most_frequent_value(self):
        # TODO: implementation
        pass

    def full_list(self):
        # TODO: implementation
        pass


def _build_predicate_map(query: mosp.MospQuery
                         ) -> Dict[db.TableRef, Union[mosp.MospPredicate, mosp.CompoundMospFilterPredicate]]:
    """The predicate map is a dictionary which maps each table to the filter predicates that apply to this table."""
    all_filter_predicates = [pred for pred in query.predicates() if not pred.is_join_predicate()]
    raw_predicate_map = collections.defaultdict(list)

    for filter_pred in all_filter_predicates:
        raw_predicate_map[filter_pred.parse_left_attribute().table].append(filter_pred)

    aggregated_predicate_map = {table: mosp.CompoundMospFilterPredicate.build_and_predicate(predicate)
                                for table, predicate in raw_predicate_map.items()}

    for tab in query.collect_tables():
        if tab not in aggregated_predicate_map:
            aggregated_predicate_map[tab] = None

    return aggregated_predicate_map


def _estimate_filtered_cardinalities(predicate_map: dict, estimator: BaseTableCardinalityEstimator, *,
                                     dbs: db.DBSchema = db.DBSchema.get_instance()) -> Dict[db.TableRef, int]:
    """Fetches the PG estimates for all tables in the predicate_map according to their associated filters."""
    cardinality_dict = {}
    for table, predicate in predicate_map.items():
        cardinality_dict[table] = estimator.estimate_rows(predicate) if predicate else estimator.all_tuples(table)
    return cardinality_dict


def _calculate_join_order(query: mosp.MospQuery, *,
                          join_cardinality_estimator: JoinCardinalityEstimator = None,
                          base_cardinality_estimator: BaseTableCardinalityEstimator = PostgresBaseTableCardinalityEstimator(),
                          subquery_generator: SubqueryGenerationStrategy = DefensiveSubqueryGeneration(),
                          dbs: db.DBSchema = db.DBSchema.get_instance()
                          ) -> Union[JoinTree, List[JoinTree]]:
    join_cardinality_estimator = join_cardinality_estimator if join_cardinality_estimator else DefaultUESCardinalityEstimator(query)
    join_graph = _JoinGraph.build_for(query)

    # In principle it could be that our query involves a cross-product between some of its relations. If that is the
    # case, we cannot simply build a single join tree b/c a tree cannot capture the semantics of cross-product of
    # multiple independent join trees very well. Therefore, we are going to return either a single join tree (which
    # should be the case for most queries), or a list of join trees (one for each closed/connected part of the join
    # graph).
    partitioned_join_trees = [_calculate_join_order_for_join_partition(query, partition,
                                                                       join_cardinality_estimator=join_cardinality_estimator,
                                                                       base_cardinality_estimator=base_cardinality_estimator,
                                                                       subquery_generator=subquery_generator,
                                                                       dbs=dbs)
                              for partition in join_graph.join_components()]
    return util.simplify(partitioned_join_trees)


def _calculate_join_order_for_join_partition(query: mosp.MospQuery, join_graph: _JoinGraph, *,
                                             join_cardinality_estimator: JoinCardinalityEstimator,
                                             base_cardinality_estimator: BaseTableCardinalityEstimator,
                                             subquery_generator: SubqueryGenerationStrategy,
                                             dbs: db.DBSchema = db.DBSchema.get_instance()) -> JoinTree:
    join_tree = JoinTree.empty_join_tree()
    base_table_estimates = _estimate_filtered_cardinalities(_build_predicate_map(query), base_cardinality_estimator,
                                                            dbs=dbs)
    attribute_frequencies = _AttributeFrequenciesLoader(dbs=dbs)
    upper_bounds = {}  # will be initialized in the main loop

    if not join_graph.free_n_m_joined_tables():
        # TODO: if the query only consists of PK/FK joins, we need to determine the optimal order here
        raise ValueError("Query only consists of PK/FK joins")

    # The UES algorithm will iteratively expand the join order one join at a time. In each iteration, the best n:m
    # join (after/before being potentially filtered via PK/FK joins) is selected. In the join graph, we keep track of
    # all the tables that we still have to join.
    n_iterations = len(join_graph.free_n_m_joined_tables())
    current_iteration = 0
    plt.figure(figsize=(10, 10 * n_iterations))
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
            filter_estimate = base_table_estimates[candidate_table]

            # Now, let's look at all remaining PK/FK joins with the candidate table. We can get these easily via
            # join_graph.free_pk_joins.
            pk_joins = [(join_view.partner, join_view.table_attribute)
                        for join_view in join_graph.free_pk_joins(candidate_table)]
            pk_fk_bounds = [attribute_frequencies[(candidate_table, attribute)] * base_table_estimates[partner]
                            for (partner, attribute) in pk_joins]  # Formula: MF(candidate, fk_attr) * |pk_table|
            candidate_min_bound = min([filter_estimate] + pk_fk_bounds)
            upper_bounds[candidate_table] = candidate_min_bound
            if candidate_min_bound < lowest_min_bound:
                lowest_min_bound = candidate_min_bound
                lowest_bound_table = candidate_table

        # If we are choosing the base table, just insert it right away and continue with the next table
        if join_tree.is_empty():
            print("Selected first table:", lowest_bound_table)
            join_tree = join_tree.with_base_table(lowest_bound_table)
            join_graph.mark_joined(lowest_bound_table)

            # FIXME: should this also include all PK/FK joins on the base table?!
            pk_joins = sorted(join_graph.free_pk_joins(lowest_bound_table),
                              key=lambda fk_join_view: base_table_estimates[fk_join_view.partner])
            for pk_table in [pk_join.partner for pk_join in pk_joins]:
                join_tree = join_tree.joined_with_base_table(pk_table)
                join_graph.mark_joined(pk_table)

            # FIXME: should update statistics here?!
            upper_bounds[join_tree] = lowest_min_bound

            current_iteration += 1
            plt.subplot(n_iterations, 1, current_iteration)
            join_graph.print()

            continue

        # Now that the bounds are up-to-date for each relation, we can select the next table to join based on the
        # number of outgoing tuples after including the relation in our current join tree.
        lowest_min_bound = np.inf
        selected_candidate = None
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
            n_values_tree = (upper_bounds[join_tree]
                             /
                             attribute_frequencies[(join_tree, tree_attribute)])
            n_values_candidate = (upper_bounds[candidate_table]
                                  /
                                  attribute_frequencies[(candidate_table, candidate_attribute)])
            candidate_bound = (min(n_values_tree, n_values_candidate)
                               * attribute_frequencies[(join_tree, tree_attribute)]
                               * attribute_frequencies[candidate_table, candidate_attribute])
            if candidate_bound < lowest_min_bound:
                lowest_min_bound = candidate_bound
                selected_candidate = candidate_table

        # We have now selected the next n:m join to execute. But before we can actually include this join in our join
        # tree, we have to figure out one last thing: what to with the Primary Key/Foreign Key joins on the selected
        # table. There are two general strategies here: most naturally, they may simply be executed after the candidate
        # table has been joined. However, a more efficient approach could be to utilize the Primary Key tables as
        # filters on the selected candidate to reduce its cardinality before the join and therefore minimize the size
        # of intermediates. This idea corresponds to an execution of the candidate join as a subquery.
        # Which idea is the right one in the current situation is not decided by the algorithm itself. Instead, the
        # decision is left to a policy (the subquery_generator), which decides the appropriate action.
        pk_joins = sorted(join_graph.free_pk_joins(selected_candidate),
                          key=lambda fk_join_view: base_table_estimates[fk_join_view.partner])
        print("Selected next table:", selected_candidate, "with PK joins", [pk_table.partner for pk_table in pk_joins])
        new_join_tree = join_tree
        if subquery_generator.execute_as_subquery(selected_candidate, join_graph,
                                                  current_bounds=upper_bounds,
                                                  current_frequencies=attribute_frequencies,
                                                  base_estimates=base_table_estimates):
            subquery_join = JoinTree().with_base_table(selected_candidate)
            for pk_join in pk_joins:
                subquery_join = subquery_join.joined_with_base_table(pk_join.partner)
            new_join_tree = new_join_tree.joined_with_subquery(subquery_join)
        else:
            new_join_tree = new_join_tree.joined_with_base_table(selected_candidate)
            for pk_join in pk_joins:
                new_join_tree = new_join_tree.joined_with_base_table(pk_join.partner)

        # Update our statistics based on the join(s) we just executed
        upper_bounds[new_join_tree] = lowest_min_bound
        for join_view in join_graph.free_n_m_join_partners(join_tree.all_tables()):
            candidate_attribute, tree_attribute = join_view.table_attribute, join_view.partner_attribute
            attribute_frequencies[(new_join_tree, tree_attribute)] *= attribute_frequencies[(selected_candidate, candidate_attribute)]

        # Update the join graph
        join_graph.mark_joined(selected_candidate)
        for fk_join_view in pk_joins:
            join_graph.mark_joined(fk_join_view.partner)

        # Store the new join tree
        join_tree = new_join_tree

        current_iteration += 1
        plt.subplot(n_iterations, 1, current_iteration)
        join_graph.print()

    # TODO: insert dangling FK-tables

    assert not join_graph.contains_free_tables()

    return join_tree


def optimize_query(query: mosp.MospQuery, *,
                   join_cardinality_estimation: str = "basic",
                   subquery_generation: str = "defensive",
                   dbs: db.DBSchema = db.DBSchema.get_instance()) -> mosp.MospQuery:
    if join_cardinality_estimation == "basic":
        cardinality_estimator = DefaultUESCardinalityEstimator(query)
    elif join_cardinality_estimation == "advanced":
        # TODO
        pass
    else:
        raise ValueError("Unknown cardinality estimation strategy: '{}'".format(join_cardinality_estimation))

    if subquery_generation == "defensive":
        subquery_generator = DefensiveSubqueryGeneration()
    elif subquery_generation == "greedy":
        subquery_generator = GreedySubqueryGeneration()
    else:
        raise ValueError("Unknown subquery generation: '{}'".format(subquery_generation))

    join_order = _calculate_join_order(query, dbs=dbs,
                                       cardinality_estimator=cardinality_estimator,
                                       subquery_generator=subquery_generator)

    # TODO: generate the optimized mosp query based on the join order

    return None
