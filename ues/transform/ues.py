
import abc
import collections
import copy
import functools
import math
import operator
import pprint  # noqa: F401
import typing
import warnings
from dataclasses import dataclass
from typing import Any, Dict, FrozenSet, Generic, Iterable, Iterator, List, Set, Union, Tuple

import matplotlib.pyplot as plt
import networkx as nx
import numpy as np

from transform import db, mosp, util


_T = typing.TypeVar("_T")


class MospQueryPreparation:
    """Removes unsupported structures from an incoming query and reconstructs the query after optimization.

    The removed features do  not influence join order or upper bounds and are therefore safe to be ignored while
    optimizing. More specifically query preparation temporarily removes the following clauses and statements:

    * aggregations in the SELECT clause - the prepared query will be SELECT * FROM ...
    * explicit JOIN statements - the referenced tables will be included in the FROM clause (and WHERE clause)
    * GROUP BY clauses - does not influence the optimal join order
    * ORDER BY clauses - does not influence the optimal join order
    * HAVING clauses - does not influence the optimal join order

    Additionally, temporary aliases will be generated as necessary, such that each table is referenced via an alias.
    The resulting query will be a pure SPJ (select, project, join) query.
    """
    def __init__(self, query: mosp.MospQuery, *, dbs: db.DBSchema = db.DBSchema.get_instance()):
        self._original_query: mosp.MospQuery = query
        self._generated_aliases: List[str] = []
        self._dbs = dbs

    def prepare_query(self) -> mosp.MospQuery:
        """Removes all unsupported/not important clauses and statements from the query."""
        prepared = dict(self._original_query.query)
        prepared["select"] = "*"
        prepared.pop("groupby", None)
        prepared.pop("orderby", None)
        prepared.pop("having", None)

        prepared = self._drop_explicit_joins(prepared)
        prepared = self._generate_table_aliases(prepared)

        return mosp.MospQuery(prepared)

    def reconstruct_query(self, optimized_query: mosp.MospQuery) -> mosp.MospQuery:
        """Adds all temporarily removed clauses and statements back to the freshly optimized query."""
        reconstructed_query = dict(optimized_query.query)

        reconstructed_query["select"] = copy.copy(self._original_query.query["select"])
        if "groupby" in self._original_query.query:
            reconstructed_query["groupby"] = copy.copy(self._original_query.query["groupby"])
        if "orderby" in self._original_query.query:
            reconstructed_query["orderby"] = copy.copy(self._original_query.query["orderby"])
        if "having" in self._original_query.query:
            reconstructed_query["having"] = copy.copy(self._original_query.query["having"])

        if self._generated_aliases:
            reconstructed_query = self._drop_table_aliases(reconstructed_query)

        return mosp.MospQuery(reconstructed_query)

    def _drop_explicit_joins(self, query_data: dict) -> dict:
        complete_from_clause = []
        complete_where_clause = []

        if "where" in query_data:
            complete_where_clause.append(query_data["where"])

        for table in query_data["from"]:
            if "join" in table:
                complete_from_clause.append(table["join"])
                if "on" in table:
                    complete_where_clause.append(table["on"])
            else:
                complete_from_clause.append(table)

        query_data["from"] = complete_from_clause
        if complete_where_clause and len(complete_where_clause) == 1:
            query_data["where"] = complete_where_clause[0]
        elif complete_where_clause and len(complete_where_clause) > 1:
            query_data["where"] = {"and": complete_where_clause}
        return query_data

    def _generate_table_aliases(self, query_data: dict) -> dict:
        tables_without_alias = [tab for tab in query_data["from"] if not isinstance(tab, dict)]

        # all tables without alias get themselves as alias
        aliased_tables = [{"name": tab, "value": tab} if not isinstance(tab, dict) else tab
                          for tab in util.enlist(query_data["from"])]
        query_data["from"] = util.simplify(aliased_tables)

        self._generated_aliases = tables_without_alias
        if "where" in query_data:
            query_data["where"] = self._add_aliases_to_attributes(query_data["where"])

        return query_data

    def _drop_table_aliases(self, query_data: dict) -> dict:
        if not self._generated_aliases:
            return query_data

        original_from = [tab if "join" in tab or tab["value"] not in self._generated_aliases else tab["value"]
                         for tab in query_data["from"]]
        query_data["from"] = original_from

        if "where" in query_data:
            query_data["where"] = self._drop_aliases_from_attributes(query_data["where"])
        for tab in query_data["from"]:
            if isinstance(tab, dict) and "join" in tab and "on" in tab:
                tab["on"] = self._drop_aliases_from_attributes(tab["on"])
                joined_table = tab["join"]["value"]
                tab["join"] = joined_table if joined_table in self._generated_aliases else tab["join"]

        return query_data

    def _add_aliases_to_attributes(self, predicate: Any) -> Any:
        if isinstance(predicate, list):
            return [self._add_aliases_to_attributes(sub_predicate) for sub_predicate in predicate]
        elif isinstance(predicate, dict):
            operation = util.dict_key(predicate)
            if operation == "literal":
                return predicate
            predicate_value = predicate[operation]
            return {operation: self._add_aliases_to_attributes(predicate_value)}
        elif isinstance(predicate, str):
            if "." in predicate:
                # attribute is already aliased!
                return predicate
            else:
                # we found a potential attribute!
                aliased_tablerefs = [db.TableRef(tab, tab) for tab in self._generated_aliases]
                corresponding_table = self._dbs.lookup_attribute(predicate, aliased_tablerefs)
                aliased_attr = f"{corresponding_table.full_name}.{predicate}"  # table is aliased by its full name
                return aliased_attr
        else:
            return predicate

    def _drop_aliases_from_attributes(self, predicate: Any) -> Any:
        if isinstance(predicate, list):
            return [self._drop_aliases_from_attributes(sub_predicate) for sub_predicate in predicate]
        elif isinstance(predicate, dict):
            operation = util.dict_key(predicate)
            if operation == "literal":
                return predicate
            predicate_value = predicate[operation]
            return {operation: self._drop_aliases_from_attributes(predicate_value)}
        elif isinstance(predicate, str):
            table, attribute = predicate.split(".")
            return attribute if table in self._generated_aliases else predicate
        else:
            return predicate


class BaseCardinalityEstimator(abc.ABC):
    """Estimator responsible for the number of rows of a potentially filtered base table."""
    @abc.abstractmethod
    def estimate_rows(self, predicate: Union[mosp.AbstractMospPredicate, List[mosp.AbstractMospPredicate]], *,
                      dbs: db.DBSchema = db.DBSchema.get_instance()) -> int:
        return NotImplemented

    def all_tuples(self, table: db.TableRef, *, dbs: db.DBSchema = db.DBSchema.get_instance()) -> int:
        return dbs.count_tuples(table)


class PostgresCardinalityEstimator(BaseCardinalityEstimator):
    def estimate_rows(self, predicate: Union[mosp.AbstractMospPredicate, List[mosp.AbstractMospPredicate]], *,
                      dbs: db.DBSchema = db.DBSchema.get_instance()) -> int:
        predicate = mosp.MospCompoundPredicate.merge_and(predicate)
        return predicate.estimate_result_rows(sampling=False, dbs=dbs)


class SamplingCardinalityEstimator(BaseCardinalityEstimator):
    def estimate_rows(self, predicate: Union[mosp.AbstractMospPredicate, List[mosp.AbstractMospPredicate]], *,
                      dbs: db.DBSchema = db.DBSchema.get_instance()) -> int:
        predicate = mosp.MospCompoundPredicate.merge_and(predicate)
        return predicate.estimate_result_rows(sampling=True, sampling_pct=25, dbs=dbs)


class PreciseCardinalityEstimator(BaseCardinalityEstimator):
    def estimate_rows(self, predicate: Union[mosp.AbstractMospPredicate, List[mosp.AbstractMospPredicate]], *,
                      dbs: db.DBSchema = db.DBSchema.get_instance()) -> int:
        predicate = mosp.MospCompoundPredicate.merge_and(predicate)
        filter_query = predicate.as_full_query(count_query=True)
        return dbs.execute_query(str(filter_query))


class JoinCardinalityEstimator(abc.ABC):
    """A cardinality estimator is capable of calculating an upper bound of the number result tuples for a given join.

    How this is achieved precisely is up to the concrete estimator.
    """

    @abc.abstractmethod
    def calculate_upper_bound(self, predicate: mosp.AbstractMospPredicate, *,
                              pk_fk_join: bool = False, fk_table: db.TableRef = None,
                              join_tree: "JoinTree" = None) -> int:
        """
        Determines the upper bound (i.e. the maximum number of result tuples) when executing the join as specified
        by the given predicate. There are two possible modes:

        For an n:m join, the current join_tree has to be specified in addition to the predicate itself. This is
        necessary to look up the current bound of the partially executed query, as well as to determine the join
        directionality.

        For a PK/FK join, the Foreign Key table has to be specified in addition to setting `pk_fk_join` to `True`.
        This is once again necessary to figure out join directions without creating dependencies to the schema or
        join graph.
        """
        return NotImplemented

    @abc.abstractmethod
    def stats(self) -> "_TableBoundStatistics":
        return NotImplemented


class DefaultUESCardinalityEstimator(JoinCardinalityEstimator):
    def __init__(self, query: mosp.MospQuery, base_cardinality_estimator: BaseCardinalityEstimator, *,
                 dbs: db.DBSchema = db.DBSchema.get_instance()):
        self.query = query
        self.stats_container = _MFVTableBoundStatistics(query, base_cardinality_estimator=base_cardinality_estimator,
                                                        dbs=dbs)

    def calculate_upper_bound(self, predicate: mosp.AbstractMospPredicate, *,
                              pk_fk_join: bool = False, fk_table: db.TableRef = None,
                              join_tree: "JoinTree" = None) -> int:
        if pk_fk_join:
            # Use simplified formula
            pk_table = util.pull_any(predicate.join_partner_of(fk_table), strict=False).table
            pk_cardinality = self.stats_container.base_estimates[pk_table]

            fk_attributes = util.enlist(predicate.attribute_of(fk_table), strict=False)
            lowest_frequency = min(self.stats_container.base_frequencies[attr] for attr in fk_attributes)

            return lowest_frequency * pk_cardinality

        # use full-fledged formula
        lowest_bound = np.inf
        join_tree_bound = self.stats_container.upper_bounds[join_tree]
        for attr1, attr2 in predicate.join_partners():
            joined_attr = attr1 if join_tree.contains_table(attr1.table) else attr2
            candidate_attr = attr1 if joined_attr == attr2 else attr2

            candidate_bound = self.stats_container.upper_bounds[candidate_attr.table]
            joined_freq = self.stats_container.joined_frequencies[joined_attr]
            candidate_freq = self.stats_container.base_frequencies[candidate_attr]

            distinct_values_joined = join_tree_bound / joined_freq
            distinct_values_candidate = candidate_bound / candidate_freq
            candidate_bound = min(distinct_values_joined, distinct_values_candidate) * joined_freq * candidate_freq
            candidate_bound = math.ceil(candidate_bound)

            if candidate_bound < lowest_bound:
                lowest_bound = candidate_bound

        return lowest_bound

    def stats(self) -> "_MFVTableBoundStatistics":
        return self.stats_container


class _TopKList(Generic[_T]):
    def __init__(self, mcv_list: List[Tuple[_T, int]], *, associated_attribute: db.AttributeRef = None,
                 remainder_frequency: int = None):
        self.associated_attribute: db.AttributeRef = associated_attribute
        self.mcv_list: List[Tuple[_T, int]] = sorted(mcv_list, key=operator.itemgetter(1), reverse=True)
        self.mcv_data: Dict[_T, int] = dict(mcv_list)

        # the double assignment to remainder_frequency is important for min_frequency to work properly on
        # empty mcv_lists!
        self.remainder_frequency: int = 1
        self.remainder_frequency: int = self.min_frequency() if remainder_frequency is None else remainder_frequency

    def has_contents(self) -> bool:
        return len(self.mcv_list) > 0

    def attribute_values(self) -> Set[_T]:
        return set(self.mcv_data.keys())

    def frequency_sum(self) -> int:
        return sum(self.mcv_data.values())

    def contents(self) -> List[Tuple[_T, int]]:
        return self.mcv_list

    def max_frequency(self) -> int:
        return max(self.mcv_data.values(), default=self.remainder_frequency)

    def min_frequency(self) -> int:
        return min(self.mcv_data.values(), default=self.remainder_frequency)

    def head(self) -> Union[Tuple[_T, int], None]:
        if not self.has_contents():
            return None
        return self.mcv_list[0]

    def snap_to(self, snap_value: int) -> "_TopKList[_T]":
        snapped_mcv = [(val, min(freq, snap_value)) for val, freq in self.mcv_list]
        snapped_remainder = min(self.remainder_frequency, snap_value)
        return _TopKList(snapped_mcv, remainder_frequency=snapped_remainder,
                         associated_attribute=self.associated_attribute)

    def count_common_elements(self, other_mcv: "_TopKList[_T]") -> int:
        own_values = set(self.mcv_data.keys())
        other_values = set(other_mcv.mcv_data.keys())
        return len(own_values & other_values)

    def intersects_with(self, other: "_TopKList[_T]") -> bool:
        return not self.mcv_data.keys().isdisjoint(other.mcv_data.keys())

    def join_cardinality_with(self, other: "_TopKList[_T]") -> int:
        cardinality_sum = 0
        for value in self:
            cardinality_sum += self[value] * other[value]
        for value in [value for value in other if value not in self]:
            cardinality_sum += self[value] * other[value]
        return cardinality_sum

    def merge_with(self, other: "_TopKList[_T]") -> "_TopKList[_T]":
        merged_list = []
        for value in self:
            merged_list.append((value, self[value] * other[value]))
        for value in [value for value in other if value not in self]:
            merged_list.append((value, self[value] * other[value]))
        merged_list.sort(key=operator.itemgetter(1))
        remainder_freq = self.remainder_frequency * other.remainder_frequency
        return _TopKList(merged_list, remainder_frequency=remainder_freq)

    def drop_values_from(self, other: "_TopKList[_T]") -> "_TopKList[_T]":
        unique_values = [(val, freq) for val, freq in self.mcv_list if val not in other]
        return _TopKList(unique_values, remainder_frequency=self.remainder_frequency,
                         associated_attribute=self.associated_attribute)

    def drop_value(self, value: _T) -> "_TopKList[_T]":
        remaining_values = [(val, freq) for val, freq in self.mcv_list if val != value]
        return _TopKList(remaining_values, remainder_frequency=self.remainder_frequency,
                         associated_attribute=self.associated_attribute)

    def __getitem__(self, value: _T) -> int:
        return self.mcv_data.get(value, self.remainder_frequency)

    def __contains__(self, value: _T) -> bool:
        return value in self.mcv_data

    def __len__(self) -> int:
        return len(self.mcv_list)

    def __iter__(self) -> Iterator[_T]:
        return list(self.mcv_data.keys()).__iter__()

    def __repr__(self) -> str:
        return str(self)

    def __str__(self) -> str:
        prefix = str(self.associated_attribute) + " :: " if self.associated_attribute else ""
        if self.mcv_list:
            contents = f"max={self.max_frequency()}, min={self.min_frequency()},"
        else:
            contents = "(no MCV data)"
        contents += f" rem={self.remainder_frequency}"
        return prefix + contents


class TopkUESCardinalityEstimator(JoinCardinalityEstimator):
    def __init__(self, query: mosp.MospQuery, base_cardinality_estimator: BaseCardinalityEstimator, *,
                 k: int, dbs: db.DBSchema = db.DBSchema.get_instance()):
        self.query = query
        self.k = k
        self.stats_container = _TopKTableBoundStatistics(query, k,
                                                         base_cardinality_estimator=base_cardinality_estimator,
                                                         dbs=dbs)
        self.dbs = dbs

    def calculate_upper_bound(self, predicate: mosp.AbstractMospPredicate, *,
                              pk_fk_join: bool = False, fk_table: db.TableRef = None,
                              join_tree: "JoinTree" = None) -> int:
        lowest_bound = np.inf
        for (attr1, attr2) in predicate.join_partners():
            if pk_fk_join:
                fk_attr = attr1 if attr1.table == fk_table else attr2
                pk_attr = attr1 if fk_attr == attr2 else attr2
                cardinality = self._calculate_pk_fk_bound(fk_attr, pk_attr)
            else:
                joined_attr = attr1 if join_tree.contains_table(attr1.table) else attr2
                candidate_attr = attr1 if joined_attr == attr2 else attr2
                cardinality = self._calculate_n_m_bound(joined_attr, candidate_attr, join_tree)

            cardinality = round(cardinality)
            if cardinality < lowest_bound:
                lowest_bound = cardinality

        return lowest_bound

    def stats(self) -> "_TopKTableBoundStatistics":
        return self.stats_container

    def _load_tuple_count(self, table: db.TableRef) -> int:
        return self.stats_container.upper_bounds.get(table, self.stats_container.base_estimates[table])

    def _calculate_pk_fk_bound(self, fk_attr: db.AttributeRef, pk_attr: db.AttributeRef) -> int:
        total_bound_fk = self.stats_container.base_estimates[fk_attr.table]
        total_bound_pk = self.stats_container.base_estimates[pk_attr.table]
        fk_mcv = self.stats_container.fetch_mcv_list(fk_attr, snap_value=total_bound_fk)
        pk_mcv = self.stats_container.fetch_mcv_list(pk_attr, snap_value=total_bound_pk)

        return self._bound_formula(fk_mcv, total_bound_fk, pk_mcv, total_bound_pk)

    def _calculate_n_m_bound(self, joined_attr: db.AttributeRef, candidate_attr: db.AttributeRef,
                             join_tree: "JoinTree") -> int:
        total_bound_joined = self.stats_container.upper_bounds[join_tree]
        total_bound_candidate = self.stats_container.upper_bounds[candidate_attr.table]
        joined_mcv = self.stats_container.fetch_mcv_list(joined_attr, joined_table=True, snap_value=total_bound_joined)
        candidate_mcv = self.stats_container.fetch_mcv_list(candidate_attr, snap_value=total_bound_candidate)

        return self._bound_formula(joined_mcv, total_bound_joined, candidate_mcv, total_bound_candidate)

    def _bound_formula(self, first_mcv: _TopKList, first_cardinality: int,
                       second_mcv: _TopKList, second_cardinality: int) -> int:
        first_mcv, second_mcv = first_mcv.snap_to(first_cardinality), second_mcv.snap_to(second_cardinality)

        bound = 0
        while ((first_mcv.has_contents() or second_mcv.has_contents())
               and first_cardinality > 0
               and second_cardinality > 0):
            first_bound, second_bound = 0, 0
            first_value_candidate, second_value_candidate = None, None
            if first_mcv.has_contents():
                first_value_candidate, first_candidate_frequency = first_mcv.head()
                first_bound = first_candidate_frequency * second_mcv[first_value_candidate]
            if second_mcv.has_contents():
                second_value_candidate, second_candidate_frequency = second_mcv.head()
                second_bound = first_mcv[second_value_candidate] * second_candidate_frequency

            selected_bound, selected_candidate = 0, None
            if first_bound >= second_bound:
                selected_bound, selected_candidate = first_bound, first_value_candidate
            else:
                selected_bound, selected_candidate = second_bound, second_value_candidate

            bound += selected_bound

            first_cardinality = max(first_cardinality - first_mcv[selected_candidate], 0)
            second_cardinality = max(second_cardinality - second_mcv[selected_candidate], 0)
            first_mcv = first_mcv.drop_value(selected_candidate).snap_to(first_cardinality)
            second_mcv = second_mcv.drop_value(selected_candidate).snap_to(second_cardinality)

        if first_mcv.remainder_frequency > 0 and second_mcv.remainder_frequency > 0:
            bound += (min(first_cardinality / first_mcv.remainder_frequency,
                          second_cardinality / second_mcv.remainder_frequency)
                      * first_mcv.remainder_frequency
                      * second_mcv.remainder_frequency)

        return math.ceil(bound)


def _is_pk_fk_join(join: mosp.MospBasePredicate, *, dbs: db.DBSchema = db.DBSchema.get_instance()) -> bool:
    first_attr, second_attr = join.collect_attributes()
    pk, fk = None, None

    if dbs.is_primary_key(first_attr):
        pk = first_attr
    elif dbs.is_primary_key(second_attr):
        if pk:
            warnings.warn("PK/PK join found: {}. Treating {} as Foreign Key.".format(join, second_attr))
            fk = second_attr
            return {"pk_fk_join": True, "pk": pk, "fk": fk}
        pk = second_attr

    if dbs.has_secondary_idx_on(first_attr):
        fk = first_attr
    elif dbs.has_secondary_idx_on(second_attr):
        fk = second_attr

    if pk is None or fk is None:
        return {"pk_fk_join": False}
    return {"pk_fk_join": True, "pk": pk, "fk": fk}


class JoinGraph:
    """The join graph provides a nice interface for querying information about the joins we have to execute.

    The graph is treated mutable in that tables will subsequently be marked as included in the join. Many methods
    operate on tables that are either already joined, or not yet joined and therefore provide different results
    depending on the current state of the query processing.
    """

    @staticmethod
    def build_for(query: mosp.MospQuery, *, dbs: db.DBSchema = db.DBSchema.get_instance()) -> "JoinGraph":
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

        predicate_map = query.predicates().predicate_map().joins

        for join_predicate in predicate_map:
            left_tab, right_tab = mosp.MospCompoundPredicate.merge_and(join_predicate).collect_tables()

            # since there may be multiple join predicates for a single join we need to find the most specific one here
            join_types = [_is_pk_fk_join(base_predicate, dbs=dbs)
                          for base_predicate in join_predicate.base_predicates()]
            join_type = next((join_type for join_type in join_types if join_type["pk_fk_join"]), join_types[0])

            pk, fk = (join_type["pk"], join_type["fk"]) if join_type["pk_fk_join"] else (None, None)
            graph.add_edge(left_tab, right_tab, pk_fk_join=join_type["pk_fk_join"], predicate=join_predicate,
                           primary_key=pk, foreign_key=fk)

        # mark each node as PK/FK node or n:m node
        for node in graph.nodes:
            neighbors = graph.adj[node]
            all_pk_fk_joins = all(join_data["pk_fk_join"] for join_data in neighbors.values())
            graph.nodes[node]["pk_fk_node"] = all_pk_fk_joins
            graph.nodes[node]["n_m_node"] = not all_pk_fk_joins

        return JoinGraph(graph)

    def __init__(self, graph: nx.Graph):
        self.graph: nx.Graph = graph

    def join_components(self) -> Set["JoinGraph"]:
        """
        A join component is a subset of all of the joins of a query, s.t. each table in the component is joined with
        at least one other table in the component.

        This implies that no join predicates exist that span multiple components. When executing the query, a cross
        product will have to be performed between the components.
        """
        return set(JoinGraph(self.graph.subgraph(component).copy())
                   for component in nx.connected_components(self.graph))

    def free_n_m_joined_tables(self) -> Set[db.TableRef]:
        """Queries the join graph for all tables that are still free and part of at least one n:m join."""
        free_tables = [tab for tab, node_data in list(self.graph.nodes.data())
                       if node_data["free"] and node_data["n_m_node"]]
        return free_tables

    def free_pk_fk_joins_with(self, table: db.TableRef) -> Dict[db.TableRef, mosp.AbstractMospPredicate]:
        """
        Determines all tables in the join graph that are joined with the given table via a PK/FK join, without
        restricting the roles of each table. Only free tables are included.

        The result maps each candidate partner to the applicable join predicate.
        """
        join_edges = [(partner, join["predicate"]) for partner, join in self.graph.adj[table].items()
                      if join["pk_fk_join"] and self.is_free(partner)]
        return dict(join_edges)

    def free_pk_joins_with(self, table: db.TableRef) -> Dict[db.TableRef, mosp.AbstractMospPredicate]:
        """
        Determines all tables in the join graph that are joined with the given table via a PK/FK join, such that the
        join partner acts as the primary key side. Only free tables are included.

        The result maps each candidate partner to the applicable join predicate.
        """
        join_edges = [(partner, join["predicate"]) for partner, join in self.graph.adj[table].items()
                      if join["pk_fk_join"] and self.is_free(partner) and partner == join["primary_key"].table]
        return dict(join_edges)

    def free_n_m_join_partners_of(self, tables: Union[db.TableRef, List[db.TableRef]]
                                  ) -> Dict[db.TableRef, List[mosp.AbstractMospPredicate]]:
        """
        Determines all (yet free) tables S which are joined with any of the given tables R, such that R and S are
        joined by an n:m join.

        The result maps each candidate partner to all applicable join predicates with any of the provided tables.
        """
        tables = util.enlist(tables)
        join_edges = []
        for table in tables:
            join_partners = [(partner, join["predicate"]) for partner, join in self.graph.adj[table].items()
                             if not join["pk_fk_join"] and self.is_free(partner)]
            join_edges.extend(join_partners)
        return self._join_edges_to_dict(join_edges)

    def is_free(self, table: db.TableRef) -> bool:
        """Checks, whether the given table is not inserted into the join tree, yet."""
        return self.graph.nodes[table]["free"]

    def is_pk_fk_table(self, table: db.TableRef) -> bool:
        """Checks, whether the given table is exclusively joined via PK/FK joins."""
        return self.graph.nodes[table]["pk_fk_node"]

    def is_free_fk_table(self, table: db.TableRef) -> bool:
        """
        Checks, whether the given table is free and takes part in at least one PK/FK join, acting as the FK partner.
        """
        return self.is_free(table) and any(True for join_data in self.graph.adj[table].values()
                                           if join_data["foreign_key"].table == table)

    def is_fk_table(self, table: db.TableRef) -> bool:
        """Checks, whether the given table takes part in at least one PK/FK join as the FK partner."""
        return any(True for join_data in self.graph.adj[table].values()
                   if join_data["foreign_key"].table == table)

    def available_join_paths(self, table: db.TableRef) -> Dict[db.TableRef, List[mosp.AbstractMospPredicate]]:
        """
        Searches for all tables that are already joined and have a valid join predicate with the given (free) table.
        """
        if not self.is_free(table):
            raise ValueError("Join paths for already joined table are undefined")
        join_edges = [(partner, join["predicate"]) for partner, join in self.graph.adj[table].items()
                      if not self.is_free(partner)]
        return self._join_edges_to_dict(join_edges)

    def used_join_paths(self, table: db.TableRef) -> Dict[db.TableRef, List[mosp.AbstractMospPredicate]]:
        """Searches for all joined tables that are joined with the given (also joined i.e. non-free) table."""
        if self.is_free(table):
            raise ValueError("Cannot search for used join paths for a free table. Use available_join_paths() instead!")
        join_edges = [(partner, join["predicate"]) for partner, join in self.graph.adj[table].items()
                      if not self.is_free(partner)]
        return self._join_edges_to_dict(join_edges)

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

    def count_tables(self) -> int:
        return len(self.graph.nodes)

    def contains_table(self, table: db.TableRef) -> bool:
        return table in self.graph.nodes

    def pull_any_table(self) -> db.TableRef:
        return next(iter(self.graph.nodes))

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

    def _join_edges_to_dict(self, join_edges: List[Tuple[db.TableRef, mosp.AbstractMospPredicate]]
                            ) -> Dict[db.TableRef, List[mosp.AbstractMospPredicate]]:
        join_partners_map = collections.defaultdict(list)
        for partner, join in join_edges:
            join_partners_map[partner].append(join)
        return dict(join_partners_map)


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
                predicate = join.predicate().joins.as_and_clause()
                join_tree = join_tree.joined_with_subquery(subquery_join, predicate=predicate)
            else:
                predicate = join.predicate().joins.as_and_clause()
                join_tree = join_tree.joined_with_base_table(join.base_table(), predicate=predicate)
        return join_tree

    @staticmethod
    def for_cross_product(sub_trees: List["JoinTree"]) -> "JoinTree":
        initial_tree, *remainder_trees = sub_trees
        cross_product_tree = initial_tree
        for remaining_tree in remainder_trees:
            cross_product_tree = cross_product_tree.joined_with_subquery(remaining_tree)
        return cross_product_tree

    def __init__(self, *, predicate: mosp.AbstractMospPredicate = None):
        self.left: Union["JoinTree", db.TableRef] = None
        self.right: Union["JoinTree", db.TableRef] = None
        if isinstance(predicate, list):
            raise ValueError()
        self.checkpoint: bool = False
        self.predicate: mosp.AbstractMospPredicate = predicate

    def is_empty(self) -> bool:
        return self.right is None

    def is_singular(self) -> bool:
        return self.left is None

    def previous_checkpoint(self, *, _inner: bool = False) -> "JoinTree":
        if self.is_singular() or self.is_empty():
            return None
        if _inner and self.checkpoint:
            return self
        return self.right.previous_checkpoint(_inner=True)

    def count_checkpoints(self) -> int:
        counter = 1 if self.checkpoint else 0

        if self.left_is_subquery():
            counter += self.left.count_checkpoints()
        if self.right_is_tree():
            counter += self.right.count_checkpoints()

        return counter

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

    def at_base_table(self) -> "JoinTree":
        if self.is_empty():
            raise util.StateError("Empty join tree")
        if self.right_is_base_table():
            return self
        return self.right.at_base_table()

    def all_attributes(self) -> Set[db.AttributeRef]:
        right_attributes = self.right.all_attributes() if self.right_is_tree() else set()
        left_attributes = self.left.all_attributes() if self.left_is_subquery() else set()
        own_attributes = set(self.predicate.collect_attributes()) if self.predicate else set()
        return right_attributes | left_attributes | own_attributes

    def contains_table(self, table: db.TableRef) -> bool:
        return table in self.all_tables()

    def left_is_base_table(self) -> bool:
        return isinstance(self.left, db.TableRef) and self.left is not None

    def right_is_base_table(self) -> bool:
        return isinstance(self.right, db.TableRef)

    def right_is_tree(self) -> bool:
        return isinstance(self.right, JoinTree)

    def left_is_subquery(self) -> bool:
        return isinstance(self.left, JoinTree)

    def with_base_table(self, table: db.TableRef) -> "JoinTree":
        self.right = table
        return self

    def joined_with_base_table(self, table: db.TableRef, *,
                               predicate: mosp.AbstractMospPredicate = None,
                               checkpoint: bool = False) -> "JoinTree":
        if not self.left:
            self.left = table
            self.predicate = predicate
            return self

        self.checkpoint = checkpoint
        new_root = JoinTree(predicate=predicate)
        new_root.left = table
        new_root.right = self
        return new_root

    def joined_with_subquery(self, subquery: "JoinTree", *,
                             predicate: mosp.AbstractMospPredicate = None,
                             checkpoint: bool = False) -> "JoinTree":
        if not self.left:
            self.left = subquery
            self.predicate = predicate
            return self

        self.checkpoint = checkpoint

        new_root = JoinTree(predicate=predicate)
        new_root.left = subquery
        new_root.right = self
        return new_root

    def traverse_right_deep(self) -> List[dict]:
        if self.is_empty():
            return []

        if self.is_singular():
            return [{"subquery": False, "table": self.right}]

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
        if self.is_empty():
            return ""

        if self.is_singular():
            return self.right

        indent_str = (" " * _indentation) + "<- "

        if self.left_is_base_table():
            left_str = indent_str + str(self.left)
        else:
            left_str = indent_str + self.left.pretty_print(_indentation=_indentation+2, _inner=True)

        if self.right_is_base_table():
            right_str = indent_str + str(self.right)
        else:
            right_str = self.right.pretty_print(_indentation=_indentation+2, _inner=True)

        checkpoint_str = "[C]" if self.checkpoint else ""

        if _inner:
            return "\n".join([checkpoint_str + left_str, right_str])
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
        if self.is_empty():
            return "[EMPTY]"

        if self.is_singular():
            return str(self.right)

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
        if self.checkpoint:
            join_str += " [C]"
        return join_str


class _MVFBaseAttributeFrequenciesLoader(Dict[db.AttributeRef, int]):
    def __init__(self, base_estimates: Dict[db.TableRef, int], dbs: db.DBSchema = db.DBSchema.get_instance()):
        self.dbs = dbs
        self.base_estimates = base_estimates
        self.attribute_frequencies = {}

    def __getitem__(self, key: db.AttributeRef) -> int:
        if key not in self.attribute_frequencies:
            top1 = self.dbs.calculate_most_common_values(key, k=1)
            if not top1:
                top1 = 1
            else:
                __, top1 = top1[0]
            top1 = min(top1, self.base_estimates[key.table])
            self.attribute_frequencies[key] = top1
        return self.attribute_frequencies[key]

    def __repr__(self) -> str:
        return str(self)

    def __str__(self) -> str:
        return str(self.attribute_frequencies)


class _MFVJoinAttributeFrequenciesLoader(Dict[db.AttributeRef, int]):
    def __init__(self, base_frequencies: _MVFBaseAttributeFrequenciesLoader):
        self.base_frequencies = base_frequencies
        self.attribute_frequencies = {}
        self.current_multipliers = {}

    def store_multiplier(self, table: db.TableRef, multiplier: int):
        self.current_multipliers[table] = min(multiplier, self.current_multipliers.get(table, np.inf))

    def __getitem__(self, key: db.AttributeRef) -> int:
        if key not in self.attribute_frequencies:
            base_frequency = self.base_frequencies[key]
            multiplier = functools.reduce(operator.mul, (multiplier for table, multiplier
                                                         in self.current_multipliers.items() if table != key.table),
                                          1)
            self.attribute_frequencies[key] = base_frequency * multiplier
        return self.attribute_frequencies[key]

    def __setitem__(self, key: db.AttributeRef, value: int):
        self.attribute_frequencies[key] = value

    def __repr__(self) -> str:
        return str(self)

    def __str__(self) -> str:
        freq_str = "Join frequencies: " + str(self.attribute_frequencies)
        mult_str = f" (current multiplier = {self.current_multipliers})"
        return freq_str + mult_str


class _TopKBaseAttributeFrequenciesLoader(Dict[db.AttributeRef, _TopKList]):
    def __init__(self, k: int, base_estimates: Dict[db.TableRef, int], dbs: db.DBSchema = db.DBSchema.get_instance()):
        self.k = k
        self.dbs = dbs
        self.base_estimates = base_estimates
        self.attribute_mcvs = {}

    def _snap_frequencies_to_max(self, frequencies: List[Tuple[Any, int]], max_freq: int) -> List[Tuple[Any, int]]:
        return [(val, min(freq, max_freq)) for val, freq in frequencies]

    def __getitem__(self, key: db.AttributeRef) -> _TopKList:
        if key not in self.attribute_mcvs:
            frequencies = self.dbs.calculate_most_common_values(key, k=self.k)
            snapped_freqs = self._snap_frequencies_to_max(frequencies, self.base_estimates[key.table])
            top_k = _TopKList(snapped_freqs, associated_attribute=key)
            self.attribute_mcvs[key] = top_k
            return top_k
        return self.attribute_mcvs[key]

    def __repr__(self) -> str:
        return str(self)

    def __str__(self) -> str:
        return str(self.attribute_mcvs)


class _TopKJoinAttributeFrequenciesLoader(Dict[db.AttributeRef, _TopKList]):
    def __init__(self, base_frequencies: _TopKBaseAttributeFrequenciesLoader):
        self.base_mcvs = base_frequencies
        self.attribute_mvcs = {}
        self.current_multipliers = {}

    def store_multiplier(self, table: db.TableRef, multiplier: int):
        self.current_multipliers[table] = min(multiplier, self.current_multipliers.get(table, np.inf))

    def adjust_frequencies(self, mcv_list: _TopKList, adjustment_factor: int) -> _TopKList:
        mcv_entries = mcv_list.contents()
        adjusted_values = [(val, freq * adjustment_factor) for val, freq in mcv_entries]
        adjusted_remainder = mcv_list.remainder_frequency * adjustment_factor
        return _TopKList(adjusted_values, remainder_frequency=adjusted_remainder,
                         associated_attribute=mcv_list.associated_attribute)

    def __getitem__(self, key: db.AttributeRef) -> _TopKList:
        if key not in self.attribute_mvcs:
            base_mcv = self.base_mcvs[key]
            multiplier = functools.reduce(operator.mul, (multiplier for table, multiplier
                                                         in self.current_multipliers.items() if table != key.table),
                                          1)
            adjusted_mcv = self.adjust_frequencies(base_mcv, multiplier)
            self.attribute_mvcs[key] = adjusted_mcv
            return adjusted_mcv
        return self.attribute_mvcs[key]

    def __setitem__(self, key: db.AttributeRef, value: _TopKList):
        self.attribute_mvcs[key] = value

    def __repr__(self) -> str:
        return str(self)

    def __str__(self) -> str:
        return "Join frequencies: " + str(self.attribute_mvcs) + f" (current multiplier = {self.current_multipliers})"


class _AttributeUpdateSet(Generic[_T]):
    def __init__(self):
        self.candidate_updates: Dict[db.AttributeRef, _T] = {}
        self.update_evaluations: Dict[db.AttributeRef, int] = {}

    def register_update(self, attribute: db.AttributeRef, update_data: _T, update_evaluation: int):
        if attribute not in self.candidate_updates:
            self.candidate_updates[attribute] = update_data
            self.update_evaluations[attribute] = update_evaluation
            return
        current_evaluation = self.update_evaluations[attribute]
        if update_evaluation < current_evaluation:
            self.candidate_updates[attribute] = update_data
            self.update_evaluations[attribute] = update_evaluation

    def __iter__(self) -> Iterator[Tuple[db.AttributeRef, _T]]:
        return list(self.candidate_updates.items()).__iter__()

    def __setitem__(self, key: db.AttributeRef, update_data: Tuple[_T, int]):
        update_contents, update_evaluation = update_data
        self.register_update(key, update_contents, update_evaluation)


class _TableBoundStatistics(abc.ABC, Generic[_T]):
    def __init__(self, base_estimates, base_frequencies, joined_frequencies, upper_bounds):
        self._base_estimates = base_estimates
        self._base_frequencies = base_frequencies
        self._joined_frequencies = joined_frequencies
        self._upper_bounds = upper_bounds

    @abc.abstractmethod
    def update_frequencies(self, joined_table: db.TableRef, join_predicate: mosp.AbstractMospPredicate, *,
                           join_tree: JoinTree) -> None:
        return NotImplemented

    @abc.abstractmethod
    def init_pk_join(self, fk_table: db.TableRef, join_predicate: mosp.AbstractMospPredicate) -> None:
        return NotImplemented

    def base_bounds(self) -> Dict[db.TableRef, int]:
        return {tab: bound for tab, bound in self.upper_bounds.items() if isinstance(tab, db.TableRef)}

    def join_bounds(self) -> Dict["JoinTree", int]:
        return {join: bound for join, bound in self.upper_bounds.items() if isinstance(join, JoinTree)}

    def _get_base_estimates(self):
        return self._base_estimates

    def _get_base_frequencies(self):
        return self._base_frequencies

    def _get_joined_frequencies(self):
        return self._joined_frequencies

    def _get_upper_bounds(self):
        return self._upper_bounds

    base_estimates: Dict[db.TableRef, int] = property(_get_base_estimates)
    """Base estimates provide an estimate of the number of tuples in a base table."""

    base_frequencies: Dict[db.AttributeRef, _T] = property(_get_base_frequencies)
    """
    Base frequencies provide a statistic-dependent estimate of the value distribution for attributes of
    base tables.
    """

    joined_frequencies: Dict[db.AttributeRef, _T] = property(_get_joined_frequencies)
    """
    Joined frequencies provide a statistic-dependent estimate of the value distribution for attributes of
    joined tables.
    """

    upper_bounds: Dict[Union[db.TableRef, "JoinTree"], int] = property(_get_upper_bounds)
    """Upper bounds provide a theoretical bound on the number of tuples in a base table or a join tree."""


class _MFVTableBoundStatistics(_TableBoundStatistics[int]):
    """Most Frequent Value bound statistics operate on the most frequent value (i.e. Top-1) per attribute."""
    def __init__(self, query: mosp.MospQuery, *,
                 base_cardinality_estimator: BaseCardinalityEstimator,
                 dbs: db.DBSchema = db.DBSchema.get_instance()):
        self.query = query
        self.dbs = dbs

        base_estimates = _estimate_filtered_cardinalities(query, base_cardinality_estimator, dbs=dbs)
        base_frequencies = _MVFBaseAttributeFrequenciesLoader(base_estimates)
        joined_frequencies = _MFVJoinAttributeFrequenciesLoader(base_frequencies)
        self._jf = joined_frequencies
        upper_bounds = {}
        super().__init__(base_estimates, base_frequencies, joined_frequencies, upper_bounds)

    def update_frequencies(self, joined_table: db.TableRef, join_predicate: mosp.AbstractMospPredicate, *,
                           join_tree: JoinTree) -> None:
        join_tree_before_update = (join_tree.previous_checkpoint() if not join_tree.is_singular()
                                   else join_tree.at_base_table())
        bound_before_update = self.upper_bounds[join_tree_before_update]
        current_bound = self.upper_bounds[join_tree]
        absolute_cardinality_increase = current_bound - bound_before_update
        if absolute_cardinality_increase < 0:
            raise util.StateError("Bounds should only increase, never decrease: "
                                  f"{current_bound} from {bound_before_update}")

        min_new_frequency = np.inf
        joined_attributes = set()
        multipliers: List[db.AttributeRef, int] = []
        update_set: _AttributeUpdateSet[int] = _AttributeUpdateSet()
        for (attr1, attr2) in join_predicate.join_partners():
            joined_attr = attr1 if join_tree_before_update.contains_table(attr1.table) else attr2
            candidate_attr = attr1 if joined_attr == attr2 else attr2
            candidate_frequency = min(self.base_frequencies[candidate_attr], absolute_cardinality_increase)
            joined_frequency = self.joined_frequencies[joined_attr]

            updated_freq = joined_frequency * candidate_frequency
            update_set[joined_attr] = (updated_freq, updated_freq)
            update_set[candidate_attr] = (updated_freq, updated_freq)

            if candidate_frequency < min_new_frequency:
                min_new_frequency = candidate_frequency

            multipliers.append((candidate_attr.table, candidate_frequency))
            joined_attributes.add(joined_attr)
            joined_attributes.add(candidate_attr)

        for attr in [attr for attr in join_tree.all_attributes() if attr not in joined_attributes]:
            self.joined_frequencies[attr] *= min_new_frequency

        multipliers_grouped = util.dict_generate_multi(multipliers)
        multipliers_reduced = util.dict_reduce_multi(multipliers_grouped, lambda __, frequencies: min(frequencies))
        for tab, multiplier in multipliers_reduced.items():
            self._jf.store_multiplier(tab, multiplier)

        for attr, updated_mcv in update_set:
            self.joined_frequencies[attr] = updated_mcv

    def init_pk_join(self, fk_table: db.TableRef, join_predicate: mosp.AbstractMospPredicate) -> None:
        min_fk_freq = np.inf
        for (attr1, attr2) in join_predicate.join_partners():
            fk_attr = attr1 if attr1.table == fk_table else attr2
            fk_frequency = self.base_frequencies[fk_attr]
            if fk_frequency < min_fk_freq:
                min_fk_freq = fk_frequency
        self._jf.store_multiplier(fk_table, min_fk_freq)

    def __repr__(self) -> str:
        return str(self)

    def __str__(self) -> str:
        return str(self.upper_bounds)


class _TopKTableBoundStatistics(_TableBoundStatistics[_TopKList]):
    def __init__(self, query: mosp.MospQuery, k, *,
                 base_cardinality_estimator: BaseCardinalityEstimator,
                 dbs: db.DBSchema = db.DBSchema.get_instance()):
        self.query = query
        self.k = k
        self.dbs = dbs

        base_estimates = _estimate_filtered_cardinalities(query, base_cardinality_estimator, dbs=dbs)
        base_frequencies = _TopKBaseAttributeFrequenciesLoader(self.k, base_estimates, dbs=dbs)
        joined_frequencies = _TopKJoinAttributeFrequenciesLoader(base_frequencies)
        self._jf = joined_frequencies
        upper_bounds = {}
        super().__init__(base_estimates, base_frequencies, joined_frequencies, upper_bounds)

    def fetch_mcv_list(self, attribute: db.AttributeRef, *, joined_table: bool = False,
                       snap_value: int = None) -> _TopKList:
        mcv_list = self.joined_frequencies[attribute] if joined_table else self.base_frequencies[attribute]
        return mcv_list.snap_to(snap_value) if snap_value is not None else mcv_list

    def update_frequencies(self, joined_table: db.TableRef, join_predicate: mosp.AbstractMospPredicate, *,
                           join_tree: JoinTree) -> None:
        join_tree_before_update = (join_tree.previous_checkpoint() if not join_tree.is_singular()
                                   else join_tree.at_base_table())
        bound_before_update = self.upper_bounds[join_tree_before_update]
        current_bound = self.upper_bounds[join_tree]
        absolute_cardinality_increase = current_bound - bound_before_update
        if absolute_cardinality_increase < 0:
            raise util.StateError("Bounds should only increase, never decrease: "
                                  f"{current_bound} from {bound_before_update}")

        min_new_frequency = np.inf
        joined_attributes = set()
        multipliers: List[Tuple[db.AttributeRef, int]] = []
        update_set: _AttributeUpdateSet[_TopKList] = _AttributeUpdateSet()
        for (attr1, attr2) in join_predicate.join_partners():
            joined_attr = attr1 if join_tree_before_update.contains_table(attr1.table) else attr2
            candidate_attr = attr1 if joined_attr == attr2 else attr2
            joined_mcv = self.fetch_mcv_list(joined_attr, joined_table=True)
            candidate_mcv = self.fetch_mcv_list(candidate_attr)

            merged_mcv = joined_mcv.merge_with(candidate_mcv).snap_to(current_bound)

            # TODO: consider "connected" frequencies
            update_set[joined_attr] = (merged_mcv, merged_mcv.max_frequency())
            update_set[candidate_attr] = (merged_mcv, merged_mcv.max_frequency())

            candidate_frequency = candidate_mcv.max_frequency()
            if candidate_frequency < min_new_frequency:
                min_new_frequency = candidate_frequency

            multipliers.append((candidate_attr.table, candidate_frequency))
            joined_attributes.add(joined_attr)
            joined_attributes.add(candidate_attr)

        for attr in [attr for attr in join_tree.all_attributes() if attr not in joined_attributes]:
            mcv = self._jf[attr]
            updated_mcv = self._jf.adjust_frequencies(mcv, min_new_frequency).snap_to(current_bound)
            self.joined_frequencies[attr] = updated_mcv

        multipliers_grouped = util.dict_generate_multi(multipliers)
        multipliers_reduced = util.dict_reduce_multi(multipliers_grouped, lambda __, frequencies: min(frequencies))
        for tab, multiplier in multipliers_reduced.items():
            self._jf.store_multiplier(tab, multiplier)

        for attr, updated_mcv in update_set:
            self.joined_frequencies[attr] = updated_mcv

    def init_pk_join(self, fk_table: db.TableRef, join_predicate: mosp.AbstractMospPredicate) -> None:
        min_fk_freq = np.inf
        for (attr1, attr2) in join_predicate.join_partners():
            fk_attr = attr1 if attr1.table == fk_table else attr2
            fk_frequency = self.base_frequencies[fk_attr].max_frequency()
            if fk_frequency < min_fk_freq:
                min_fk_freq = fk_frequency
        self._jf.store_multiplier(fk_table, min_fk_freq)


@dataclass
class ExceptionRule:
    label: str = ""
    query: str = ""
    subquery_generation: bool = True

    def __repr__(self) -> str:
        return str(self)

    def __str__(self) -> str:
        rule = f"ExceptionRule :: label={self.label}"
        if self.query:
            rule += f" query='{self.query}'"
        rule += f" subquery_generation={self.subquery_generation}"
        return rule


class ExceptionList:
    def __init__(self, rules: Dict[Union[str, mosp.MospQuery], ExceptionRule]):
        self.rules = {self._normalize_key(query): rule for query, rule in rules.items()}

    def _normalize_key(self, key: Union[str, mosp.MospQuery]) -> str:
        if isinstance(key, str):
            return str(mosp.MospQuery.parse(key))
        return str(key)

    def __contains__(self, key: Union[str, mosp.MospQuery]) -> bool:
        key = self._normalize_key(key)
        return key in self.rules

    def __getitem__(self, key: Union[str, mosp.MospQuery]) -> ExceptionRule:
        key = self._normalize_key(key)
        return self.rules.get(key, None)

    def __repr__(self) -> str:
        return str(self)

    def __str__(self) -> str:
        return str(self.rules)


class SubqueryGenerationStrategy(abc.ABC):
    """
    A subquery generator is capable of both deciding whether a certain join should be implemented as a subquery, as
    well as rolling out the transformation itself.
    """

    @abc.abstractmethod
    def execute_as_subquery(self, candidate: db.TableRef, join_graph: JoinGraph, join_tree: JoinTree, *,
                            stats: _MFVTableBoundStatistics,
                            exceptions: ExceptionList = None, query: mosp.MospQuery = None) -> bool:
        return NotImplemented


class DefensiveSubqueryGeneration(SubqueryGenerationStrategy):
    def execute_as_subquery(self, candidate: db.TableRef, join_graph: JoinGraph, join_tree: JoinTree, *,
                            stats: _MFVTableBoundStatistics,
                            exceptions: ExceptionList = None, query: mosp.MospQuery = None) -> bool:
        if exceptions and query in exceptions:
            should_generate_subquery = exceptions[query].subquery_generation
            if not should_generate_subquery:
                return False
        return (stats.upper_bounds[candidate] < stats.base_estimates[candidate]
                and join_graph.count_selected_joins() > 2)


class GreedySubqueryGeneration(SubqueryGenerationStrategy):
    def execute_as_subquery(self, candidate: db.TableRef, join_graph: JoinGraph, join_tree: JoinTree, *,
                            stats: _MFVTableBoundStatistics,
                            exceptions: ExceptionList = None, query: mosp.MospQuery = None) -> bool:
        if exceptions and query in exceptions:
            should_generate_subquery = exceptions[query].subquery_generation
            if not should_generate_subquery:
                return False
        return join_graph.count_selected_joins() > 2


class NoSubqueryGeneration(SubqueryGenerationStrategy):
    def execute_as_subquery(self, candidate: db.TableRef, join_graph: JoinGraph, join_tree: JoinTree, *,
                            stats: _MFVTableBoundStatistics,
                            exceptions: ExceptionList = None, query: mosp.MospQuery = None) -> bool:
        return False


class SmartSubqueryGeneration(SubqueryGenerationStrategy):
    def execute_as_subquery(self, candidate: db.TableRef, join_graph: JoinGraph, join_tree: JoinTree, *,
                            stats: _MFVTableBoundStatistics,
                            exceptions: ExceptionList = None, query: mosp.MospQuery = None) -> bool:
        if exceptions and query in exceptions:
            should_generate_subquery = exceptions[query].subquery_generation
            if not should_generate_subquery:
                return False
        return (stats.upper_bounds[candidate] < stats.base_estimates[candidate] / 100
                and join_graph.count_selected_joins() > 2)


def _estimate_filtered_cardinalities(query: mosp.MospQuery, estimator: BaseCardinalityEstimator, *,
                                     dbs: db.DBSchema = db.DBSchema.get_instance()) -> Dict[db.TableRef, int]:
    """Fetches the PG estimates for all tables in the predicate_map according to their associated filters."""
    cardinality_dict = {}
    predicate_map = query.predicates().filters
    for table in query.collect_tables():
        cardinality_dict[table] = (estimator.estimate_rows(predicate_map[table], dbs=dbs) if table in predicate_map
                                   else estimator.all_tuples(table, dbs=dbs))
    return cardinality_dict


@dataclass
class TrackerEntry:
    full_join: List[db.TableRef]
    candidate_table: Union[db.TableRef, List[db.TableRef]]
    join_bound: int
    candidate_bound: int
    intermediate_bound: int
    index_available: bool = False


class BoundsTracker:
    @staticmethod
    def load_from_json(json_data: list, *, query: mosp.MospQuery = None) -> "BoundsTracker":
        if not json_data:
            json_data = []

        parsed_entries = {}
        current_join_path_length = -np.inf
        current_max_join_path = None

        for raw_entry in json_data:
            parsed_entry = BoundsTracker._parse_entry(raw_entry)
            entry_length = len(parsed_entry.full_join)
            entry_key = frozenset(parsed_entry.full_join)
            parsed_entries[entry_key] = parsed_entry

            if entry_length > current_join_path_length:
                current_join_path_length = entry_length
                current_max_join_path = parsed_entry.full_join

        tracker = BoundsTracker(query=query)
        tracker._bounds_container = parsed_entries
        tracker._current_join_path = current_max_join_path
        tracker._rebuild_index_data()
        return tracker

    @staticmethod
    def _parse_entry(json_data: dict) -> TrackerEntry:
        if "join" not in json_data:
            raise ValueError("Not a valid bounds tracker json, no join given: " + str(json_data))
        parsed_join = [db.TableRef.parse(raw_tab) for raw_tab in json_data["join"]]

        raw_candidate = json_data.get("candidate_table", None)
        if not raw_candidate:
            parsed_candidate = None
        elif isinstance(raw_candidate, list):
            parsed_candidate = [db.TableRef.parse(raw_tab) for raw_tab in raw_candidate]
        else:
            parsed_candidate = db.TableRef.parse(raw_candidate)

        join_bound = json_data.get("join_bound", -1)
        intermediate_bound = json_data.get("intermediate_bound", -1)
        candidate_bound = json_data.get("candidate_bound", -1)
        index_available = json_data.get("index_available", False)

        final_entry = TrackerEntry(parsed_join, parsed_candidate, join_bound, candidate_bound, intermediate_bound,
                                   index_available)
        return final_entry

    def __init__(self, query: mosp.MospQuery = None):
        self._query = query
        self._bounds_container: Dict[FrozenSet[db.TableRef], TrackerEntry] = {}
        self._current_join_path: List[db.TableRef] = []
        self._indexed_tables: Dict[db.TableRef, bool] = collections.defaultdict(bool)  # bool() defaults to False

    def current_bound(self) -> int:
        key = self._to_key(self._current_join_path)
        return self._bounds_container[key].join_bound

    def fetch_bound(self, join: Iterable[db.TableRef]) -> TrackerEntry:
        key = self._to_key(join)
        return self._bounds_container.get(key, None)

    def initialize(self, first_table: db.TableRef, bound: int) -> None:
        entry = TrackerEntry([first_table], first_table, bound, bound, 0)
        key = frozenset([first_table])
        self._bounds_container[key] = entry
        self._current_join_path.append(first_table)

    def store_bound(self, next_table: db.TableRef, *, candidate_bound: int, join_bound: int,
                    indexed_table: bool = False) -> None:
        full_path = self._current_join_path + [next_table]
        current_bound = self._bounds_container[self._to_key(self._current_join_path)].join_bound
        entry = TrackerEntry(full_path, next_table, join_bound, candidate_bound, current_bound, indexed_table)
        key = self._to_key(full_path)
        self._bounds_container[key] = entry
        self._current_join_path.append(next_table)
        if indexed_table:
            self._indexed_tables[next_table] = True

    def merge_with(self, other_tracker: "BoundsTracker", join_bound: int = None) -> None:
        current_bound = self.current_bound()
        partner_bound = other_tracker.current_bound()
        join_bound = (current_bound * partner_bound) if not join_bound else join_bound
        self._bounds_container = util.dict_merge(other_tracker._bounds_container, self._bounds_container)

        full_path = self._current_join_path + other_tracker._current_join_path
        key = self._to_key(full_path)

        self._bounds_container[key] = TrackerEntry(full_path, candidate_table=list(other_tracker._current_join_path),
                                                   join_bound=join_bound,
                                                   intermediate_bound=current_bound,
                                                   candidate_bound=partner_bound,
                                                   index_available=False)
        self._current_join_path.extend(other_tracker._current_join_path)

    def _rebuild_index_data(self) -> None:
        self._indexed_tables = collections.defaultdict(bool, {entry.candidate_table: entry.index_available
                                                              for entry in self._bounds_container.values()})

    def __json__(self) -> list:
        jsonized_entries = []
        for entry in self._bounds_container.values():
            jsonized_entries.append({
                "join": [str(tab) for tab in entry.full_join],
                "join_bound": entry.join_bound,
                "intermediate_bound": entry.intermediate_bound,
                "candidate_table": str(entry.candidate_table),
                "candidate_bound": entry.candidate_bound,
                "index_available": entry.index_available
            })
        return jsonized_entries

    def _to_key(self, key: Iterable[db.TableRef]) -> FrozenSet[db.TableRef]:
        return key if isinstance(key, frozenset) else frozenset(key)

    def __getitem__(self, key: Iterable[db.TableRef]) -> int:
        key = self._to_key(key)
        return self._bounds_container[key]

    def __contains__(self, key: Iterable[db.TableRef]) -> bool:
        key = self._to_key(key)
        return key in self._bounds_container

    def __repr__(self) -> str:
        return str(self)

    def __str__(self) -> str:
        return str(list(self._bounds_container.values()))


def _absorb_pk_fk_hull_of(table: db.TableRef, *, join_graph: JoinGraph, join_tree: JoinTree,
                          subquery_generator: SubqueryGenerationStrategy,
                          bounds_tracker: BoundsTracker = None, pk_fk_bound: int = None,
                          base_table_estimates: Dict[db.TableRef, int], pk_only: bool = False,
                          verbose: bool = False, trace: bool = False) -> JoinTree:
    # TODO: the choice of estimates and the iteration itself are actually not optimal. We do not consider filters at
    # all!
    # A better strategy would be: always merge PKs in first (since they can only reduce the intermediate size)
    # Thereby we would effectively treat FK tables as n:m tables! Why did we make that distinction in the first place?

    logger = util.make_logger(verbose or trace)
    JoinEdge = collections.namedtuple("JoinEdge", ["table", "predicates"])

    if pk_only:
        raw_join_paths = join_graph.free_pk_joins_with(table)
    else:
        raw_join_paths = join_graph.free_pk_fk_joins_with(table)
    join_paths: Dict[db.TableRef, List[mosp.AbstractMospPredicate]] = util.dict_update(raw_join_paths, util.enlist)
    candidate_estimates: Dict[db.TableRef, int] = {joined_table: base_table_estimates[joined_table]
                                                   for joined_table in join_paths}

    pk_fk_join_sequence: List[JoinEdge] = []
    while candidate_estimates:
        # always insert the table with minimum cardinality next
        next_pk_fk_join = util.argmin(candidate_estimates)
        pk_fk_join_sequence.append(
            JoinEdge(table=next_pk_fk_join,
                     predicates=mosp.MospCompoundPredicate.merge_and(join_paths[next_pk_fk_join])))
        join_graph.mark_joined(next_pk_fk_join)

        if bounds_tracker:
            bounds_tracker.store_bound(next_pk_fk_join, candidate_bound=base_table_estimates[next_pk_fk_join],
                                       join_bound=pk_fk_bound, indexed_table=True)

        logger(".. Also including PK/FK join from hull:", next_pk_fk_join)

        # after inserting the join into our join tree, new join paths may become available
        if pk_only:
            fresh_joins = join_graph.free_pk_joins_with(next_pk_fk_join)
        else:
            fresh_joins = join_graph.free_pk_fk_joins_with(next_pk_fk_join)
        join_paths = util.dict_merge(join_paths, util.dict_update(fresh_joins, util.enlist),
                                     update=lambda __, existing_paths, new_paths: existing_paths + new_paths)
        candidate_estimates = util.dict_merge(candidate_estimates,
                                              {join: base_table_estimates[join] for join in fresh_joins})

        candidate_estimates.pop(next_pk_fk_join)
        # TODO: if inserting a FK join, update statistics here

    # TODO: check if executed as subquery and insert into join tree
    for join_edge in pk_fk_join_sequence:
        # TODO: for now we just always use the first predicate available. Is this sufficient or does the choice of
        # predicate matter?
        join_tree = join_tree.joined_with_base_table(join_edge.table, predicate=join_edge.predicates)
    return join_tree


@dataclass
class JoinOrderOptimizationResult:
    final_order: JoinTree
    intermediate_bounds: BoundsTracker
    final_bound: int
    regular: bool


def _calculate_join_order(query: mosp.MospQuery, *,
                          join_estimator: JoinCardinalityEstimator = None,
                          base_estimator: BaseCardinalityEstimator = PostgresCardinalityEstimator(),
                          subquery_generator: SubqueryGenerationStrategy = DefensiveSubqueryGeneration(),
                          exceptions: ExceptionList = None,
                          visualize: bool = False, visualize_args: dict = None,
                          verbose: bool = False, trace: bool = False,
                          dbs: db.DBSchema = db.DBSchema.get_instance()
                          ) -> Union[JoinOrderOptimizationResult, List[JoinOrderOptimizationResult]]:
    join_estimator = join_estimator if join_estimator else DefaultUESCardinalityEstimator(query, dbs=dbs)
    join_graph = JoinGraph.build_for(query, dbs=dbs)

    # In principle it could be that our query involves a cross-product between some of its relations. If that is the
    # case, we cannot simply build a single join tree b/c a tree cannot capture the semantics of cross-product of
    # multiple independent join trees very well. Therefore, we are going to return either a single join tree (which
    # should be the case for most queries), or a list of join trees (one for each closed/connected part of the join
    # graph).
    partitioned_join_trees = [_calculate_join_order_for_join_partition(query, partition,
                                                                       join_cardinality_estimator=join_estimator,
                                                                       base_cardinality_estimator=base_estimator,
                                                                       subquery_generator=subquery_generator,
                                                                       exceptions=exceptions,
                                                                       verbose=verbose, trace=trace,
                                                                       visualize=visualize,
                                                                       visualize_args=visualize_args,
                                                                       dbs=dbs)
                              for partition in join_graph.join_components()]
    return util.simplify(partitioned_join_trees)


@dataclass
class _DirectedJoinEdge:
    partner: db.AttributeRef
    predicate: mosp.AbstractMospPredicate

    def __repr__(self) -> str:
        return str(self)

    def __str__(self) -> str:
        return str(self.partner)


def _calculate_join_order_for_join_partition(query: mosp.MospQuery, join_graph: JoinGraph, *,
                                             join_cardinality_estimator: JoinCardinalityEstimator,
                                             base_cardinality_estimator: BaseCardinalityEstimator,
                                             subquery_generator: SubqueryGenerationStrategy,
                                             exceptions: ExceptionList = None,
                                             visualize: bool = False, visualize_args: dict = None,
                                             verbose: bool = False, trace: bool = False,
                                             dbs: db.DBSchema = db.DBSchema.get_instance()
                                             ) -> JoinOrderOptimizationResult:
    trace_logger = util.make_logger(trace)
    logger = util.make_logger(verbose or trace)

    join_tree = JoinTree.empty_join_tree()
    stats = join_cardinality_estimator.stats()

    bounds_tracker = BoundsTracker(query)  # TODO

    if join_graph.count_tables() == 1:
        trace_logger(".. No joins found")
        only_table = join_graph.pull_any_table()
        join_tree = join_tree.with_base_table(only_table)
        final_bound = stats.base_estimates[only_table]
        return JoinOrderOptimizationResult(join_tree, final_bound=final_bound, intermediate_bounds=None, regular=False)

    if not join_graph.free_n_m_joined_tables():
        # If the query contains more than one table (i.e. at least one join b/c we already circumvented cross products)
        # but none of the joins is an n:m join, than all joins have to be Primary Key/Foreign Key joins. If that is the
        # UES algorithm is not directly applicable, but we can at least use its building blocks to derive a decent join
        # order.
        # TODO: documentation : Why is final_bound the pk_fk_bound, how do we iterate, ... ?
        trace_logger(".. No n:m joins found, calculating snowflake-query order")
        fk_estimates = {table: estimate for table, estimate in stats.base_estimates.items()
                        if join_graph.contains_table(table) and join_graph.is_free_fk_table(table)}
        first_fk_table = util.argmin(fk_estimates)
        join_tree = join_tree.with_base_table(first_fk_table)
        join_graph.mark_joined(first_fk_table)
        bounds_tracker.initialize(first_fk_table, stats.base_estimates[first_fk_table])
        final_bound = max(stats.base_estimates[fk_tab] for fk_tab in join_tree.all_tables()
                          if join_graph.is_fk_table(fk_tab))
        join_tree = _absorb_pk_fk_hull_of(first_fk_table, join_graph=join_graph, join_tree=join_tree,
                                          subquery_generator=NoSubqueryGeneration(),
                                          base_table_estimates=stats.base_estimates,
                                          bounds_tracker=bounds_tracker, pk_fk_bound=final_bound)
        assert not join_graph.contains_free_tables()
        return JoinOrderOptimizationResult(join_tree, final_bound=final_bound, intermediate_bounds=bounds_tracker,
                                           regular=False)

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
            pk_joins = join_graph.free_pk_joins_with(candidate_table).values()

            # For these Primary Key Joins we need to determine the upper bound of the join to use as a potential filter
            # The Bound formula is: MF(candidate, fk_attr) * |pk_table|
            pk_fk_bounds = [join_cardinality_estimator.calculate_upper_bound(predicate, pk_fk_join=True,
                                                                             fk_table=candidate_table)
                            for predicate in pk_joins]
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
            pk_joins = sorted([_DirectedJoinEdge(partner=partner, predicate=predicate) for partner, predicate
                               in join_graph.free_pk_joins_with(lowest_bound_table).items()],
                              key=lambda fk_join_view: stats.base_estimates[fk_join_view.partner])
            logger("Selected first table:", lowest_bound_table, "with PK/FK joins",
                   [pk_table.partner for pk_table in pk_joins])

            bounds_tracker.initialize(lowest_bound_table, lowest_min_bound)

            for pk_join in pk_joins:
                trace_logger(".. Adding PK join with", pk_join.partner, "on", pk_join.predicate)
                join_tree = join_tree.joined_with_base_table(pk_join.partner, predicate=pk_join.predicate)
                join_graph.mark_joined(pk_join.partner)
                bounds_tracker.store_bound(pk_join.partner, candidate_bound=stats.base_estimates[pk_join.partner],
                                           join_bound=lowest_min_bound, indexed_table=True)
                join_tree = _absorb_pk_fk_hull_of(pk_join.partner, join_graph=join_graph, join_tree=join_tree,
                                                  subquery_generator=NoSubqueryGeneration(),
                                                  base_table_estimates=stats.base_estimates,
                                                  pk_only=True,
                                                  bounds_tracker=bounds_tracker, pk_fk_bound=lowest_min_bound,
                                                  verbose=verbose, trace=trace)
                stats.init_pk_join(lowest_bound_table, pk_join.predicate)
                stats.upper_bounds[join_tree] = lowest_min_bound

            # This has nothing to do with the actual algorithm and is merely some technical code for visualization
            if visualize:
                current_iteration += 1
                plt.subplot(n_iterations, 1, current_iteration)
                join_graph.print(title=f"Join graph after base table selection, selected table: {lowest_bound_table}",
                                 **visualize_args)
            trace_logger(".. Base estimates:", stats.base_estimates)
            trace_logger(".. Current bounds:", stats.upper_bounds)
            trace_logger("")
            continue

        trace_logger(".. Current frequencies:", stats.joined_frequencies)

        # Now that the bounds are up-to-date for each relation, we can select the next table to join based on the
        # number of outgoing tuples after including the relation in our current join tree.
        lowest_min_bound = np.inf
        selected_candidate = None
        candidate_bounds = {}
        n_m_join_partners = util.dict_explode(join_graph.free_n_m_join_partners_of(join_tree.all_tables()))
        for candidate_table, predicate in n_m_join_partners:
            # We examine the join between the candidate table and a table that is already part of the join tree.
            # To do so, we need to figure out on which attributes the join takes place. There is one attribute
            # belonging to the existing join tree (the tree_attribute) and one attribute for the candidate table (the
            # candidate_attribute). Likewise, we need to find out how many distinct values are in each relation.
            # The final formula is
            # min(upper(T) / MF(T, a), upper(C) / MF(C, b)) * MF(T, a) * MF(C, b)
            # for join tree T with join attribute a and candidate table C with join attribute b.
            # TODO: this should be subject to the join cardinality estimation strategy, not hard-coded.
            candidate_bound = join_cardinality_estimator.calculate_upper_bound(predicate, join_tree=join_tree)
            tree_table = util.pull_any(predicate.join_partner_of(candidate_table), strict=False).table
            trace_logger(f".. Checking candidate {predicate} between {tree_table}/{candidate_table} with "
                         f"bound {candidate_bound}")

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
        trace_logger(".. Candidate bounds:", candidate_bounds)

        # We have now selected the next n:m join to execute. But before we can actually include this join in our join
        # tree, we have to figure out one last thing: what to with the Primary Key/Foreign Key joins on the selected
        # table. There are two general strategies here: most naturally, they may simply be executed after the candidate
        # table has been joined. However, a more efficient approach could be to utilize the Primary Key tables as
        # filters on the selected candidate to reduce its cardinality before the join and therefore minimize the size
        # of intermediates. This idea corresponds to an execution of the candidate join as a subquery.
        # Which idea is the right one in the current situation is not decided by the algorithm itself. Instead, the
        # decision is left to a policy (the subquery_generator), which decides the appropriate action.

        join_graph.mark_joined(selected_candidate, n_m_join=True, trace=trace)
        join_predicate = (mosp.MospCompoundPredicate.merge_and(
            util.flatten(list(
                join_graph.used_join_paths(selected_candidate).values())),
            alias_map=query._build_alias_map()))
        pk_joins = sorted([_DirectedJoinEdge(partner=partner, predicate=predicate) for partner, predicate
                           in join_graph.free_pk_joins_with(selected_candidate).items()],
                          key=lambda fk_join_view: stats.base_estimates[fk_join_view.partner])
        logger("Selected next table:", selected_candidate, "with PK/FK joins",
               [pk_table.partner for pk_table in pk_joins], "on predicate", join_predicate)

        if pk_joins and subquery_generator.execute_as_subquery(selected_candidate, join_graph, join_tree,
                                                               stats=stats, exceptions=exceptions, query=query):
            subquery_join = JoinTree().with_base_table(selected_candidate)
            subquery_bounds_tracker = BoundsTracker()
            subquery_bounds_tracker.initialize(selected_candidate, stats.base_estimates[selected_candidate])

            for pk_join in pk_joins:
                trace_logger(".. Adding PK join with", pk_join.partner, "on", pk_join.predicate)
                subquery_join = subquery_join.joined_with_base_table(pk_join.partner, predicate=pk_join.predicate)
                join_graph.mark_joined(pk_join.partner)
                subquery_bounds_tracker.store_bound(pk_join.partner,
                                                    candidate_bound=stats.base_estimates[pk_join.partner],
                                                    join_bound=stats.upper_bounds[selected_candidate],
                                                    indexed_table=True)
                subquery_join = _absorb_pk_fk_hull_of(pk_join.partner, join_graph=join_graph, join_tree=subquery_join,
                                                      subquery_generator=NoSubqueryGeneration(),
                                                      base_table_estimates=stats.base_estimates,
                                                      pk_only=True,
                                                      bounds_tracker=subquery_bounds_tracker,
                                                      pk_fk_bound=stats.upper_bounds[selected_candidate])
            join_tree = join_tree.joined_with_subquery(subquery_join, predicate=join_predicate, checkpoint=True)

            bounds_tracker.merge_with(subquery_bounds_tracker, join_bound=lowest_min_bound)
            logger(".. Creating subquery for PK joins", subquery_join)
        else:
            join_tree = join_tree.joined_with_base_table(selected_candidate, predicate=join_predicate, checkpoint=True)
            bounds_tracker.store_bound(selected_candidate, candidate_bound=stats.base_estimates[selected_candidate],
                                       join_bound=lowest_min_bound)
            for pk_join in pk_joins:
                trace_logger(".. Adding PK join with", pk_join.partner, "on", pk_join.predicate)
                join_tree = join_tree.joined_with_base_table(pk_join.partner, predicate=pk_join.predicate)
                join_graph.mark_joined(pk_join.partner)
                bounds_tracker.store_bound(pk_join.partner, candidate_bound=stats.base_estimates[pk_join.partner],
                                           join_bound=stats.upper_bounds[selected_candidate], indexed_table=True)
                join_tree = _absorb_pk_fk_hull_of(pk_join.partner, join_graph=join_graph, join_tree=join_tree,
                                                  subquery_generator=NoSubqueryGeneration(),
                                                  base_table_estimates=stats.base_estimates,
                                                  pk_only=True,
                                                  bounds_tracker=bounds_tracker,
                                                  pk_fk_bound=stats.upper_bounds[selected_candidate])

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
    trace_logger("Final join frequencies:", stats.joined_frequencies)
    trace_logger("Final upper bounds:", stats.join_bounds())
    trace_logger("Final join ordering:", join_tree)

    return JoinOrderOptimizationResult(join_tree, bounds_tracker, stats.upper_bounds[join_tree], True)


def _determine_referenced_attributes(join_sequence: List[dict]) -> Dict[db.TableRef, Set[db.AttributeRef]]:
    referenced_attributes = collections.defaultdict(set)
    for join in join_sequence:
        if join["subquery"]:
            subquery_referenced_attributes = _determine_referenced_attributes(join["children"])
            for table, attributes in subquery_referenced_attributes.items():
                referenced_attributes[table] |= attributes
            predicate: mosp.AbstractMospPredicate = join["predicate"]
            for attribute in predicate.collect_attributes():
                referenced_attributes[attribute.table].add(attribute)
        elif "predicate" in join or join["subquery"]:
            predicate: mosp.AbstractMospPredicate = join["predicate"]
            for attribute in predicate.collect_attributes():
                referenced_attributes[attribute.table].add(attribute)
        else:
            continue
    referenced_attributes = dict(referenced_attributes)
    return referenced_attributes


def _collect_tables(join_sequence: List[dict]) -> Set[db.TableRef]:
    tables = set()
    for join in join_sequence:
        if join["subquery"]:
            tables |= _collect_tables(join["children"])
        else:
            tables.add(join["table"])
    return tables


def _rename_predicate_if_necessary(predicate: mosp.AbstractMospPredicate,
                                   table_renamings: Dict[db.TableRef, db.TableRef]) -> mosp.AbstractMospPredicate:
    for table in predicate.collect_tables():
        if table in table_renamings:
            predicate = predicate.rename_table(from_table=table, to_table=table_renamings[table],
                                               prefix_attribute=True)
    return predicate


def _generate_mosp_data_for_sequence(original_query: mosp.MospQuery, join_sequence: List[dict], *,
                                     join_predicates: Dict[db.TableRef,
                                                           Dict[db.TableRef, List[mosp.AbstractMospPredicate]]] = None,
                                     referenced_attributes: Dict[db.TableRef, Set[db.AttributeRef]] = None,
                                     table_renamings: Dict[db.TableRef, db.TableRef] = None,
                                     joined_tables: Set[db.TableRef] = None,
                                     in_subquery: bool = False):

    # TODO: lots and lots of documentation

    predicate_map = original_query.predicates().predicate_map().filters
    join_predicates = (join_predicates if join_predicates
                       else original_query.predicates().predicate_map().joins.contents())

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
        applicable_join_predicates: List[mosp.AbstractMospPredicate] = []
        if join["subquery"]:
            subquery_mosp = _generate_mosp_data_for_sequence(original_query, join["children"],
                                                             referenced_attributes=referenced_attributes,
                                                             table_renamings=table_renamings,
                                                             join_predicates=join_predicates,
                                                             joined_tables=joined_tables,
                                                             in_subquery=True)
            subquery_tables = _collect_tables(join["children"])

            # modify the subquery such that all necessary attributes are exported
            select_clause_with_attributes: List[db.AttributeRef] = []
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
            full_predicate = mosp.MospCompoundPredicate.merge_and([pred for pred in full_predicate])
            mosp_predicate = util.simplify(full_predicate).to_mosp()

            mosp_join = {"join": mosp.tableref_to_mosp(join_partner),
                         "on": mosp_predicate}
            joined_tables.add(join_partner)

        for predicate in applicable_join_predicates:
            join_table, related_table = predicate.collect_tables()
            join_predicates[join_table][related_table].remove(predicate)
            join_predicates[related_table][join_table].remove(predicate)
        from_list.append(mosp_join)

    select_clause = {"value": {"count": "*"}}
    mosp_data = {"select": select_clause, "from": from_list}
    return mosp_data


# TODO: should not only return final bounds after join, but also the bounds of each input side
# implementing this for cross-product queries is a bit more work

@dataclass
class OptimizationResult:
    query: mosp.MospQuery
    bounds: BoundsTracker
    final_bound: int


def optimize_query(query: mosp.MospQuery, *,
                   table_cardinality_estimation: str = "explain",
                   join_cardinality_estimation: str = "basic",
                   subquery_generation: str = "defensive",
                   topk_list_length: int = None,
                   exceptions: ExceptionList = None,
                   dbs: db.DBSchema = db.DBSchema.get_instance(),
                   visualize: bool = False, visualize_args: dict = None,
                   verbose: bool = False, trace: bool = False,
                   introspective: bool = False) -> Union[mosp.MospQuery, OptimizationResult]:

    logger = util.make_logger(verbose or trace)

    # if there are no joins in the query, there is nothing to do
    if not isinstance(query.from_clause(), list) or not util.contains_multiple(query.from_clause()):
        logger("Query contains no joins, nothing to do.")
        return query

    logger("Input query:", query)
    query_preparation = MospQueryPreparation(query, dbs=dbs)
    prepared_query = query_preparation.prepare_query()

    if table_cardinality_estimation == "sample":
        base_estimator = SamplingCardinalityEstimator()
    elif table_cardinality_estimation == "explain":
        base_estimator = PostgresCardinalityEstimator()
    elif table_cardinality_estimation == "precise":
        base_estimator = PreciseCardinalityEstimator()
    else:
        raise ValueError("Unknown base table estimation strategy: '{}'".format(table_cardinality_estimation))

    if join_cardinality_estimation == "basic":
        join_estimator = DefaultUESCardinalityEstimator(prepared_query, base_cardinality_estimator=base_estimator,
                                                        dbs=dbs)
    elif join_cardinality_estimation == "topk":
        k = topk_list_length if topk_list_length else 15
        logger("Running TopK cardinality estimation with k =", k)
        join_estimator = TopkUESCardinalityEstimator(prepared_query, k=k, base_cardinality_estimator=base_estimator,
                                                     dbs=dbs)
    else:
        raise ValueError("Unknown cardinality estimation strategy: '{}'".format(join_cardinality_estimation))

    if subquery_generation == "defensive":
        subquery_generator = DefensiveSubqueryGeneration()
    elif subquery_generation == "greedy":
        subquery_generator = GreedySubqueryGeneration()
    elif subquery_generation == "disabled":
        subquery_generator = NoSubqueryGeneration()
    elif subquery_generation == "smart":
        subquery_generator = SmartSubqueryGeneration()
    else:
        raise ValueError("Unknown subquery generation: '{}'".format(subquery_generation))

    optimization_result = _calculate_join_order(prepared_query, dbs=dbs,
                                                base_estimator=base_estimator, join_estimator=join_estimator,
                                                subquery_generator=subquery_generator,
                                                exceptions=exceptions,
                                                visualize=visualize, visualize_args=visualize_args,
                                                verbose=verbose, trace=trace)

    if util.contains_multiple(optimization_result):
        # If our query contains a cross-product, we need to carry out some additional final steps:
        # First up, the join order will contain not one entry, but multiple entries - one for each part of the query.
        # In the end, all cross-products will be implemented as subqueries, joining the initial query without any
        # predicate
        multiple_optimization_results: List[JoinOrderOptimizationResult] = optimization_result

        # In order to construct an efficient final query, these partial join orders have to be sorted such that
        # each additional partial order introduces as few new tuples as possible
        ordered_join_trees = sorted(multiple_optimization_results, key=operator.attrgetter("final_bound"))

        # Based on this sorted sequence of join orders, the corresponding queries can be generated
        mosp_datasets = [_generate_mosp_data_for_sequence(prepared_query,
                                                          optimizer_run.final_order.traverse_right_deep())
                         for optimizer_run in ordered_join_trees]

        # The queries than need to be stitched together to form a final result query
        first_set, *remaining_sets = mosp_datasets
        for partial_query in remaining_sets:
            partial_query["select"] = {"value": "*"}
            first_set["from"].append({"join": {"value": partial_query}})
        final_query = mosp.MospQuery(first_set)
        final_query = query_preparation.reconstruct_query(final_query)

        if not introspective:
            # If introspection is off, we are done now.
            return final_query

        # But we do not only need the query if introspection is on. In addition, we need to inform the user of the
        # optimization method about the upper bounds that have been calculated along the way.
        # In order to do so, we need to merge the partial upper bounds of all the subqueries/cross product parts
        merged_bounds = BoundsTracker()
        for intermediate_bounds in [partial_result.intermediate_bounds for partial_result
                                    in ordered_join_trees if partial_result.intermediate_bounds]:
            merged_bounds.merge_with(intermediate_bounds)
        return OptimizationResult(final_query, merged_bounds, merged_bounds.current_bound())
    elif optimization_result:
        # If our query does not contain cross-products, our job is singificantly easier - just extract the necessary
        # data and supply it in a nice format.
        single_optimization_result: JoinOrderOptimizationResult = optimization_result
        join_sequence = single_optimization_result.final_order.traverse_right_deep()
        mosp_data = _generate_mosp_data_for_sequence(prepared_query, join_sequence)
        final_query = mosp.MospQuery(mosp_data)
        final_query = query_preparation.reconstruct_query(final_query)
        if introspective:
            return OptimizationResult(final_query, single_optimization_result.intermediate_bounds,
                                      single_optimization_result.final_bound)
        return final_query
    else:
        raise util.StateError("No optimization result")
