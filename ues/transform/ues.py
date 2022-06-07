
import abc
import collections
from typing import Dict, List, Set, Union

from transform import db, mosp


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


class SubqueryGenerationStrategy(abc.ABC):
    """
    A subquery generator is capable of both deciding whether a certain join should be implemented as a subquery, as
    well as rolling out the transformation itself.
    """

    @abc.abstractmethod
    def execute_as_subquery(self) -> bool:
        pass


class DefensiveSubqueryGeneration(SubqueryGenerationStrategy):
    pass


class GreedySubqueryGeneration(SubqueryGenerationStrategy):
    pass


# TODO: potentially we don't need the _TableJoin class at all?
class _TableJoin:
    """
    A Table join is a tailored view on joins, containing just the tables being joined as well as the kind of
    join (n:m or PK/FK).

    More specifically, a table join is a base-level join, i.e. a join between two base relations.

    For a structure that is capable of representing higher-level joins, i.e. of joins of joins, see _JoinTree
    """
    def __init__(self, predicate: mosp.MospPredicate):
        """
        A TableJoin will be automatically constructed based on the join predicate.
        All necessary information is extracted from this object.

        By default, the join is assumed to be an n:m join, altough this can be changed later on via
        the `mark_pk_fk_join` method.
        """
        if not predicate.is_join_predicate():
            raise ValueError("Not a join predicate: '{}'".format(predicate))
        first, second = predicate.parse_left_attribute(), predicate.parse_right_attribute()
        if second.table.qualifier() < first.table.qualifier():
            first, second = second, first
        self.first = first.table
        self.second = second.table
        self.first_attr = first
        self.second_attr = second
        self.predicate = predicate
        self.pk, self.fk = None, None

    def mark_pk_fk_join(self, pk: db.TableRef, fk: db.TableRef):
        """Annotates this join as being a primary key/foreign key join."""
        if pk not in [self.first, self.second] or fk not in [self.first, self.second]:
            raise ValueError(f"pk {pk} or fk {fk} not in join {self}")
        self.pk = pk
        self.fk = fk

    def is_pk_fk_join(self) -> bool:
        """Checks, whether this join is known to be a Primary key/Foreign key join."""
        return self.pk is not None

    def is_n_m_join(self) -> bool:
        """Checks, whether this join is _believed_ (or known) to be an n:m join."""
        return self.pk is None

    def tables(self) -> Set[db.TableRef]:
        """Provides both joined tables."""
        return set([self.first, self.second])

    def __hash__(self):
        return hash(self.predicate)

    def __eq__(self, other: object):
        if not isinstance(other, _TableJoin):
            return False
        return self.predicate == other.predicate

    def __repr__(self):
        return str(self)

    def __str__(self):
        if self.is_pk_fk_join():
            return f"JOIN PK {self.pk}, FK {self.fk} ON {self.predicate}"
        return f"JOIN {self.first}, {self.second} ON {self.predicate}"


class _JoinGraph:
    """The join graph provides a nice interface for querying information about the joins we have to execute."""
    pass


class _JoinTree:
    """The join tree contains a semi-ordered sequence of joins.

    The ordering results from the different levels of the tree, but joins within the same level and parent are
    unordered.
    """
    pass


def _build_predicate_map(query: mosp.MospQuery
                         ) -> Dict[db.TableRef, List[Union[mosp.MospPredicate, mosp.CompoundMospFilterPredicate]]]:
    """The predicate map is a dictionary which maps each table to the filter predicates that apply to this table."""
    all_filter_predicates = [pred for pred in query.predicates() if not pred.is_join_predicate()]
    predicate_map = collections.defaultdict(list)
    for filter_pred in all_filter_predicates:
        predicate_map[filter_pred.parse_left_attribute().table].append(filter_pred)
    return predicate_map


def _estimate_filtered_cardinalities(predicate_map: dict, *,
                                     dbs: db.DBSchema = db.DBSchema.get_instance()) -> Dict[db.TableRef, int]:
    """Fetches the PG estimates for all tables in the predicate_map according to their associated filters."""
    return {table: predicates.estimate_result_rows(dbs=dbs) for table, predicates in predicate_map.items()}


def _detect_joins(query: mosp.MospQuery, *, dbs: db.DBSchema = db.DBSchema.get_instance()) -> Set[_TableJoin]:
    """Finds all joins in the query and determines whether they are an n:m join, or a PK/FK join."""
    join_predicates = [pred for pred in query.predicates() if pred.is_join_predicate()]
    joins = set([_TableJoin(pred) for pred in join_predicates])
    for join in joins:
        pk, fk = None, None
        if dbs.is_primary_key(join.first_attr):
            pk = join.first
        elif dbs.is_primary_key(join.second_attr):
            pk = join.second

        if dbs.has_secondary_idx_on(join.first_attr):
            fk = join.first
        elif dbs.has_secondary_idx_on(join.second_attr):
            fk = join.second

        if pk and fk:
            join.mark_pk_fk_join(pk, fk)


def _calculate_join_order(query: mosp.MospQuery, *,
                          cardinality_estimator: JoinCardinalityEstimator = DefaultUESCardinalityEstimator(),
                          subquery_generator: SubqueryGenerationStrategy = DefensiveSubqueryGeneration(),
                          dbs: db.DBSchema = db.DBSchema.get_instance()
                          ) -> List[_TableJoin]:
    join_order = []  # the resulting join order will be stored here

    # the cardinality estimates contain Postgres estimates on how many tuples will be contained in each base table,
    # after applying all filter predicates on that table
    base_table_cardinality_estimates = _estimate_filtered_cardinalities(_build_predicate_map(query), dbs=dbs)

    all_joins = _detect_joins(query, dbs=dbs)  # all joins specified in the query

    # all tables that participate in at least one n:m join
    n_m_join_tables = [join for join in all_joins if join.is_n_m_join()]  # FIXME

    # all tables that only participate in PK/FK joins
    pk_fk_join_tables = [join for join in all_joins if join.is_pk_fk_join()]  # FIXME

    # The UES algorithm will iteratively expand the join order one join at a time. In each iteration, the best n:m
    # join (after/before being potentially filtered via PK/FK joins) is selected. In the working set, we keep track of
    # all the tables that we still have to join.
    n_m_join_tables_working_set = list(n_m_join_tables)
    while n_m_join_tables_working_set:
        # continue as long as their are still tables to be joined in our working set
        pass

    return join_order


def optimize_query(query: mosp.MospQuery, *,
                   cardinality_estimation: str = "basic",
                   subquery_generation: str = "defensive",
                   dbs: db.DBSchema = db.DBSchema.get_instance()) -> mosp.MospQuery:
    if cardinality_estimation == "basic":
        cardinality_estimator = DefaultUESCardinalityEstimator(query)
    elif cardinality_estimation == "advanced":
        # TODO: implementation
        pass
    else:
        raise ValueError("Unknown cardinality estimation strategy: '{}'".format(cardinality_estimation))

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
