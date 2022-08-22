
import enum
import math
import warnings
from typing import Dict, FrozenSet, List

from transform import db, mosp, util


class QueryNode(enum.Enum):
    SeqScan = "SeqScan"
    IndexScan = "IndexScan"
    IndexOnlyScan = "IndexOnlyScan"
    NestLoop = "NestLoop"
    HashJoin = "HashJoin"
    SortMergeJoin = "MergeJoin"

    def is_join(self) -> bool:
        return self == QueryNode.NestLoop or self == QueryNode.HashJoin

    def is_scan(self) -> bool:
        return self in [QueryNode.SeqScan, QueryNode.IndexScan, QueryNode.IndexOnlyScan]

    def __str__(self) -> str:
        return self.value


def _join_id(join: mosp.MospJoin) -> int:
    return hash(frozenset(join.collect_tables()))


class HintedMospQuery:
    """A HintedMospQuery augments SQL queries by PostgreSQL query hints. This assumes a fixed join order."""
    def __init__(self, query: mosp.MospQuery):
        self.query: mosp.MospQuery = query

        # Build the query join paths. A join path describes which joins have to be executed before a certain join
        # can run. This assumes a fixed join order. For convenience, each join path also contains the join in question
        # itself. I.e., The join path for joining table C could look like A B C, indicating that the join between
        # A and B has to executed first.
        self.join_paths: Dict[int, List[db.TableRef]] = dict()
        base_table = query.base_table()
        curr_join_path = [base_table]
        for join in query.joins():
            curr_join_path.extend(join.collect_tables())
            self.join_paths[_join_id(join)] = list(curr_join_path)  # copy the join path to prevent unintended updates

            # FIXME: this algorithm currently only works for 1 layer of subqueries
            # subqueries with subqueries are unsupported. A recursive algorithm should solve this problem quite nicely.
            if join.is_subquery():
                sq_base_table = join.base_table()
                sq_join_path = [sq_base_table]
                for sq_join in join.subquery.joins():
                    sq_join_path.extend(sq_join.collect_tables())
                    self.join_paths[_join_id(sq_join)] = list(sq_join_path)  # copy the join path once again

        self.scan_hints: Dict[db.TableRef, QueryNode] = dict()
        self.join_hints: Dict[int, QueryNode] = dict()
        self.cardinality_bounds: Dict[int, int] = dict()
        self.join_contents: Dict[int, mosp.MospJoin] = dict()
        self.bounds_stats: Dict[FrozenSet[db.TableRef], Dict[str, int]] = dict()

    def force_nestloop(self, join: mosp.MospJoin):
        jid = _join_id(join)
        self.join_hints[jid] = QueryNode.NestLoop
        self.join_contents[jid] = join

    def force_hashjoin(self, join: mosp.MospJoin):
        jid = _join_id(join)
        self.join_hints[jid] = QueryNode.HashJoin
        self.join_contents[jid] = join

    def force_mergejoin(self, join: mosp.MospJoin) -> None:
        jid = _join_id(join)
        self.join_hints[jid] = QueryNode.SortMergeJoin
        self.join_contents[jid] = join

    def force_seqscan(self, table: db.TableRef):
        self.scan_hints[table] = QueryNode.SeqScan

    def force_idxscan(self, table: db.TableRef):
        # we can use an IndexOnlyScan here, b/c IndexOnlyScan falls back to IndexScan automatically if necessary
        self.scan_hints[table] = QueryNode.IndexOnlyScan

    def set_upperbound(self, join: mosp.MospJoin, nrows: int):
        jid = _join_id(join)
        self.cardinality_bounds[jid] = nrows
        self.join_contents[jid] = join

    def merge_with(self, other_query: "HintedMospQuery") -> None:
        self.scan_hints = util.dict_merge(self.scan_hints, other_query.scan_hints)
        self.join_hints = util.dict_merge(self.join_hints, other_query.join_hints)
        self.cardinality_bounds = util.dict_merge(self.cardinality_bounds, other_query.cardinality_bounds)
        self.join_contents = util.dict_merge(self.join_contents, other_query.join_contents)

    def store_bounds_stats(self, join: FrozenSet[db.TableRef], bounds: Dict[str, int]) -> None:
        self.bounds_stats[join] = bounds

    def generate_sqlcomment(self, *, strip_empty: bool = False) -> str:
        if strip_empty and not self.scan_hints and not self.join_hints and not self.cardinality_bounds:
            return ""

        scan_hints_stringified = "\n".join(self._scan_hint_to_str(tab) for tab in self.scan_hints.keys())
        join_hints_stringified = "\n".join(self._join_hint_to_str(join_id) for join_id in self.join_hints.keys())
        cardinality_bounds_stringified = "\n".join(self._cardinality_bound_to_str(join_id)
                                                   for join_id in self.cardinality_bounds.keys())

        return "\n".join(s for s in ["/*+",
                                     scan_hints_stringified, join_hints_stringified, cardinality_bounds_stringified,
                                     "*/"] if s)

    def generate_query(self, *, strip_empty: bool = False) -> str:
        hint = self.generate_sqlcomment(strip_empty=strip_empty)
        return "\n".join(hint, self.query.text() + ";")

    def _scan_hint_to_str(self, base_table: db.TableRef) -> str:
        operator = self.scan_hints[base_table]
        return f"{operator.value}({base_table.qualifier()})"

    def _join_hint_to_str(self, join_id: int) -> str:
        full_join_path = self._join_path_to_str(join_id)
        return f"{self.join_hints[join_id].value}({full_join_path})"

    def _cardinality_bound_to_str(self, join_id: int) -> str:
        full_join_path = self._join_path_to_str(join_id)
        n_rows = self.cardinality_bounds[join_id]
        return f"Rows({full_join_path} #{n_rows})"

    def _join_path_to_str(self, join_id: int) -> str:
        return " ".join(tab.qualifier() for tab in self.join_paths[join_id])

    def __repr__(self) -> str:
        return str(self)

    def __str__(self) -> str:
        return self.generate_sqlcomment()


def idxnlj_subqueries(query: mosp.MospQuery, *, nestloop="first", idxscan="fk") -> HintedMospQuery:
    if idxscan not in ["pk", "fk"]:
        raise ValueError("idxscan must be either 'pk' or 'fk', not '{}'".format(idxscan))
    if nestloop not in ["first", "all"]:
        raise ValueError("nestloop must be either 'first' or 'all', not '{}'".format(nestloop))

    hinted_query = HintedMospQuery(query)
    for sq in [sq.subquery for sq in query.subqueries()]:
        fk_table = sq.base_table()
        if idxscan == "fk":
            hinted_query.force_idxscan(fk_table)

        if nestloop == "first":
            first_pk_join = sq.joins()[0]
            hinted_query.force_nestloop(first_pk_join)

            if idxscan == "pk":
                pk_table = first_pk_join.base_table()
                hinted_query.force_idxscan(pk_table)
        elif nestloop == "all":
            for join_idx, join in enumerate(sq.joins()):
                hinted_query.force_nestloop(join)

                if idxscan == "pk" or join_idx > 0:
                    pk_table = join.base_table()
                    hinted_query.force_idxscan(pk_table)
    return hinted_query


def bound_hints(query: mosp.MospQuery, bounds_data: Dict[FrozenSet[db.TableRef], int]) -> HintedMospQuery:
    hinted_query = HintedMospQuery(query)
    visited_tables = [query.base_table()]
    for join in query.joins():
        if join.is_subquery():
            subquery_hints = bound_hints(join.subquery, bounds_data)
            hinted_query.merge_with(subquery_hints)
        visited_tables.extend(join.collect_tables())
        tables_key = frozenset(visited_tables)
        if tables_key in bounds_data:
            hinted_query.set_upperbound(join, bounds_data[tables_key])
    return hinted_query


def operator_hints(query: mosp.MospQuery, bounds_data: Dict[FrozenSet[db.TableRef], int]) -> HintedMospQuery:
    hinted_query = HintedMospQuery(query)
    visited_tables = [query.base_table()]
    for join in query.joins():
        if join.is_subquery():
            subquery_hints = operator_hints(join.subquery, bounds_data)
            hinted_query.merge_with(subquery_hints)
        visited_tables.extend(join.collect_tables())
        tables_key = frozenset(visited_tables)
        upper_bound = bounds_data.get(tables_key, None)
        if upper_bound:
            nlj_bound = upper_bound ** 2
            hashjoin_bound = 2 * upper_bound
            mergejoin_bound = upper_bound * math.log(upper_bound)

            if nlj_bound <= hashjoin_bound and nlj_bound <= mergejoin_bound:
                warnings.warn("Choosing NLJ")
                hinted_query.force_nestloop(join)
            elif hashjoin_bound <= nlj_bound and hashjoin_bound <= mergejoin_bound:
                hinted_query.force_hashjoin(join)
            elif mergejoin_bound <= nlj_bound and mergejoin_bound <= hashjoin_bound:
                warnings.warn("Choosing MergeJoin")
                hinted_query.force_mergejoin(join)
            else:
                raise util.StateError("The universe dissolves..")

            hinted_query.store_bounds_stats(tables_key, {"ues": upper_bound,
                                                         "nlj": nlj_bound,
                                                         "hashjoin": hashjoin_bound,
                                                         "mergejoin": mergejoin_bound})

    return hinted_query
