import enum
from typing import Dict, List

from transform import db, mosp


class QueryNode(enum.Enum):
    SeqScan = "SeqScan"
    IndexScan = "IndexScan"
    IndexOnlyScan = "IndexOnlyScan"
    NestLoop = "NestLoop"
    HashJoin = "HashJoin"

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

    def force_nestloop(self, join: mosp.MospJoin):
        jid = _join_id(join)
        self.join_hints[jid] = QueryNode.NestLoop
        self.join_contents[jid] = join

    def force_hashjoin(self, join: mosp.MospJoin):
        jid = _join_id(join)
        self.join_hints[jid] = QueryNode.HashJoin
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


def idxnlj_subqueries(query: mosp.MospQuery, *, nestloop="first", idxscan="fk"):
    if idxscan not in ["pk", "fk"]:
        raise ValueError("idxscan must be either 'pk' or 'fk'")
    if nestloop not in ["first", "all"]:
        raise ValueError("nestloop must be either 'first' or 'all'")

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
