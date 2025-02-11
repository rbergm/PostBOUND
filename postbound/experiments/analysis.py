"""Provides a collection of utilities related to query optimization."""

from __future__ import annotations

from typing import Optional, Any


from .. import db, qal, util
from .._qep import QueryPlan


def possible_plans_bound(query: qal.SqlQuery, *,
                         join_operators: set[str] = {"nested-loop join", "hash join", "sort-merge join"},
                         scan_operators: set[str] = {"sequential scan", "index scan"}) -> int:
    """Computes a quick upper bound on the maximum number of possible query execution plans for a given query.

    This upper bound is a very coarse one, based on three assumptions:

    1. any join sequence (even involving cross-products) of any form (i.e. right-deep, bushy, ...) is allowed
    2. the choice of scan operators and join operators can be varied freely
    3. each table can be scanned using arbitrary operators

    The number of real-world query execution plans will typically be much smaller, because cross-products are only
    used if really necessary and the selected join operator influences the scan operators and vice-versa.

    Parameters
    ----------
    query : qal.SqlQuery
        The query for which the bound should be computed
    join_operators : set[str], optional
        The allowed join operators, by default {"nested-loop join", "hash join", "sort-merge join"}
    scan_operators : set[str], optional
        The allowed scan operators, by default {"sequential scan", "index scan"}

    Returns
    -------
    int
        An upper bound on the number of possible query execution plans
    """
    n_tables = len(query.tables())

    join_orders = util.stats.catalan_number(n_tables)
    joins = (n_tables - 1) * len(join_operators)
    scans = n_tables * len(scan_operators)

    return join_orders * joins * scans


def actual_plan_cost(query: qal.SqlQuery, analyze_plan: QueryPlan, *,
                     database: Optional[db.Database] = None) -> float:
    """Utility to compute the true cost of a query plan based on the actual cardinalities.

    Parameters
    ----------
    query : qal.SqlQuery
        The query to analyze
    analyze_plan : QueryPlan
        The executed query which also contains the true cardinalities
    database : Optional[db.Database], optional
        The database providing the cost model. If omitted, the database is inferred from the database pool.

    Returns
    -------
    float
        _description_
    """
    if not analyze_plan.is_analyze():
        raise ValueError("The provided plan is not an ANALYZE plan")
    database = database if database is not None else db.DatabasePool().get_instance()
    hinted_query = database.hinting().generate_hints(query, analyze_plan.as_optimized_plan())
    return database.optimizer().cost_estimate(hinted_query)


def text_diff(left: str, right: str, *, sep: str = " | ") -> str:
    """Merges two text snippets to allow for a comparison on a per-line basis.

    The two snippets are split into their individual lines and then merged back together.

    Parameters
    ----------
    left : str
        The text snippet to display on the left-hand side.
    right : str
        The text snippet to display on the right-hand side.
    sep : str, optional
        The separator to use between the left and right text snippets, by default `` | ``.

    Returns
    -------
    str
        The combined text snippet
    """
    left_lines = left.splitlines()
    right_lines = right.splitlines()

    max_left_len = max(len(line) for line in left_lines)
    left_lines_padded = [line.ljust(max_left_len) for line in left_lines]

    merged_lines = [f"{left_line}{sep}{right_line}" for left_line, right_line in zip(left_lines_padded, right_lines)]
    return "\n".join(merged_lines)


def star_query_cardinality(query: qal.SqlQuery, fact_table_pk_column: qal.ColumnReference, *,
                           database: Optional[db.Database] = None, verbose: bool = False) -> int:
    """Utility function to manually compute the cardinality of a star query.

    This function is intended for situations where the database is unable to compute the cardinality because the intermediates
    involved in the query become to large or the query plans are simply too bad. It operates by manually computing the number
    of output tuples for each of the entries in the fact table by sequentially joining the fact table with each dimension
    table.

    Parameters
    ----------
    query : qal.SqlQuery
        The query to compute the cardinality for. This is assumed to be a **SELECT \\*** query and the actual **SELECT** clause
        is ignored completely.
    fact_table_pk_column : qal.ColumnReference
        The fact table's primary key column. All dimension tables must perform an equi-join on this column.
    database : Optional[db.Database], optional
        The actual database. If this is omitted, the current database from the database pool is used.
    verbose : bool, optional
        Whether progress information should be printed during the computation. If this is enabled, the function will report
        every 1000th value processed.

    Returns
    -------
    int
        The cardinality (i.e. number of output tuples) of the query

    Warnings
    --------
    Currently, this function works well for simple SPJ-based queries, more complicated features might lead to wrong results.
    Similarly, only pure star queries are supported, i.e. there has to be one central fact table and each dimension table
    performs exactly one equi-join with the fact table's primary key. There may not be additional joins on the dimension
    tables. If such additional dimension joins exist, they have to be pre-processed (e.g. by introducing materialized views)
    and the query has to be rewritten to operate on the views instead.
    It is the user's responsibility to ensure that the query is well-formed in these regards.
    """
    logger = util.make_logger(verbose, prefix=util.timestamp)
    database = db.DatabasePool().get_instance().current_database() if database is None else database
    fact_table = (fact_table_pk_column.table if fact_table_pk_column.is_bound()
                  else database.schema().lookup_column(fact_table_pk_column, query.tables()))
    if fact_table is None:
        raise ValueError(f"Cannot infer fact table from column '{fact_table_pk_column}'")
    fact_table_pk_column = fact_table_pk_column.bind_to(fact_table)

    id_vals_query = qal.parse_query(f"""
                                    SELECT {fact_table_pk_column}, COUNT(*) AS card
                                    FROM {fact_table}
                                    GROUP BY {fact_table_pk_column}""")
    if query.predicates().filters_for(fact_table):
        filter_clause = qal.Where(query.predicates().filters_for(fact_table))
        id_vals_query = qal.transform.add_clause(id_vals_query, filter_clause)
    id_vals: list[tuple[Any, int]] = database.execute_query(id_vals_query)

    base_query_fragments: dict[qal.AbstractPredicate, qal.SqlQuery] = {}
    for join_pred in query.predicates().joins_for(fact_table):
        join_partner = join_pred.join_partners_of(fact_table)
        if not len(join_partner) == 1:
            raise ValueError("Currently only singular joins are supported")

        partner_table: qal.ColumnReference = util.simplify(join_partner).table
        query_fragment = qal.transform.extract_query_fragment(query, [fact_table, partner_table])
        base_query_fragments[join_pred] = qal.transform.as_count_star_query(query_fragment)

    total_cardinality = 0
    total_ids = len(id_vals)
    for value_idx, (id_value, current_card) in enumerate(id_vals):
        if value_idx % 1000 == 0:
            logger("--", value_idx, "out of", total_ids, "values processed")

        id_filter = qal.BinaryPredicate.equal(qal.ColumnExpression(fact_table_pk_column), qal.StaticValueExpression(id_value))

        for join_pred, base_query in base_query_fragments.items():
            if current_card == 0:
                break

            expanded_predicate = qal.CompoundPredicate.create_and([base_query.where_clause.predicate, id_filter])
            expanded_where_clause = qal.Where(expanded_predicate)

            dimension_query = qal.transform.replace_clause(base_query, expanded_where_clause)
            dimension_card = database.execute_query(dimension_query)

            current_card *= dimension_card

        total_cardinality += current_card

    return total_cardinality
