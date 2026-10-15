"""Tests for `postbound.duckdb` -- the DuckDB backend, run against a real in-memory DuckDB (via quacklab).

All tests here are tier 1 (`embedded`): they need a real DuckDB engine, but no server or database file, because
the behaviours under test are about whether DuckDB *accepts* the SQL that PostBOUND generates. A double would
share the backend's assumptions about DuckDB's dialect and could only confirm a bug, not catch it. The synthetic
schema is created per test, so nothing is downloaded or written to disk.

`PreciseStatistics.histogram` is run here as well, because its SQL and the shape of the result set (in particular
how NULLs are grouped and sorted) are decided by the engine. Its decision logic is covered offline in
`tests/unit/test_db_stats.py`, which scripts the result sets observed here.

The pure formatting part of the EXPLAIN regression (the *duckdb* flavor of `format_quick`) is covered offline in
`tests/unit/test_qal_formatter.py`; the parsing of EXPLAIN output and the scan-to-table binding in
`tests/unit/test_duckdb.py`.
"""

from __future__ import annotations

from collections.abc import Iterator
from pathlib import Path
from typing import Any, cast

import pytest

from postbound import ColumnReference, ScanOperator, TableReference, parser, transform
from postbound.db import ForeignKeyRef, Histogram, PreciseStatistics, PrimaryKeyRef
from postbound.duckdb import DuckDBDatabase

pytestmark = pytest.mark.embedded

# -- fixtures -----------------------------------------------------------------------------------------------


def parse(sql: str):
    return parser.parse_query(sql, bind_columns=False)


FILTER_QUERY = parse("SELECT * FROM r WHERE r.a = 1")


@pytest.fixture
def duckdb_instance() -> Iterator[DuckDBDatabase]:
    """An in-memory DuckDB with a tiny table `r(a, b)`. The constructor does not register it on the pool."""
    instance = DuckDBDatabase(Path(":memory:"))
    instance.execute_query("CREATE TABLE r (a INTEGER, b INTEGER)")
    instance.execute_query("INSERT INTO r VALUES (1, 10), (2, 20), (3, 30)")
    yield instance
    instance.close()


def create_column(instance: DuckDBDatabase, values: list[int | None]) -> ColumnReference:
    """Fills a fresh table `h(v)` with `values` and returns the reference to its column."""
    instance.execute_query("CREATE TABLE h (v INTEGER)")
    rows = ", ".join("(NULL)" if value is None else f"({value})" for value in values)
    instance.execute_query(f"INSERT INTO h VALUES {rows}")
    return ColumnReference("v", TableReference("h"))


# -- PreciseStatistics.histogram ----------------------------------------------------------------------------


def test_precise_histogram_builds_equi_depth_buckets_on_duckdb(duckdb_instance: DuckDBDatabase) -> None:
    column = create_column(duckdb_instance, [1, 2, 3, 4, 5, 6, 7, 8])

    hist = PreciseStatistics(duckdb_instance).histogram(column, n_bins=4)

    assert hist == Histogram([2, 4, 6, 8], [2, 2, 2, 2], lower=1, bucket_interpolation="approx-uni")


# -- DuckDBSchema ---------------------------------------------------------------------------------------------

SCHEMA_DDL = [
    "CREATE TABLE p (a INTEGER, b INTEGER, c INTEGER, PRIMARY KEY (b, a))",
    "CREATE TABLE s (id INTEGER PRIMARY KEY, u INTEGER UNIQUE, x INTEGER, y INTEGER, UNIQUE (x, y))",
    "CREATE TABLE ch (a INTEGER, b INTEGER, k INTEGER, FOREIGN KEY (b, a) REFERENCES p (b, a), "
    "FOREIGN KEY (k) REFERENCES s (id))",
    "CREATE INDEX s_x_idx ON s (x)",
    "CREATE INDEX s_yx_idx ON s (y, x)",
]
P, S, CH = TableReference("p"), TableReference("s"), TableReference("ch")


@pytest.fixture
def schema_instance(duckdb_instance: DuckDBDatabase) -> DuckDBDatabase:
    for statement in SCHEMA_DDL:
        duckdb_instance.execute_query(statement)
    return duckdb_instance


def test_duckdb_schema_lists_tables_and_columns_in_order(schema_instance: DuckDBDatabase) -> None:
    schema = schema_instance.schema()

    assert schema.tables() == {TableReference("r"), P, S, CH}
    assert [col.name for col in schema.columns(S)] == ["id", "u", "x", "y"]


def test_duckdb_schema_primary_key_of_follows_the_key_definition_order(schema_instance: DuckDBDatabase) -> None:
    """The key is declared as ``(b, a)`` while the table defines ``a`` before ``b``."""
    schema = schema_instance.schema()

    pk = schema.primary_key_of(P)

    assert pk is not None and [col.name for col in pk.columns] == ["b", "a"]
    assert schema.primary_key_of(S) == PrimaryKeyRef([ColumnReference("id", S)])
    assert schema.primary_key_of(CH) is None


def test_duckdb_schema_is_primary_key_for_composite_keys(schema_instance: DuckDBDatabase) -> None:
    schema = schema_instance.schema()
    a, b = ColumnReference("a", P), ColumnReference("b", P)

    assert schema.is_primary_key([a, b]) is True
    assert schema.is_primary_key(a) is False
    assert schema.is_primary_key(ColumnReference("id", S)) is True


def test_duckdb_schema_is_unique_for_single_and_composite_constraints(schema_instance: DuckDBDatabase) -> None:
    schema = schema_instance.schema()
    x, y = ColumnReference("x", S), ColumnReference("y", S)

    assert schema.is_unique(ColumnReference("u", S)) is True
    assert schema.is_unique([y, x]) is True
    assert schema.is_unique(x) is False


def test_duckdb_schema_datatype_and_nullability(schema_instance: DuckDBDatabase) -> None:
    schema = schema_instance.schema()

    assert schema.datatype(ColumnReference("x", S)) == "integer"
    assert schema.is_nullable(ColumnReference("x", S)) is True
    assert schema.is_nullable(ColumnReference("id", S)) is False


# -- regression tests --------------------------------------------------------------------------------------


def test_hint_service_formats_explain_queries_without_postgres_only_options(duckdb_instance: DuckDBDatabase) -> None:
    """Regression guard for the v0.22.1 DuckDB EXPLAIN fix: `DuckDBHintService.format_query` used to call
    `format_quick(..., flavor="postgres")`. Since v0.22.0 that flavor writes `EXPLAIN (SETTINGS, SUMMARY,
    VERBOSE, FORMAT JSON)`, which DuckDB does not support, so the backend could not run any EXPLAIN query.
    """
    explain_query = transform.as_explain(FILTER_QUERY)

    formatted = duckdb_instance.hinting().format_query(explain_query)

    header = formatted.splitlines()[0]
    assert header == "EXPLAIN (FORMAT JSON)"


def test_query_plan_of_a_parsed_query_runs_on_duckdb(duckdb_instance: DuckDBDatabase) -> None:
    """Regression guard for the v0.22.1 DuckDB EXPLAIN fix: `DuckDBOptimizer.query_plan` wraps a `SqlQuery` in an
    EXPLAIN and formats it through the hint service, which used the Postgres flavor and thereby emitted the
    Postgres-only `SETTINGS`/`SUMMARY`/`VERBOSE` options. DuckDB rejected the statement (`NotImplementedException: Unimplemented explain type: settings`)
    before planning even started. (Plain-string queries took a separate hand-written `EXPLAIN (FORMAT JSON)` path
    and were unaffected, which is why this test passes a parsed query.)
    """
    plan = duckdb_instance.optimizer().query_plan(FILTER_QUERY)

    assert plan.operator == ScanOperator.SequentialScan


def test_execute_query_runs_a_parsed_explain_query_on_duckdb(duckdb_instance: DuckDBDatabase) -> None:
    """Regression guard for the v0.22.1 DuckDB EXPLAIN fix, on the `DuckDBDatabase.execute_query` path: it formats
    every `SqlQuery` through the hint service as well, so executing an `EXPLAIN` / `EXPLAIN ANALYZE` query object
    directly failed with the same `NotImplementedException` as `query_plan`.
    """
    explain_queries = [transform.as_explain(FILTER_QUERY), transform.as_explain_analyze(FILTER_QUERY)]

    result_sets = [duckdb_instance.execute_query(query, raw=True) for query in explain_queries]

    assert [result_set[0][0] for result_set in result_sets] == ["physical_plan", "analyzed_plan"]


def test_analyze_plan_returns_the_measured_plan(duckdb_instance: DuckDBDatabase) -> None:
    """Regression guard for the v0.22.1 DuckDB `analyze_plan` fix: `DuckDBOptimizer.analyze_plan` passed `parsed[0]`
    to `parse_duckdb_plan`, assuming the JSON *list* that plain `EXPLAIN (FORMAT JSON)` returns. With ANALYZE,
    DuckDB returns a single JSON *object* (a profiling root with the plan below `children`), so indexing it with `0`
    raised a `KeyError` and no query could be analyzed.
    """
    plan = duckdb_instance.optimizer().analyze_plan(FILTER_QUERY)

    assert plan.operator == ScanOperator.SequentialScan
    assert plan.actual_cardinality == 1


def test_query_plan_binds_scan_nodes_to_their_tables(duckdb_instance: DuckDBDatabase) -> None:
    """Regression guard for the v0.22.1 DuckDB table-binding fix, end to end: `_bind_node_to_table` compared the
    `Table` entry of a scan node with `TableReference.full_name` by plain string equality, but DuckDB reports fully
    qualified names (`memory.main.r` here) while the query references just `r`. Every scan node stayed unbound, so
    `query_plan` returned plans without any tables. The offline tests in `tests/unit/test_duckdb.py` pin the captured
    output; this one guards against DuckDB changing the reported name format underneath them.
    """
    plan = duckdb_instance.optimizer().query_plan(FILTER_QUERY)

    assert plan.base_table == TableReference("r")
    assert plan.tables() == {TableReference("r")}


def test_precise_histogram_excludes_nulls_on_duckdb(duckdb_instance: DuckDBDatabase) -> None:
    """Regression guard for the unreleased histogram fixes: DuckDB returns the NULL group of the histogram's GROUP BY
    query last as ``(None, 0)``. With fewer rows than bins, `_infer_histogram_bounds` turned it into the upper bound of
    the histogram, and every estimate at or above the largest value raised a `TypeError` comparing with *None*.
    """
    column = create_column(duckdb_instance, [1, 1, 2, 3, None, None])

    hist = PreciseStatistics(duckdb_instance).histogram(column)

    assert list(hist) == [(1, 2), (2, 1), (3, 1)]
    assert hist.frequency_below(5) == 4


def test_precise_histogram_of_a_single_distinct_value_on_duckdb(duckdb_instance: DuckDBDatabase) -> None:
    """Regression guard for the unreleased histogram fixes: the one-row result set was simplified to a plain tuple
    before the bounds were inferred, which crashed. See `test_histogram_of_a_column_with_a_single_distinct_value` in
    `tests/unit/test_db_stats.py`.
    """
    column = create_column(duckdb_instance, [7, 7])

    hist = PreciseStatistics(duckdb_instance).histogram(column)

    assert list(hist) == [(7, 2)]


# -- regression tests --------------------------------------------------------------------------------------------


def test_duckdb_schema_all_foreign_keys_on_resolves_composite_and_simple_keys(schema_instance: DuckDBDatabase) -> None:
    """Regression guard for the composite key rework: the generic implementation relied on Postgres semantics of
    ``constraint_column_usage`` and reported every key of ``ch`` as referencing ``ch`` itself. The foreign key is
    declared as ``(b, a)`` while ``p`` defines ``a`` before ``b``, so this also checks the column pairing.
    """
    fks = schema_instance.schema().all_foreign_keys_on(CH)

    assert {fk.referenced_table for fk in fks} == {P, S}
    composite = next(fk for fk in fks if fk.referenced_table == P)
    assert dict(composite.mapping()) == {
        ColumnReference("b", CH): ColumnReference("b", P),
        ColumnReference("a", CH): ColumnReference("a", P),
    }
    simple = next(fk for fk in fks if fk.referenced_table == S)
    assert simple == ForeignKeyRef([ColumnReference("k", CH)], [ColumnReference("id", S)])


def test_duckdb_schema_foreign_keys_on_selects_by_the_exact_column_set(schema_instance: DuckDBDatabase) -> None:
    schema = schema_instance.schema()
    a, b, k = (ColumnReference(name, CH) for name in "abk")

    assert {fk.referenced_table for fk in schema.foreign_keys_on([a, b])} == {P}
    assert {fk.referenced_table for fk in schema.foreign_keys_on(k)} == {S}
    assert schema.foreign_keys_on(a) == set()
    assert schema.foreign_keys_on(ColumnReference("k", TableReference("ch", "c"))) == schema.foreign_keys_on(k)


def test_duckdb_schema_has_no_foreign_keys_for_tables_without_any(schema_instance: DuckDBDatabase) -> None:
    assert schema_instance.schema().all_foreign_keys_on(S) == set()


def test_duckdb_schema_as_graph_contains_composite_foreign_keys(schema_instance: DuckDBDatabase) -> None:
    graph = schema_instance.schema().as_graph()

    assert {tuple(sorted(col.name for col in fk.fk_cols)) for fk in graph.edges[CH, P]["foreign_keys"]} == {("a", "b")}
    assert graph.has_edge(CH, S)


def test_duckdb_schema_join_equivalence_follows_simple_foreign_keys_only(schema_instance: DuckDBDatabase) -> None:
    classes = schema_instance.schema().join_equivalence_keys()

    assert classes[ColumnReference("k", CH)] == {ColumnReference("k", CH), ColumnReference("id", S)}
    assert ColumnReference("a", CH) not in classes


def test_duckdb_schema_all_indexes_on_lists_constraint_backed_and_explicit_indexes(
    schema_instance: DuckDBDatabase,
) -> None:
    """Regression guard for the composite key rework: the statement filtered the table functions by columns that do
    not exist (``catalog_name``, and a ``constraint_type`` on ``duckdb_indexes()``) and was rejected by DuckDB.
    """
    indexes = schema_instance.schema().all_indexes_on(S)

    assert indexes["s_id_pkey"] == {ColumnReference("id", S)}
    assert indexes["s_u_key"] == {ColumnReference("u", S)}
    assert indexes["s_x_y_key"] == {ColumnReference("x", S), ColumnReference("y", S)}
    assert indexes["s_x_idx"] == {ColumnReference("x", S)}
    assert indexes["s_yx_idx"] == {ColumnReference("x", S), ColumnReference("y", S)}


def test_duckdb_schema_all_indexes_on_does_not_report_not_null_constraints(schema_instance: DuckDBDatabase) -> None:
    """Regression guard for the composite key rework: ``duckdb_constraints()`` also lists NOT NULL and CHECK
    constraints, which made every NOT NULL column look indexed.
    """
    schema = schema_instance.schema()

    assert set(schema.all_indexes_on(P)) == {"p_b_a_pkey"}
    assert schema.has_secondary_index(ColumnReference("a", P)) is False
    assert schema.has_secondary_index(ColumnReference("x", S)) is True


def test_duckdb_schema_all_indexes_on_lists_foreign_keys_on_their_own_columns(schema_instance: DuckDBDatabase) -> None:
    indexes = schema_instance.schema().all_indexes_on(CH)

    assert sorted(sorted(col.name for col in cols) for cols in indexes.values()) == [["a", "b"], ["k"]]


def test_duckdb_schema_indexes_on_matches_composite_indexes(schema_instance: DuckDBDatabase) -> None:
    schema = schema_instance.schema()
    x, y = ColumnReference("x", S), ColumnReference("y", S)

    assert schema.indexes_on(x) == {"s_x_idx"}
    assert schema.indexes_on([y, x]) == {"s_yx_idx", "s_x_y_key"}  # the explicit index and the UNIQUE constraint


def test_duckdb_schema_table_info_combines_keys_and_indexes(schema_instance: DuckDBDatabase) -> None:
    info = schema_instance.schema()[CH]

    assert info.primary_key is None
    assert {fk.referenced_table for fk in info.outgoing_foreign_keys} == {P, S}
    assert schema_instance.schema()[P].incoming_foreign_keys == {
        fk for fk in info.outgoing_foreign_keys if fk.referenced_table == P
    }
    assert info["k"].indexed is True  # foreign keys count as indexed, like in the generic implementation


def test_duckdb_database_describe_lists_keys_and_indexes_of_every_table(schema_instance: DuckDBDatabase) -> None:
    """Regression guard for the composite key rework: `describe()` failed with the binder errors of the schema lookups."""
    description = schema_instance.describe()

    schema_info = cast(dict[str, Any], description)["schema_info"]
    tables = {entry["table"]: entry for entry in schema_info}
    assert set(tables) == {"r", "p", "s", "ch"}
    assert tables["s"]["primary_key"] is not None
    assert tables["ch"]["foreign_keys"]
    assert tables["s"]["indexes"]
