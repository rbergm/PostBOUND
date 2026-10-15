"""Tests for the test doubles themselves.

The doubles are load-bearing: once real tests depend on them, a silently wrong double produces confidently
wrong results everywhere. These tests pin the two properties that matter most -- that the doubles refuse
what they were not told about, and that `StaticSchema` really does run the production `DatabaseSchema`
composition rather than replacing it.
"""

from __future__ import annotations

import pytest

from postbound import Cardinality, UnboundColumnError
from postbound._core import ColumnReference, TableReference
from postbound.db import Database, DatabaseSchema, ForeignKeyRef, PrimaryKeyRef
from tests.doubles import (
    FakeDatabase,
    FakeStatistics,
    ScriptedCursor,
    StaticSchema,
    UnexpectedQueryError,
    column_spec,
    foreign_key_spec,
)

TITLE = TableReference("title")
MOVIE_INFO = TableReference("movie_info")

SCHEMA_SPEC = {
    "title": [
        column_spec("id", "integer", nullable=False, primary_key=True),
        column_spec("title", "text"),
        column_spec("production_year", "integer", indexed=True),
    ],
    "movie_info": [
        column_spec("id", "integer", nullable=False, primary_key=True),
        column_spec("movie_id", "integer", references="title.id"),
        column_spec("info", "text"),
    ],
}


# -- ScriptedCursor ----------------------------------------------------------------------------------


def test_scripted_cursor_returns_scripted_rows() -> None:
    cursor = ScriptedCursor({"information_schema.tables": [("title",), ("movie_info",)]})

    cursor.execute("SELECT table_name FROM information_schema.tables WHERE table_schema = %s", ["public"])

    assert cursor.fetchall() == [("title",), ("movie_info",)]


def test_scripted_cursor_records_sql_and_parameters() -> None:
    cursor = ScriptedCursor({"information_schema.columns": []})

    cursor.execute("SELECT column_name FROM information_schema.columns WHERE table_name = %s", ["title"])

    assert len(cursor.calls) == 1
    assert cursor.calls[0].parameters == ("title",)
    assert cursor.calls[0].mentions("information_schema.columns", "table_name")


def test_scripted_cursor_matching_ignores_formatting() -> None:
    """Whitespace and case must not affect matching -- the production SQL is textwrap.dedent-ed."""
    cursor = ScriptedCursor({"FROM information_schema.TABLES": [("title",)]})

    cursor.execute("SELECT   table_name\n  FROM\n    information_schema.tables\n")

    assert cursor.fetchall() == [("title",)]


def test_scripted_cursor_rejects_unscripted_query() -> None:
    """An unknown query must raise, not return empty.

    Returning [] would let a test assert on silence and pass for the wrong reason.
    """
    cursor = ScriptedCursor({"information_schema.tables": []})

    with pytest.raises(UnexpectedQueryError, match="unscripted query"):
        cursor.execute("SELECT * FROM pg_stats")


def test_scripted_cursor_non_strict_returns_empty() -> None:
    cursor = ScriptedCursor(strict=False)

    cursor.execute("SELECT * FROM anything")

    assert cursor.fetchall() == []


def test_scripted_cursor_fetchone_returns_first_row() -> None:
    cursor = ScriptedCursor({"pg_class": [("42",), ("43",)]})

    cursor.execute("SELECT reltuples FROM pg_class")

    assert cursor.fetchone() == ("42",)


def test_scripted_cursor_fetchone_on_empty_result() -> None:
    cursor = ScriptedCursor({"pg_class": []})

    cursor.execute("SELECT reltuples FROM pg_class")

    assert cursor.fetchone() is None


# -- StaticSchema ------------------------------------------------------------------------------------


@pytest.fixture
def schema() -> StaticSchema:
    return StaticSchema(SCHEMA_SPEC)


def test_static_schema_reports_tables(schema: StaticSchema) -> None:
    assert schema.tables() == {TITLE, MOVIE_INFO}


def test_static_schema_reports_columns(schema: StaticSchema) -> None:
    assert sorted(col.name for col in schema.columns(TITLE)) == ["id", "production_year", "title"]


@pytest.mark.parametrize(
    ("table", "column", "expected"),
    [
        ("title", "id", True),
        ("title", "production_year", False),
        ("movie_info", "id", True),
        ("movie_info", "movie_id", False),
    ],
)
def test_static_schema_primary_keys(schema: StaticSchema, table: str, column: str, expected: bool) -> None:
    assert schema.is_primary_key(ColumnReference(column, TableReference(table))) is expected


def test_static_schema_has_index_composes_production_logic(schema: StaticSchema) -> None:
    """`has_index` is *not* overridden -- it is the real implementation over the two leaf lookups.

    This is the point of the double: production composition stays under test instead of being replaced.
    """
    assert "has_index" not in vars(StaticSchema), "has_index must stay inherited from DatabaseSchema"

    assert schema.has_index(ColumnReference("id", TITLE)) is True  # via is_primary_key
    assert schema.has_index(ColumnReference("production_year", TITLE)) is True  # via has_secondary_index
    assert schema.has_index(ColumnReference("title", TITLE)) is False


def test_static_schema_lookup_column_composes_production_logic(schema: StaticSchema) -> None:
    """`lookup_column` is inherited and resolves through `columns()`."""
    assert "lookup_column" not in vars(StaticSchema)

    resolved = schema.lookup_column(ColumnReference("production_year", TITLE), [TITLE, MOVIE_INFO])

    assert resolved == TITLE


def test_base_lookup_column_resolves_an_unqualified_column(schema: StaticSchema) -> None:
    """`lookup_column` fetches the first matching table from the leaf lookups."""
    assert schema.lookup_column("production_year", [TITLE, MOVIE_INFO]) is TITLE
    assert schema.lookup_column(ColumnReference("production_year"), [TITLE, MOVIE_INFO]) is TITLE
    assert schema.lookup_column("production_year", [TITLE], expect_match=True) is TITLE


def test_static_schema_table_info_composes_production_logic(schema: StaticSchema) -> None:
    """``schema[table]`` builds a `TableInfo` purely from the leaf lookups."""
    info = schema[TITLE]

    assert info.table == TITLE
    assert info.primary_key == PrimaryKeyRef([ColumnReference("id", TITLE)])
    assert {col.column.name for col in info.columns} == {"id", "title", "production_year"}
    assert info["title"].datatype == "text"
    assert info["id"].nullable is False


def test_static_schema_foreign_keys(schema: StaticSchema) -> None:
    fks = schema.foreign_keys_on(ColumnReference("movie_id", MOVIE_INFO))

    assert fks == {ForeignKeyRef([ColumnReference("movie_id", MOVIE_INFO)], [ColumnReference("id", TITLE)])}


def test_static_schema_is_iterable_as_mapping(schema: StaticSchema) -> None:
    """`DatabaseSchema` is a Mapping; iteration must yield the tables."""
    assert set(schema) == {TITLE, MOVIE_INFO}
    assert len(schema) == 2


def test_static_schema_rejects_unknown_column(schema: StaticSchema) -> None:
    with pytest.raises(KeyError):
        schema.datatype(ColumnReference("nonexistent", TITLE))


# -- StaticSchema: composite keys and constraints ---------------------------------------------------

PARENT = TableReference("parent")
CHILD = TableReference("child")
P_A, P_B, P_C = (ColumnReference(name, PARENT) for name in "abc")
C_A, C_B = ColumnReference("a", CHILD), ColumnReference("b", CHILD)


@pytest.fixture
def composite() -> StaticSchema:
    return StaticSchema(
        {
            "parent": ["a", "b", column_spec("c", unique=True), "d"],
            "child": ["a", "b", column_spec("k", indexed=True), column_spec("r", references="parent.c")],
        },
        primary_keys={"parent": ["b", "a"]},
        foreign_keys=[foreign_key_spec("child", ["a", "b"], "parent", ["a", "b"])],
        unique={"parent": [["a", "d"]]},
        indexes={"child": {"child_a_k_idx": ["a", "k"]}},
    )


def test_static_schema_primary_key_keeps_the_declared_column_order(composite: StaticSchema) -> None:
    pk = composite.primary_key_of(PARENT)

    assert pk == PrimaryKeyRef([P_B, P_A])
    assert pk is not None and [col.name for col in pk.columns] == ["b", "a"]


def test_static_schema_composes_a_primary_key_from_several_flagged_columns() -> None:
    schema = StaticSchema({"t": [column_spec("x", primary_key=True), column_spec("y", primary_key=True), "z"]})

    t = TableReference("t")
    assert schema.primary_key_of(t) == PrimaryKeyRef([ColumnReference("x", t), ColumnReference("y", t)])
    assert schema.is_primary_key(ColumnReference("x", t)) is False


def test_static_schema_table_without_primary_key_has_none(composite: StaticSchema) -> None:
    assert composite.primary_key_of(CHILD) is None


@pytest.mark.parametrize(
    ("columns", "expected"),
    [
        ([P_A, P_B], True),
        ([P_B, P_A], True),
        ([P_A], False),
        ([P_A, P_B, P_C], False),
    ],
    ids=["definition-order", "reversed", "part-of-the-key", "superset"],
)
def test_static_schema_is_primary_key_for_composite_keys(composite: StaticSchema, columns, expected: bool) -> None:
    assert composite.is_primary_key(columns) is expected


def test_static_schema_is_primary_key_ignores_table_aliases(composite: StaticSchema) -> None:
    aliased = TableReference("parent", "p")

    assert composite.is_primary_key([ColumnReference("a", aliased), ColumnReference("b", aliased)]) is True


def test_static_schema_unique_constraints(composite: StaticSchema) -> None:
    assert composite.is_unique(P_C) is True
    assert composite.is_unique([ColumnReference("d", PARENT), P_A]) is True
    assert composite.is_unique(P_A) is False
    assert composite.is_unique([P_A, P_B]) is False  # the primary key is not a UNIQUE constraint


def test_static_schema_foreign_keys_include_composite_and_column_level_keys(composite: StaticSchema) -> None:
    assert composite.all_foreign_keys_on(CHILD) == {
        ForeignKeyRef([C_A, C_B], [P_A, P_B]),
        ForeignKeyRef([ColumnReference("r", CHILD)], [P_C]),
    }
    assert composite.all_foreign_keys_on(PARENT) == set()


def test_static_schema_composes_foreign_keys_on_columns_from_the_table_level_lookup(composite: StaticSchema) -> None:
    assert "foreign_keys_on" not in vars(StaticSchema), "foreign_keys_on must stay inherited from DatabaseSchema"

    assert composite.foreign_keys_on([C_B, C_A]) == {ForeignKeyRef([C_A, C_B], [P_A, P_B])}
    assert composite.foreign_keys_on(C_A) == set()


def test_static_schema_indexes_are_named_like_postgres_does(composite: StaticSchema) -> None:
    assert dict(composite.all_indexes_on(PARENT)) == {
        "parent_pkey": {P_A, P_B},
        "parent_c_key": {P_C},
        "parent_a_d_key": {P_A, ColumnReference("d", PARENT)},
    }
    assert dict(composite.all_indexes_on(CHILD)) == {
        "child_k_idx": {ColumnReference("k", CHILD)},
        "child_a_k_idx": {C_A, ColumnReference("k", CHILD)},
    }


def test_static_schema_does_not_index_foreign_keys(composite: StaticSchema) -> None:
    assert composite.has_index(ColumnReference("r", CHILD)) is False
    assert composite.has_index([C_A, C_B]) is False


def test_static_schema_composes_index_checks_from_the_table_level_lookup(composite: StaticSchema) -> None:
    for name in ("has_secondary_index", "indexes_on", "has_index"):
        assert name not in vars(StaticSchema), f"{name} must stay inherited from DatabaseSchema"

    k = ColumnReference("k", CHILD)
    assert composite.indexes_on([k, C_A]) == {"child_a_k_idx"}
    assert composite.has_secondary_index([P_B, P_A]) is False  # the primary key index does not count
    assert composite.has_secondary_index(P_C) is True  # backed by the UNIQUE constraint
    assert composite.has_index([P_B, P_A]) is True


@pytest.mark.parametrize(
    ("kwargs", "message"),
    [
        ({"primary_keys": {"nope": ["a"]}}, "Unknown table 'nope'"),
        ({"primary_keys": {"t": ["nope"]}}, "Unknown column 't.nope'"),
        ({"unique": {"t": [["a", "nope"]]}}, "Unknown column 't.nope'"),
        ({"indexes": {"t": {"i": ["nope"]}}}, "Unknown column 't.nope'"),
        ({"foreign_keys": [foreign_key_spec("t", ["a"], "u", ["nope"])]}, "Unknown column 'u.nope'"),
        ({"foreign_keys": [foreign_key_spec("t", ["a", "b"], "u", ["x"])]}, "mismatching column counts"),
    ],
    ids=["pk-table", "pk-column", "unique-column", "index-column", "fk-column", "fk-length"],
)
def test_static_schema_rejects_inconsistent_constraints(kwargs: dict, message: str) -> None:
    with pytest.raises(ValueError, match=message):
        StaticSchema({"t": ["a", "b"], "u": ["x"]}, **kwargs)


def test_static_schema_rejects_two_primary_key_definitions() -> None:
    with pytest.raises(ValueError, match="multiple primary key definitions"):
        StaticSchema({"t": [column_spec("a", primary_key=True), "b"]}, primary_keys={"t": ["b"]})


def test_static_schema_table_lookups_reject_unknown_tables(composite: StaticSchema) -> None:
    unknown = TableReference("unknown")

    for lookup in (composite.primary_key_of, composite.all_foreign_keys_on, composite.all_indexes_on):
        with pytest.raises(KeyError, match="Unknown table"):
            lookup(unknown)


def test_static_schema_multi_column_predicates_apply_the_production_argument_checks(composite: StaticSchema) -> None:
    for predicate in (composite.is_primary_key, composite.is_unique):
        with pytest.raises(ValueError, match="empty"):
            predicate([])
        with pytest.raises(ValueError, match="same table"):
            predicate([P_A, C_A])
        with pytest.raises(UnboundColumnError):
            predicate(ColumnReference("a"))
        with pytest.raises(KeyError):
            predicate(ColumnReference("nope", PARENT))


def test_static_schema_overrides_every_leaf_lookup_that_has_a_catalog_query() -> None:
    """A leaf lookup that is not overridden would reach for the cursor of the placeholder database and fail."""
    leaves = {
        "tables",
        "columns",
        "is_view",
        "is_primary_key",
        "primary_key_of",
        "is_unique",
        "all_foreign_keys_on",
        "all_indexes_on",
        "datatype",
        "is_nullable",
    }

    assert leaves <= set(vars(StaticSchema))


# -- FakeDatabase ------------------------------------------------------------------------------------


def test_fake_database_satisfies_the_abc() -> None:
    """Instantiating proves every abstract member is implemented.

    This is the reason the doubles are hand-written rather than MagicMock: adding an abstract method to
    `Database` breaks this test immediately instead of silently passing.
    """
    instance = FakeDatabase()

    assert isinstance(instance, Database)
    assert isinstance(instance.schema(), DatabaseSchema)


def test_fake_database_returns_scripted_result() -> None:
    database = FakeDatabase(results={"count(*)": [(42,)]})

    assert database.execute_query("SELECT count(*) FROM title") == 42


def test_fake_database_raw_result_is_unsimplified() -> None:
    database = FakeDatabase(results={"count(*)": [(42,)]})

    assert database.execute_query("SELECT count(*) FROM title", raw=True) == [(42,)]


def test_fake_database_records_executed_queries() -> None:
    database = FakeDatabase(results={"count(*)": [(1,)]})

    database.execute_query("SELECT  count(*)\nFROM title")

    assert database.executed_queries == ["select count(*) from title"]


def test_fake_database_rejects_unscripted_query() -> None:
    database = FakeDatabase()

    with pytest.raises(UnexpectedQueryError, match="unscripted query"):
        database.execute_query("SELECT * FROM title")


def test_fake_database_expect_result_is_chainable() -> None:
    database = FakeDatabase().expect_result("min(id)", [(1,)]).expect_result("max(id)", [(9,)])

    assert database.execute_query("SELECT min(id) FROM title") == 1
    assert database.execute_query("SELECT max(id) FROM title") == 9


def test_fake_database_tracks_lifecycle() -> None:
    database = FakeDatabase()

    database.reset_connection()
    database.reset_connection()
    database.close()

    assert database.reset_count == 2
    assert database.closed is True


def test_fake_database_exposes_static_schema() -> None:
    database = FakeDatabase(schema_spec=SCHEMA_SPEC)

    assert database.schema().tables() == {TITLE, MOVIE_INFO}


# -- FakeStatistics ----------------------------------------------------------------------------------


def test_fake_statistics_returns_configured_values() -> None:
    stats = FakeStatistics(
        total_rows={"title": 2_528_312},
        num_distinct={"title.production_year": 133},
        null_frac={"title.production_year": 0.02},
    )

    assert stats.total_rows(TITLE) == Cardinality(2_528_312)
    assert stats.num_distinct(ColumnReference("production_year", TITLE)) == 133
    assert stats.null_frac(ColumnReference("production_year", TITLE)) == pytest.approx(0.02)


def test_fake_statistics_returns_none_for_unknown() -> None:
    """*None* is the interface's "maintained but no value" signal, not an error."""
    stats = FakeStatistics()

    assert stats.total_rows(TITLE) is None
    assert stats.min_max(ColumnReference("id", TITLE)) is None
    assert stats.histogram(ColumnReference("id", TITLE)) is None


# -- pool isolation ----------------------------------------------------------------------------------


def test_fake_db_fixture_registers_current_database(fake_db: FakeDatabase) -> None:
    from postbound.db import DatabasePool

    assert DatabasePool.get_instance().current_database() is fake_db


def test_pool_is_clean_between_tests() -> None:
    """The previous test registered a database; it must not still be visible here.

    Pool leakage is not hypothetical -- it previously made `test_relalg` bind columns against whichever
    database an earlier module happened to leave behind.
    """
    from postbound.db import DatabasePool

    assert DatabasePool.get_instance().empty()
